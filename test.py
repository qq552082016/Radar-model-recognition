import os
import json
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from model import SupCon_Swin as create_model
from utils.utils import read_split_data

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

def count_parameters(model):
    """
    统计模型的参数量。
    :param model: PyTorch 模型
    :return: 可训练参数数量和总参数数量
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return trainable_params, total_params

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 数据路径
    data_path = "./datasets"  # 替换为你的数据路径
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(data_path)

    # 图像预处理
    img_size = 192
    data_transform = transforms.Compose(
        [transforms.Resize(int(img_size * 1.14)),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    # 加载模型
    model = create_model(num_classes=9, head='mlp').to(device)
    model_weight_path = "./weights/model-1141_mlp_supcon.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()


    # 计算模型参数量
    trainable_params, total_params = count_parameters(model)
    print(f"可训练参数量: {trainable_params:,}")
    print(f"总参数量: {total_params:,}")

    # 类别索引文件
    json_path = './train_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    with open(json_path, "r") as json_file:
        class_indict = json.load(json_file)

        # 修改后的验证集预测部分
        all_preds = []
        all_labels = []
        all_features = []  # 存储head前的特征

        for img_path, label in zip(val_images_path, val_images_label):
            img = Image.open(img_path).convert('RGB')
            img = data_transform(img).unsqueeze(0).to(device)  # [1, C, H, W]

            with torch.no_grad():
                # 获取head前的特征（保持Tensor）
                feat, x_pooled = model(img, return_pooled=True)  # 获取池化后的特征

                # 使用完整模型获取预测结果
                # feat = model(img)  # [1, feat_dim]
                predict_cla = torch.argmax(feat).item()

            # 存储数据（转换为numpy）
            all_features.append(x_pooled.cpu().numpy().squeeze())  # [dim_in]
            all_preds.append(predict_cla)
            all_labels.append(label)

        # 转换为numpy数组
        all_features = np.array(all_features)  # [num_samples, dim_in]
        all_preds = np.array(all_preds)
        all_labels = np.array(val_images_label)

        # 计算混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)

        # 归一化混淆矩阵（按行归一化）
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # 计算评估指标
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")

        # 新增：t-SNE可视化模块
        def plot_tsne(features, labels, title='t-SNE Visualization'):
            # 执行t-SNE降维
            tsne = TSNE(n_components=2, random_state=42, perplexity=40)
            features_tsne = tsne.fit_transform(features)

            # 可视化设置
            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1],
                                  c=labels, cmap='tab20', alpha=0.6)
            plt.colorbar(scatter, ticks=range(len(class_indict)))
            plt.title(title)
            plt.xlabel('t-SNE Dimension 1')
            plt.ylabel('t-SNE Dimension 2')
            plt.savefig(title)  # 保存图片
            plt.close()

        # 执行可视化（真实标签和预测标签对比）
        plot_tsne(all_features, all_labels, 't-SNE with True Labels')
        plot_tsne(all_features, all_preds, 't-SNE with Predicted Labels')

        plt.figure(figsize=(10, 8))
        class_names = [f'设备型号{i}' for i in range(len(class_indict))]
        sns.heatmap(cm_normalized,
                    annot=True,
                    cmap="Blues",
                    xticklabels=class_names,  # x轴标签为预测类别名称
                    yticklabels=class_names)  # y轴标签为真实类别名称
        plt.title('混淆矩阵')
        plt.xlabel('预测标签')  # x轴标签说明
        plt.ylabel('真实标签')  # y轴标签说明
        plt.xticks(rotation=45)  # x轴标签旋转45度防重叠
        plt.yticks(rotation=0)  # y轴标签不旋转
        plt.tight_layout()  # 调整布局
        plt.show()
if __name__ == "__main__":
        main()