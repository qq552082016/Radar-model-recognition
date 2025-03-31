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
from model import SupCon_Swin_increment as create_model
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

def main(increment_phase):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 数据路径（根据增量阶段动态调整）
    data_path = f"./datasets_val"
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(data_path)

    # 图像预处理
    img_size = 192
    data_transform = transforms.Compose(
        [transforms.Resize(int(img_size * 1.14)),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # 加载模型（动态调整类别数和映射器数量）
    num_classes = 6 + increment_phase * 1  # 基础类别6个，每阶段增加1个类别（根据实际情况调整）
    num_mappers = increment_phase + 1  # 初始阶段有1个映射器，每阶段增加1个

    # 创建模型，传递类别数和映射器数量
    model = create_model(num_classes=num_classes).to(device)

    if increment_phase > 0:
        # 加载前一阶段模型权重
        prev_model_path = f"./weights/model-increment-{increment_phase}.pth"
        assert os.path.exists(prev_model_path), f"Previous model weights '{prev_model_path}' not exist."
        prev_weights = torch.load(prev_model_path, map_location=device)

        # 调整模型结构以匹配 start_increment - 1 阶段
        for i in range(increment_phase+1):
            if i > 0:  # 从第1阶段开始扩展映射器
                last_mapper = model.mappers[-1]
                new_mapper = nn.Linear(model.dim_in, model.feat_dim).to(device)
                new_mapper.load_state_dict(last_mapper.state_dict())
                model.mappers.append(new_mapper)

        # 调整分类器
        new_input_dim = len(model.mappers) * model.feat_dim
        model.last_fc = nn.Linear(new_input_dim, num_classes).to(device)
    # 加载对应阶段的模型权重
    model_weight_path = f"weights/model-increment-{increment_phase}.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    # 计算模型参数量
    trainable_params, total_params = count_parameters(model)
    print(f"可训练参数量: {trainable_params:,}")
    print(f"总参数量: {total_params:,}")

    # 类别索引文件
    json_path = './train_indices.json' #所有类别的索引文件
    assert os.path.exists(json_path), f"file: '{json_path}' does not exist."
    with open(json_path, "r") as json_file:
        class_indict = json.load(json_file)

    # 验证集预测
    all_preds = []
    all_labels = []
    all_features = []  # 存储head前的特征

    for img_path, label in zip(val_images_path, val_images_label):
        img = Image.open(img_path).convert('RGB')
        img = data_transform(img).unsqueeze(0).to(device)  # [1, C, H, W]

        with torch.no_grad():
            # 获取head前的特征
            feat, x_pooled = model(img, return_pooled=True)  # 获取池化后的特征
            # 获取预测结果
            predict_cla = torch.argmax(feat).item()

        # 存储数据
        all_features.append(x_pooled.cpu().numpy().squeeze())  # [dim_in]
        all_preds.append(predict_cla)
        all_labels.append(label)

    # 转换为numpy数组
    all_features = np.array(all_features)  # [num_samples, dim_in]
    all_preds = np.array(all_preds)
    all_labels = np.array(val_images_label)

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
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

    # t-SNE可视化模块
    def plot_tsne(features, labels, title='t-SNE Visualization'):
        tsne = TSNE(n_components=2, random_state=42, perplexity=40)
        features_tsne = tsne.fit_transform(features)

        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1],
                              c=labels, cmap='tab20', alpha=0.6)
        plt.colorbar(scatter, ticks=range(len(class_indict)))
        plt.title(title)
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.savefig(title)  # 保存图片
        plt.close()

    # 执行t-SNE可视化
    plot_tsne(all_features, all_labels, 't-SNE with True Labels')
    plot_tsne(all_features, all_preds, 't-SNE with Predicted Labels')

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    class_names = [f'设备型号{i}' for i in range(len(class_indict))]
    sns.heatmap(cm_normalized,
                annot=True,
                cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    increment_phase = 0
    main(increment_phase)