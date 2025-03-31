import os
import json
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score
import seaborn as sns
from model import SupCon_Swin as create_model
from utils.utils import read_val_data

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

def compute_energy(logits):
    """计算能量分数 E(x) = -log(sum(exp(logits)))"""
    return -torch.log(torch.sum(torch.exp(logits), dim=1))

def save_roc_data(fpr, tpr, roc_auc, filename):
    """保存ROC数据到文件"""
    np.savez(filename, fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    print(f"ROC数据已保存到 {filename}")

def load_roc_data(filepaths):
    """加载多个模型的ROC数据"""
    roc_data = []
    for path in filepaths:
        data = np.load(path)
        roc_data.append({
            'fpr': data['fpr'],
            'tpr': data['tpr'],
            'roc_auc': data['roc_auc']
        })
    return roc_data

def plot_multiple_roc(roc_data_list, labels, title="原型距离法在不同开放度下的ROC曲线"):
    """绘制多个模型的ROC曲线"""
    plt.figure(figsize=(10, 8))
    for data, label in zip(roc_data_list, labels):
        plt.plot(data['fpr'], data['tpr'],
                 lw=3,
                 label=f'{label} (AUC = {data["roc_auc"][0]:.2f})')
    plt.plot([0, 1], [0, 1], [0, 1], [0, 1], color='navy', lw=4, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 数据路径
    data_path = "./datasets_val"  # 替换为你的数据路径
    val_images_path, val_images_label = read_val_data(data_path)

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

    # 类别索引文件
    json_path = './val_class_indices.json'
    assert os.path.exists(json_path), "file: '{}' does not exist.".format(json_path)
    with open(json_path, "r") as json_file:
        class_indict = json.load(json_file)
    num_known_classes = len(class_indict) - 1  # 假设最后一个类别是未知类
    # 设置能量分数阈值（需根据验证集调整）
    energy_threshold = -2.6 # 示例值，实际需要优化

    # 修改后的验证集预测部分
    all_preds = []
    all_labels = []
    all_energies = []
    all_features = []  # 存储head前的池化特征

    # 遍历验证集
    for img_path, label in zip(val_images_path, val_images_label):
        img = Image.open(img_path).convert('RGB')
        img = data_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output, x_pooled = model(img, return_pooled=True)
            energy = compute_energy(output)

            # 保存能量分数和特征
            all_energies.append(energy.item())
            all_features.append(x_pooled.cpu().numpy().squeeze())

            # 保存真实标签和预测结果
            all_labels.append(label)
            if energy.item() > energy_threshold:  # 根据阈值判断是否为未知类
                all_preds.append(num_known_classes)  # 未知类
            else:
                all_preds.append(torch.argmax(output).item())

    # 转换为 numpy 数组
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_energies = np.array(all_energies)
    all_features = np.array(all_features)  # [num_samples, dim_in]

    # 生成二分类标签（已知类: 1, 未知类: 0）
    binary_labels = (all_labels < num_known_classes).astype(int)

    # 计算 AUROC
    try:
        auroc = roc_auc_score(binary_labels, -all_energies)  # 能量越低，越可能是已知类
        print(f"AUROC: {auroc:.4f}")
    except ValueError as e:
        print(f"Error calculating AUROC: {e}")
        auroc = 0.0

    # 绘制 ROC 曲线
    fpr, tpr, thresholds = roc_curve(binary_labels, -all_energies)
    roc_auc = auc(fpr, tpr)

    # 保存ROC数据到文件
    save_path = "./roc_data/energy_osr_roc.npz"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_roc_data(fpr, tpr, np.array([roc_auc]), save_path)

    # # # 示例：加载多个模型的ROC数据并绘图
    # # # 假设我们有两个模型的ROC数据文件
    # roc_files = ["roc_data/distance_roc_100.npz", "roc_data/distance_roc_200.npz", "roc_data/distance_roc.npz", "roc_data/distance_roc_600.npz"]
    # roc_data_list = load_roc_data(roc_files)
    # plot_multiple_roc(roc_data_list, labels=["0.087", "0.161", "0.223", "0.364"])
    #
    # plt.figure(figsize=(8, 6))
    # plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic (ROC)')
    # plt.legend(loc="lower right")
    # plt.show()

    # 计算混淆矩阵（包括未知类）
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_indict))))

    # 归一化混淆矩阵（按行归一化）
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # 处理NaN

    # 计算评估指标（仅对已知类）
    known_mask = all_labels < len(class_indict)
    accuracy = accuracy_score(all_labels[known_mask], all_preds[known_mask])
    precision = precision_score(all_labels[known_mask], all_preds[known_mask], average='macro', zero_division=0)
    recall = recall_score(all_labels[known_mask], all_preds[known_mask], average='macro', zero_division=0)
    f1 = f1_score(all_labels[known_mask], all_preds[known_mask], average='macro', zero_division=0)

    print(f"Accuracy (all classes): {accuracy:.4f}")
    print(f"Precision (all classes): {precision:.4f}")
    print(f"Recall (all classes): {recall:.4f}")
    print(f"F1-Score (all classes): {f1:.4f}")

    # t-SNE可视化函数
    def plot_tsne(features, labels, title='t-SNE Visualization'):
        tsne = TSNE(n_components=2, random_state=42, perplexity=40)
        features_tsne = tsne.fit_transform(features)
        plt.figure(figsize=(12, 8))

        # 绘制已知类
        known_mask = labels < len(class_indict) - 1
        scatter_known = plt.scatter(features_tsne[known_mask, 0], features_tsne[known_mask, 1],
                                    c=labels[known_mask], cmap='tab20', alpha=0.6)

        # 绘制未知类（黑色）
        unknown_mask = labels == len(class_indict) - 1
        scatter_unknown = plt.scatter(features_tsne[unknown_mask, 0], features_tsne[unknown_mask, 1],
                                      c='black', label='未知类', alpha=0.6)

        plt.colorbar(scatter_known, ticks=range(len(class_indict) - 1))
        plt.title(title)
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        if unknown_mask.any():
            plt.legend()
        plt.savefig(title)
        plt.close()

    # 生成t-SNE图
    plot_tsne(all_features, all_labels, 't-SNE with True Labels')
    plot_tsne(all_features, all_preds, 't-SNE with Predicted Labels')
    # 混淆矩阵可视化
    plt.figure(figsize=(10, 8))
    class_names = [f'设备型号{i}' for i in range(len(class_indict)-1)] + ['未知类']
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
    main()