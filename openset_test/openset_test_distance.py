import os
import json
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score
import seaborn as sns
from model import SupCon_Swin as create_model
from utils.utils import read_val_data, read_split_data

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题


def calc_distance(feature, mean, eu_weight=0.5, distance_type='eucos'):
    """计算特征与类中心的距离"""
    if distance_type == 'eucos':
        euclidean = np.linalg.norm(feature - mean)
        cosine = np.dot(feature, mean) / (np.linalg.norm(feature) * np.linalg.norm(mean))
        return eu_weight * euclidean + (1 - eu_weight) * (1 - cosine)
    elif distance_type == 'euclidean':
        return np.linalg.norm(feature - mean)
    elif distance_type == 'cosine':
        return 1 - (np.dot(feature, mean) / (np.linalg.norm(feature) * np.linalg.norm(mean)))
    else:
        raise ValueError("Unsupported distance type")

def compute_class_means(features, labels, num_classes):
    """计算每个类别的特征均值"""
    means = {}
    for c in range(num_classes):
        class_features = features[labels == c]
        if len(class_features) > 0:
            means[c] = torch.mean(class_features, dim=0).cpu().numpy()
        else:
            means[c] = np.zeros(features.shape[-1])
    return means

# 添加ROC数据保存函数
def save_roc_data(fpr, tpr, roc_auc, filename):
    """保存ROC数据到文件"""
    np.savez(filename, fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    print(f"ROC数据已保存到 {filename}")

# 主函数
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 数据路径和划分
    data_path = "./datasets"  # 替换为你的数据路径
    train_images_path, train_images_label, _, _ = read_split_data(data_path)

    data_path = "./datasets_val"
    val_images_path, val_images_label = read_val_data(data_path)
    # 图像预处理
    img_size = 192
    data_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载模型
    model = create_model(num_classes=9, head='mlp').to(device)  # 假设9个已知类
    model_weight_path = "./weights/model-1141_mlp_supcon.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    # 加载类别索引
    json_path = './train_indices.json'
    with open(json_path, "r") as json_file:
        class_indict = json.load(json_file)
    num_classes = len(class_indict)  # 已知类的数量

    # 提取训练集特征
    train_features = []
    train_labels = []
    for img_path, label in zip(train_images_path, train_images_label):
        img = Image.open(img_path).convert('RGB')
        img = data_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            _, feat_3d = model(img, return_feat=True)
        pooled_feat = torch.mean(feat_3d.squeeze(0), dim=0).cpu().numpy()
        train_features.append(pooled_feat)
        train_labels.append(label)

    train_features = np.array(train_features)
    train_labels = np.array(train_labels)

    # 计算类中心和半径
    means = compute_class_means(torch.tensor(train_features), torch.tensor(train_labels), num_classes)
    radius = {}
    for c in range(num_classes):
        class_features = train_features[train_labels == c]
        if len(class_features) > 0:
            distances = [calc_distance(f, means[c], eu_weight=0.2) for f in class_features]
            radius[c] = np.percentile(distances, 50)  # 95%分位数作为半径
        else:
            radius[c] = 0.0


    # 测试集预测
    all_preds = []
    all_labels = []
    all_features = []  # 存储head前的池化特征
    all_min_distances = []  # 存储到最近类的距离

    for img_path, label in zip(val_images_path, val_images_label):
        img = Image.open(img_path).convert('RGB')
        img = data_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            _, feat_3d, x_pooled = model(img, return_feat=True, return_pooled=True)
        pooled_feat = torch.mean(feat_3d.squeeze(0), dim=0).cpu().numpy()

        # 计算到所有类的距离
        distances = []
        for c in range(num_classes - 1):
            dist = calc_distance(pooled_feat, means[c], eu_weight=0.1)
            distances.append(dist)
        distances = np.array(distances)

        # 找到最近的3个类
        sorted_indices = np.argsort(distances)
        top3_classes = sorted_indices[:3]

        # 判断是否属于已知类
        predict_cla = num_classes - 1  # 默认未知类
        for c in top3_classes:
            if distances[c] < radius[c]+0.01:
                predict_cla = c
                break

        # 保存结果
        all_preds.append(predict_cla)
        all_labels.append(label)
        all_features.append(x_pooled.cpu().numpy().squeeze())
        all_min_distances.append(np.min(distances))  # 保存到最近类的距离


    # 转换为numpy数组
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_min_distances = np.array(all_min_distances)
    all_features = np.array(all_features)

    # 生成二分类标签（已知类:1，未知类:0）
    binary_labels = (all_labels < num_classes).astype(int)

    # 计算 AUROC
    try:
        auroc = roc_auc_score(binary_labels, -all_min_distances)  # 能量越低，越可能是已知类
        print(f"AUROC: {auroc:.4f}")
    except ValueError as e:
        print(f"Error calculating AUROC: {e}")
        auroc = 0.0

    # 计算ROC数据
    fpr, tpr, _ = roc_curve(binary_labels, -all_min_distances)
    roc_auc = auc(fpr, tpr)

    # 保存ROC数据
    os.makedirs("roc_data", exist_ok=True)
    save_roc_data(fpr, tpr, np.array([roc_auc]), "roc_data/distance_roc_0.npz")


    # 计算混淆矩阵（包括未知类）
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_indict))))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # 处理NaN

    # 计算评估指标（仅对已知类）
    known_mask = all_labels < len(class_indict)-1
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
    class_names = [f'设备型号{i}' for i in range(len(class_indict) - 1)] + ['未知类']
    sns.heatmap(cm_normalized, annot=True, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()