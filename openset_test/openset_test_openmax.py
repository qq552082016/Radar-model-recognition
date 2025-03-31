import os
import json
import torch
import torch.nn.functional as F
from PIL import Image
import torch.nn as nn
from torchvision import models, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score
import seaborn as sns
from model import SupCon_Swin as create_model
from utils.utils import read_val_data, read_split_data
from utils.weibull import compute_class_means, compute_similarities, fit_weibull, compute_openmax_prob, calc_distance

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题


# OpenMax-like预测
def openmax_predict(weibull_models, categories, feature, logits, alpha=3, distance_type='eucos', eu_weight=0.1):
    """
    使用 OpenMax 机制计算输入样本的概率。

    Args:
        weibull_models (dict): 每个类别的 Weibull 模型字典，包含 'mean' 和 'weibull'。
        categories (list): 已知类别列表。
        feature (torch.Tensor): 输入特征，形状 [L, C]。
        logits (torch.Tensor): 模型输出的 logits，形状 [num_classes]。
        alpha (int): 用于调整 logits 的 top-alpha 参数，默认为 3。
        distance_type (str): 距离类型，默认为 'eucos'。
        eu_weight (float): 欧氏距离的权重，默认为 0.5。

    Returns:
        openmax_prob (np.ndarray): OpenMax 概率，形状 [num_classes + 1]，包含已知类和未知类的概率。
    """
    nb_classes = len(categories)
    logits = logits.cpu().numpy()  # 转换为 NumPy 数组，形状 [num_classes]

    # 获取 top-alpha logits 的索引
    ranked_list = np.argsort(logits)[::-1][:alpha]  # 降序排列，取前 alpha 个
    alpha_weights = [((alpha + 1) - i) / float(alpha) for i in range(1, alpha + 1)]
    omega = np.zeros(nb_classes)
    omega[ranked_list] = alpha_weights

    # 计算调整后的得分和未知类得分
    scores, scores_u = [], []
    for channel, logit_channel in enumerate([logits]):  # 单通道处理
        score_channel, score_channel_u = [], []
        for c, category in enumerate(categories):
            model = weibull_models.get(category, None)
            if model and model['weibull']:
                channel_dist = calc_distance(feature, model['mean'], eu_weight, distance_type)
                wscore = model['weibull'].w_score(channel_dist)  # Weibull 得分
                modified_score = logit_channel[c] * (1 - wscore * omega[c])
            else:
                modified_score = logit_channel[c]
            score_channel.append(modified_score)
            score_channel_u.append(logit_channel[c] - modified_score)
        scores.append(score_channel)
        scores_u.append(score_channel_u)

    scores = np.asarray(scores)  # 形状 [1, nb_classes]
    scores_u = np.asarray(scores_u)  # 形状 [1, nb_classes]

    # 计算 OpenMax 概率
    openmax_prob = compute_openmax_prob(scores, scores_u)

    return openmax_prob

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
    num_classes = len(class_indict)-1  # 已知类的数量

    # 提取训练集特征
    train_features = []
    train_labels = []
    train_logits = []
    for img_path, label in zip(train_images_path, train_images_label):
        img = Image.open(img_path).convert('RGB')
        img = data_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output, feat_3d = model(img, return_feat=True)  # feat_3d: [1, L, C]
        train_features.append(feat_3d.squeeze(0).to(device))
        train_labels.append(label)
        train_logits.append(output.squeeze(0))
    train_features = torch.stack(train_features)  # [N, L, C]
    train_labels = torch.tensor(train_labels)
    train_logits = torch.stack(train_logits)

    # 计算类均值
    global means  # 设为全局变量以在openmax_predict中使用
    means = compute_class_means(train_features, train_labels, num_classes)

    # 计算每个类别样本与类均值的 eucos 距离
    dists = {}
    for c in range(num_classes):
        class_features = train_features[train_labels == c]
        class_mean = means[c]
        class_dists = [calc_distance(feature, class_mean, eu_weight=0.1, distance_type='eucos') for feature in
                       class_features]
        dists[c] = np.array(class_dists)

    # 拟合 Weibull 模型（仅 EVT 建模）
    weibull_models = fit_weibull(means, dists, list(range(num_classes)))

    # 计算 OpenMax 分数并提取阈值
    max_known_probs = {c: [] for c in range(num_classes)}
    for i in range(len(train_features)):
        feature = train_features[i]
        logit = train_logits[i]
        label = train_labels[i].item()
        openmax_prob = openmax_predict(weibull_models, list(range(num_classes)), feature, logit)
        max_known_prob = np.max(openmax_prob[:-1])
        max_known_probs[label].append(max_known_prob)

    # 为每个类别计算 90% 分位数阈值
    thresholds = {}
    for c in range(num_classes):
        if max_known_probs[c]:
            thresholds[c] = np.percentile(max_known_probs[c], 50)
        else:
            thresholds[c] = 0.5  # 默认阈值


    json_path = './val_class_indices.json'
    with open(json_path, "r") as json_file:
        class_indict = json.load(json_file)
    num_classes = len(class_indict) - 1  # 已知类的数量

    # 测试集预测
    all_preds = []
    all_labels = []
    all_features = []  # 存储head前的池化特征

    all_max_known_probs = []  # 存储已知类最大概率

    for img_path, label in zip(val_images_path, val_images_label):
        img = Image.open(img_path).convert('RGB')
        img = data_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output, feat_3d, x_pooled = model(img, return_feat=True, return_pooled=True)
        feature = feat_3d.squeeze(0)
        logits = output.squeeze(0)

        # 修改部分：使用softmax代替openmax
        softmax_prob = torch.nn.functional.softmax(logits, dim=0).cpu().numpy()

        max_known_prob = np.max(softmax_prob)  # 直接取所有类别概率最大值
        predicted_category = np.argmax(softmax_prob)  # 预测类别

        threshold = thresholds[predicted_category] - 0.3

        if max_known_prob >= threshold:
            predict_cla = predicted_category
        else:
            predict_cla = num_classes  # 未知类

        # 存储数据
        all_preds.append(predict_cla)
        all_labels.append(label)
        all_max_known_probs.append(max_known_prob)
        all_features.append(x_pooled.cpu().numpy().squeeze())

    # 后续保持不变
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_max_known_probs = np.array(all_max_known_probs)
    all_features = np.array(all_features)

    binary_labels = (all_labels < num_classes).astype(int)
    # 计算 AUROC
    try:
        auroc = roc_auc_score(binary_labels, all_max_known_probs)
        print(f"AUROC: {auroc:.4f}")
    except ValueError as e:
        print(f"Error calculating AUROC: {e}")
        auroc = 0.0

    # 计算ROC数据
    fpr, tpr, _ = roc_curve(binary_labels, all_max_known_probs)
    roc_auc = auc(fpr, tpr)

    # 保存ROC数据
    os.makedirs("roc_data", exist_ok=True)
    save_roc_data(fpr, tpr, np.array([roc_auc]), "roc_data/openmax_roc.npz")


    # 计算混淆矩阵（包括未知类）
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_indict))))
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