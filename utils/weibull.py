
import torch
import torch.nn.functional as F
import numpy as np
import libmr

# 计算每个类的均值特征向量
def compute_class_means(features, labels, num_classes):
    means = []
    for c in range(num_classes):
        class_features = features[labels == c]
        if len(class_features) > 0:
            mean = class_features.mean(dim=0)
        else:
            mean = torch.zeros(features.shape[1:]).to(features.device)
        means.append(mean)
    return torch.stack(means, dim=0)  # 返回 [num_classes, L, C]

# 计算特征与类均值的相似度
def compute_similarities(features, means, labels):
    similarities = []
    for i, feature in enumerate(features):
        class_mean = means[labels[i]]
        similarity = F.cosine_similarity(feature.flatten(), class_mean.flatten(), dim=0)
        similarities.append(similarity.item())
    return np.array(similarities)

# 对相似度尾部拟合Weibull模型
def fit_weibull(means, dists, categories, tailsize=20):
    """
    为每个类别拟合 Weibull 模型（仅 EVT 建模）。

    Args:
        means (torch.Tensor): 每个类别的均值，形状 [num_classes, L, C]。
        dists (dict): 每个类别的距离数组字典，键为类别索引，值为 np.ndarray。
        categories (list): 类别列表。
        tailsize (int): 用于拟合的尾部数据大小，默认为 20。

    Returns:
        dict: 包含每个类别的均值和 Weibull 模型的字典。
    """
    weibull_models = {}
    for category in categories:
        class_dists = dists.get(category, np.array([]))
        if len(class_dists) > 0:
            mr = libmr.MR()
            tailtofit = np.sort(class_dists)[-tailsize:]
            mr.fit_high(tailtofit, len(tailtofit))
            weibull_models[category] = {
                'mean': means[category],
                'weibull': mr
            }
        else:
            weibull_models[category] = None
    return weibull_models
def compute_openmax_prob(scores, scores_u):
    """
    计算 OpenMax 概率，与参考程序一致。

    Args:
        scores (np.ndarray): 调整后的已知类得分，形状 [channels, nb_classes]。
        scores_u (np.ndarray): 未分配的得分，形状 [channels, nb_classes]。

    Returns:
        modified_scores (np.ndarray): OpenMax 概率，形状 [nb_classes + 1]。
    """
    prob_scores, prob_unknowns = [], []
    for s, su in zip(scores, scores_u):
        channel_scores = np.exp(s)  # 已知类得分
        channel_unknown = np.exp(np.sum(su))  # 未知类得分
        total_denom = np.sum(channel_scores) + channel_unknown
        prob_scores.append(channel_scores / total_denom)
        prob_unknowns.append(channel_unknown / total_denom)

    # 取所有通道的平均值
    scores = np.mean(prob_scores, axis=0)  # 形状 [nb_classes]
    unknowns = np.mean(prob_unknowns, axis=0)  # 标量
    modified_scores = np.append(scores, unknowns)  # 形状 [nb_classes + 1]
    return modified_scores


def calc_distance(feature, mean, eu_weight, distance_type='eucos'):
    """
    计算特征与均值之间的 eucos 距离。

    Args:
        feature (torch.Tensor): 输入特征，形状 [L, C]。
        mean (torch.Tensor): 类均值，形状 [L, C]。
        eu_weight (float): 欧氏距离的权重。
        distance_type (str): 距离类型，默认为 'eucos'。

    Returns:
        float: eucos 距离值。
    """
    if distance_type == 'eucos':
        # 转换为 NumPy 数组并展平
        feature = feature.cpu().numpy().flatten()
        mean = mean.cpu().numpy().flatten()

        # 计算欧氏距离
        euclidean_dist = np.linalg.norm(feature - mean)

        # 计算余弦距离
        cosine_dist = 1 - np.dot(feature, mean) / (np.linalg.norm(feature) * np.linalg.norm(mean))

        # 返回加权和
        return eu_weight * euclidean_dist + (1 - eu_weight) * cosine_dist
    else:
        raise ValueError(f"Unsupported distance type: {distance_type}")