
from __future__ import print_function

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.02, contrast_mode='all',
                 base_temperature=0.08, min_temp=0.05, decay=0.995, margin=6):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.min_temp = min_temp
        self.decay = decay
        self.margin = margin


    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))


        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask_neg = (1 - mask)  # 初始负样本掩码
        mask_neg = mask_neg.repeat(anchor_count, contrast_count)
        mask = mask.repeat(anchor_count, contrast_count)


        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask

        # 标识需要惩罚的高相似度负样本（相似度 > -self.margin）
        high_sim_neg_mask = (logits > -self.margin) * mask_neg.bool()  # [N, N]
        # ===== 核心修改：让supconloss忽略高相似度负样本 =====
        # 创建新的logits_mask（屏蔽高相似度负样本）
        adjusted_logits_mask = logits_mask.clone()
        adjusted_logits_mask[high_sim_neg_mask] = 0  # 高相似度负样本不参与log_softmax计算

        # # 计算负样本 Margin 惩罚项（相似度超过 margin 的部分）
        neg_similarity = logits.clone()
        # neg_similarity[mask_neg == 0] = -1e8  # 屏蔽非负样本位置
        margin_loss = torch.clamp(neg_similarity - (-self.margin), min=0).sum(dim=1)
        margin_loss = margin_loss.mean()

        # 重新计算exp_logits和log_prob（排除高相似度负样本）
        exp_logits = torch.exp(logits) * adjusted_logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))


        # 正样本损失计算
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1))

        # print("logits范围:", logits.min().item(), logits.max().item())  # 应限制在[-50, 50]
        # print("exp_logits.sum最小值:", exp_logits.sum(dim=1).min().item())  # 应 > 1e-8
        # print("mask.sum(1)最小值:", mask.sum(1).min().item())  # 应 > 0

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        self.temperature = max(self.temperature * self.decay, self.min_temp)
        # + 0.05 * margin_loss
        return loss
                #

# class SupConLoss(nn.Module):
#     def __init__(self, temperature=0.07, base_temperature=0.07):
#         super(SupConLoss, self).__init__()
#         self.temperature = temperature
#         self.base_temperature = base_temperature
#
#     def forward(self, features, labels):
#         """输入:
#             features: 3D特征 [B, L, C] (B=batch size, L=patch数量, C=特征维度)
#             labels: 标签 [B]
#         """
#         B, L, C = features.shape
#
#         # 1. 展平特征和扩展标签
#         features = features.reshape(B * L, C)  # [B*L, C]
#         features = F.normalize(features, dim=1)
#
#         # 扩展标签到每个空间位置 [B*L]
#         labels = labels.unsqueeze(1).expand(-1, L).reshape(-1)  # [B*L]
#
#         # 2. 计算相似度矩阵
#         similarity = torch.matmul(features, features.T)  # [B*L, B*L]
#         similarity /= self.temperature
#
#         # 3. 构建正样本掩码 (相同标签且非自身)
#         mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).bool()  # [B*L, B*L] (bool类型)
#         self_mask = torch.eye(B * L, dtype=torch.bool, device=features.device)  # 对角线掩码
#         mask_pos = mask & (~self_mask)  # 排除自身
#
#         # 4. 计算对比损失
#         logits_max, _ = torch.max(similarity, dim=1, keepdim=True)
#         logits = similarity - logits_max.detach()  # 数值稳定
#
#         exp_logits = torch.exp(logits)
#
#         # 正样本对数概率
#         log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))
#         mean_log_prob_pos = (mask_pos * log_prob).sum(dim=1) / (mask_pos.sum(dim=1))
#
#         # 损失计算
#         loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos.mean()
#         return loss