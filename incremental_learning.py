import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from collections import defaultdict
import argparse
import numpy as np
# 假设你有这些工具函数和类
from utils.my_dataset import MyDataSet  # 自定义数据集类
from utils.utils import read_split_data, train_one_epoch, evaluate  # 数据读取和训练函数
from model import SupCon_Swin_increment as create_model  # 你的模型
from PIL import Image
import pickle
import torch.cuda as cuda


def build_exemplar_set(model, device, seen_classes, train_images_path, train_images_label, num_exemplars=50):
    """
    构建范例集，为所有已见类别生成或更新范例。

    参数:
        model: 已训练的模型
        device: 计算设备 (如 torch.device('cuda'))
        seen_classes: 所有已见类别列表
        train_images_path: 训练图像路径列表
        train_images_label: 训练图像标签列表
        num_exemplars: 每个类别的范例数

    返回:
        exemplar_set: 包含所有已见类别范例的字典
    """
    model.eval()
    exemplar_set = defaultdict(list)

    # 数据预处理
    data_transform = transforms.Compose([
        transforms.RandomResizedCrop(192),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 为每个已见类别生成范例
    for cls in seen_classes:
        cls_images = [img for img, label in zip(train_images_path, train_images_label) if label == cls]
        if not cls_images:
            print(f"警告: 类别 {cls} 无可用图像")
            continue
        features = []
        # 提取特征
        for img_path in cls_images:
            img = Image.open(img_path).convert('RGB')
            img = data_transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                _, feat_3d = model(img, return_feat=True)
            feat_pooled = feat_3d.mean(dim=1).cpu().numpy().squeeze(0)
            features.append(feat_pooled)
            del img, feat_3d
            torch.cuda.empty_cache()
        features = np.array(features)
        # 计算特征中心并选择范例
        center = np.mean(features, axis=0)
        distances = np.linalg.norm(features - center, axis=1)
        indices = np.argsort(distances)[:min(num_exemplars, len(cls_images))].tolist()
        exemplar_set[cls] = [cls_images[i] for i in indices]

    return exemplar_set
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 创建保存目录
    if not os.path.exists("./weights"):
        os.makedirs("./weights")
    if not os.path.exists("./exemplars"):
        os.makedirs("./exemplars")

    # 增量数据集路径
    base_data_path = args.data_path  # 例如 './dataset'
    increments = [os.path.join(base_data_path, f'phase{i}') for i in range(7)]  # ['./dataset/phase0', ..., './dataset/phase6']
    start_increment = args.start_increment  # 从第几次增量开始

    # 初始化模型
    model = create_model(num_classes=args.num_classes, feat_dim=128).to(device)

    # 加载预训练权重（可选）
    if args.weights != "":
        assert os.path.exists(args.weights), f"weights file: '{args.weights}' not exist."
        weights_dict = torch.load(args.weights, map_location=device)
        model.load_state_dict(weights_dict, strict=False)

    # 记录已见类别、范例集和累积验证集
    seen_classes = []
    exemplar_set = defaultdict(list)
    cumulative_val_images_path = []  # 累积的验证集图像路径
    cumulative_val_images_label = []  # 累积的验证集标签

    # 数据预处理
    img_size = 192
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(int(img_size * 1.143)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # 如果从非第0阶段开始，加载前一阶段的模型和范例集，并调整模型结构
    if start_increment > 0:
        # 加载前一阶段模型权重
        prev_model_path = f"./weights/model-increment-{start_increment - 1}.pth"
        assert os.path.exists(prev_model_path), f"Previous model weights '{prev_model_path}' not exist."
        prev_weights = torch.load(prev_model_path, map_location=device)

        # 加载前一阶段的 exemplar_set
        exemplar_path = f"./exemplars/exemplar_set_{start_increment - 1}.pkl"
        if os.path.exists(exemplar_path):
            with open(exemplar_path, 'rb') as f:
                exemplar_set = pickle.load(f)
            print(f"已加载前一阶段范例集: {exemplar_path}")
        else:
            print(f"警告: 未找到增量 {start_increment - 1} 的范例集文件 '{exemplar_path}'，将不使用范例集继续训练。")

        # 重建 seen_classes
        seen_classes = list(exemplar_set.keys())

        # 调整模型结构以匹配 start_increment - 1 阶段
        for i in range(start_increment):
            if i > 0:  # 从第1阶段开始扩展映射器
                last_mapper = model.mappers[-1]
                new_mapper = nn.Linear(model.dim_in, model.feat_dim).to(device)
                new_mapper.load_state_dict(last_mapper.state_dict())
                model.mappers.append(new_mapper)

        # 调整分类器
        num_classes_so_far = len(seen_classes)
        new_input_dim = len(model.mappers) * model.feat_dim
        model.last_fc = nn.Linear(new_input_dim, num_classes_so_far).to(device)

        # 加载前一阶段权重
        model.load_state_dict(prev_weights, strict=False)

        # 初始化累积验证集
        for i in range(start_increment):
            _, _, val_images_path, val_images_label = read_split_data(increments[i])
            cumulative_val_images_path.extend(val_images_path)
            cumulative_val_images_label.extend(val_images_label)

    # 增量学习循环
    for k in range(start_increment, len(increments)):
        print(f"\n=== 开始增量阶段 {k} ===")

        # 读取当前阶段数据
        train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(increments[k])

        # 获取当前阶段的新类别
        new_classes = list(set(train_images_label) - set(seen_classes))
        seen_classes.extend(new_classes)
        seen_classes = list(set(seen_classes))

        # 将当前阶段的验证集添加到累积验证集中
        cumulative_val_images_path.extend(val_images_path)
        cumulative_val_images_label.extend(val_images_label)

        # 重置训练数据：当前阶段数据 + 上一阶段的 exemplar_set
        if k > 0:
            for param in model.encoder.parameters():
                param.requires_grad = False

            exemplar_images = []
            exemplar_labels = []
            for cls_idx, imgs in exemplar_set.items():
                exemplar_images.extend(imgs)  # 使用上一阶段的范例
                exemplar_labels.extend([cls_idx] * len(imgs))
            train_images_path = train_images_path + exemplar_images  # 当前数据 + 旧范例
            train_images_label = train_images_label + exemplar_labels

            last_mapper = model.mappers[-1]
            new_mapper = nn.Linear(model.dim_in, model.feat_dim).to(device)
            new_mapper.load_state_dict(last_mapper.state_dict())
            model.mappers.append(new_mapper)

            # 扩展分类器
            old_fc = model.last_fc
            num_old_classes = old_fc.out_features
            num_new_classes = len(new_classes)
            total_classes = num_old_classes + num_new_classes
            new_input_dim = len(model.mappers) * model.feat_dim

            new_fc = nn.Linear(new_input_dim, total_classes).to(device)
            with torch.no_grad():
                new_fc.weight[:num_old_classes, :old_fc.in_features] = old_fc.weight
                new_fc.bias[:num_old_classes] = old_fc.bias
            model.last_fc = new_fc

        # 构建训练数据集和验证数据集
        train_dataset = MyDataSet(images_path=train_images_path, images_class=train_images_label,
                                  transform=data_transform["train"])
        val_dataset = MyDataSet(images_path=cumulative_val_images_path, images_class=cumulative_val_images_label,
                                transform=data_transform["val"])
        batch_size = args.batch_size
        nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  pin_memory=True, num_workers=nw)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                pin_memory=True, num_workers=nw)
        # 设置优化器（简化示例）
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5E-2)
        # 初始化最佳的 val_loss 和 val_acc
        best_val_loss = float('inf')
        best_val_acc = 0.0

        # 训练当前阶段
        for epoch in range(args.epochs):
            train_loss, train_acc = train_one_epoch(model, optimizer, train_loader, device, epoch)
            val_loss, val_acc = evaluate(model, val_loader, device, epoch)
            print(
                f"[增量 {k} 轮次 {epoch}] 训练损失: {train_loss:.3f}, 训练精度: {train_acc:.3f}, 验证精度: {val_acc:.3f}")

            # 保存模型
            if val_loss < min(best_val_loss, 0.4) or val_acc > best_val_acc:
                best_val_loss = min(best_val_loss, val_loss)
                best_val_acc = max(best_val_acc, val_acc)
                torch.save(model.state_dict(), f"./weights/model-increment-{k}.pth")
                print(f"模型保存至 ./weights/model-increment-{k}.pth")

        # 训练结束后构建新的范例集
        exemplar_set = build_exemplar_set(model, device, seen_classes,
                                          train_images_path, train_images_label,
                                          num_exemplars=100)
        print(f"更新后的范例集类别数: {len(exemplar_set.keys())}")

        # 保存范例集
        exemplar_save_path = f"./exemplars/exemplar_set_{k}.pkl"
        with open(exemplar_save_path, 'wb') as f:
            pickle.dump(exemplar_set, f)
        print(f"范例集已保存至: {exemplar_save_path}")


        cuda.empty_cache()  # 释放显存缓存

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=6)  # 初始类别数
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--data-path', type=str, default='./datasets_increment')  # 数据集根目录
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--start-increment', type=int, default=0)  # 从第几次增量开始
    parser.add_argument('--device', default='cuda:0')
    opt = parser.parse_args()
    main(opt)