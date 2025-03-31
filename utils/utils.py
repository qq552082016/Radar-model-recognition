import os
import sys
import json
import pickle
import random

import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
from Supercon_loss import SupConLoss

def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 尝试加载已有的类别索引
    json_path = 'train_indices.json'
    if os.path.exists(json_path):
        with open(json_path, 'r') as json_file:
            class_indices = json.load(json_file)
        print("已加载现有的类别索引。")
    else:
        class_indices = {}
        print("未找到现有的类别索引，从头开始创建。")

    # 遍历文件夹，获取当前阶段的类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    flower_class.sort()

    # 更新类别索引：为新类别分配全局唯一索引
    for cla in flower_class:
        if cla not in class_indices:
            class_indices[cla] = len(class_indices)
            print(f"添加新类别: {cla}，索引为 {class_indices[cla]}")

    # 保存更新后的类别索引
    with open(json_path, 'w') as json_file:
        json.dump(class_indices, json_file, indent=4)
    print("已保存更新后的类别索引。")

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型

    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的全局索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("数据集中共找到 {} 张图片。".format(sum(every_class_num)))
    print("训练集包含 {} 张图片。".format(len(train_images_path)))
    print("验证集包含 {} 张图片。".format(len(val_images_path)))

    return train_images_path, train_images_label, val_images_path, val_images_label

def read_val_data(root: str):
    """
    从 root 文件夹下的 train 和 val 文件夹中读取训练集和验证集的图片路径及标签。

    参数:
        root (str): 数据集根目录，包含 train 和 val 文件夹。

    返回:
        train_images_path (list): 训练集图片路径列表。
        train_images_label (list): 训练集图片标签列表。
        val_images_path (list): 验证集图片路径列表。
        val_images_label (list): 验证集图片标签列表。
    """
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    val_dir = os.path.join(root)
    assert os.path.exists(val_dir), "val folder does not exist."


    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('val_class_indices.json', 'w') as json_file:
        json_file.write(json_str)


    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型

    # 定义函数用于读取指定文件夹中的图片路径和标签
    def load_images_and_labels(folder, class_indices):
        images_path = []
        images_label = []
        for cla in flower_class:
            cla_path = os.path.join(folder, cla)
            if os.path.exists(cla_path):  # 确保类别文件夹存在
                images = [os.path.join(cla_path, img) for img in os.listdir(cla_path)
                          if os.path.splitext(img)[-1] in supported]
                image_class = class_indices[cla]
                images_path.extend(images)
                images_label.extend([image_class] * len(images))
        return images_path, images_label

    # 加载训练集和验证集
    val_images_path, val_images_label = load_images_and_labels(val_dir, class_indices)

    print("{} images were found in the validation set.".format(len(val_images_path)))

    return val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    supcon_loss_fn = SupConLoss()  # 实例化你的对比损失
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, (images, labels) in enumerate(data_loader):
        images = images.to(device)
        labels = labels.to(device)
        sample_num += images.shape[0]
        # 前向传播（获取分类结果和特征）
        # pred = model(images)
        pred, feat_3d = model(images, return_feat=True)
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        # 计算两种损失
        ce_loss = loss_function(pred, labels)          # 分类损失
        supcon_loss_value = supcon_loss_fn(feat_3d, labels)  # 使用3D特征
        # total_loss = ce_loss + 0.2 * supcon_loss_value  # 加权和（权重可调）
        total_loss = ce_loss
        #
        # 反向传播
        total_loss.backward()
        accu_loss += total_loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        # if not torch.isfinite(total_loss):
        #     print('WARNING: non-finite loss, ending training ', total_loss)
        #     sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

# def train_one_epoch(model, optimizer, data_loader, device, epoch):
#     model.train()
#
#     loss_function = torch.nn.CrossEntropyLoss()
#     accu_loss = torch.zeros(1).to(device)  # 累计损失
#     accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
#     optimizer.zero_grad()
#
#     sample_num = 0
#     data_loader = tqdm(data_loader)
#     for step, data in enumerate(data_loader):
#         images, labels = data
#         sample_num += images.shape[0]
#
#         pred = model(images.to(device))
#         pred_classes = torch.max(pred, dim=1)[1]
#         accu_num += torch.eq(pred_classes, labels.to(device)).sum()
#
#         loss = loss_function(pred, labels.to(device))
#         # + SupConLoss(pred, labels.to(device))
#         loss.backward()
#         accu_loss += loss.detach()
#
#         data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
#                                                                                accu_loss.item() / (step + 1),
#                                                                                accu_num.item() / sample_num)
#
#         if not torch.isfinite(loss):
#             print('WARNING: non-finite loss, ending training ', loss)
#             sys.exit(1)
#
#         optimizer.step()
#         optimizer.zero_grad()
#
#     return accu_loss.item() / (step + 1), accu_num.item() / sample_num
@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
