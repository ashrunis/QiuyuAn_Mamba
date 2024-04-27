import torch
import json
import random
import numpy as np
from torch import nn
from config import Config
from torch.utils.data import Subset
from collections import defaultdict
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from prettytable import PrettyTable


def balance(data_dir: str, num: int):
    dataset = datasets.ImageFolder(root=data_dir, transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]))
    
    # 记录索引
    flower_list = dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)
        
    # 获取每个类别的图片索引
    class_indices = {cls: [] for cls in range(len(dataset.classes))}
    for idx, (_, class_id) in enumerate(dataset.imgs):
        class_indices[class_id].append(idx)
    # 从每个类别中随机选择num张图片
    num_samples_per_class = num
    selected_indices = []
    for class_id, indices in class_indices.items():
        if len(indices) >= num_samples_per_class:
            selected_indices.extend(random.sample(indices, num_samples_per_class))
        else:
            # 如果一个类别的图片不足num张，则选取该类别的所有图片
            selected_indices.extend(indices)
    # 创建新的子数据集
    balance_dataset = Subset(dataset, selected_indices)
    return balance_dataset


def stratified_split(dataset, train_ratio):
    """
    Split a dataset into a stratified train and validation set.

    Args:
    dataset (Dataset): The dataset to split.
    train_ratio (float, optional): The proportion of the dataset to include in the train split. Defaults to 0.8.

    Returns:
    Tuple[Subset, Subset]: The train and validation datasets.
    """
    # 创建一个字典，用于存储每个类别的索引
    class_indices = defaultdict(list)

    # 遍历数据集，记录每个类别的索引
    for idx, (_, class_id) in enumerate(dataset):
        class_indices[class_id].append(idx)

    # 进行分层抽样
    train_indices = []
    val_indices = []

    for _, indices in class_indices.items():
        np.random.shuffle(indices)
        split = int(np.floor(train_ratio * len(indices)))
        train_indices += indices[:split]
        val_indices += indices[split:]

    # 打乱索引
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)

    # 创建子数据集
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    return train_dataset, val_dataset


def compute_mean_std(dataset):
    """
    计算数据集的均值和方差，用于在数据预处理中进行更好的标准化
    :param dataset:
    :return: mean, std
    :usage: mean, std = compute_mean_std(data_loader)
    """
    data_loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKER)

    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in data_loader:
        # 计算通道总和
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        # 计算通道平方的总和
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean**2)**0.5

    return mean, std


def load_dataset(data_dir: str):
    """
    载入并划分数据集
    :param data_dir:
    :return train_dataset, val_dataset, test_dataset:
    """
    # 平衡类别
    balance_dataset = balance(data_dir, Config.DATA_PER_CLASS)
    
    # 计算数据集的均值和方差
    mean, std = compute_mean_std(balance_dataset)
        
    # 分层划分为训练集、验证集
    train_dataset, val_dataset = stratified_split(balance_dataset, 0.7)
    val_dataset, test_dataset = stratified_split(val_dataset, 0.5)
    
    # 定义transforms
    train_dataset.dataset.transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.8),
        transforms.RandomRotation(10, expand=False, center=None),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_dataset.dataset.transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_dataset.dataset.transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    return train_dataset, val_dataset, test_dataset


class ConfusionMatrix(object):
    
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)

    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()