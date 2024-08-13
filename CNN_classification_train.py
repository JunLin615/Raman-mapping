import numpy as np
import matplotlib.pyplot as plt
from pyts.image import RecurrencePlot

import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
import glob

from torch.utils import data
from PIL import Image
from torchvision import transforms
import Pretreatment as pr

import cv2
#--------训练集生成-------------#
from torch.utils.data import Dataset
import sys
from torch.utils.tensorboard import SummaryWriter
import torchvision

import os
import datetime
#import time
class MyDataset(data.Dataset):
    """
 
    """
    def __init__(self, root,method = 'Recurrence_plot',wn_range = [500,1800],num_classes = 2):
        #root里的文件地址，要用/而非\。
        #num_classes 分类类别总数
        self.imgs_path = glob.glob(root + '/*.txt')
        self.method = method
        self.wn_range = wn_range
        self.num_classes =num_classes
        #self.lables_path = glob.glob(root + '/lable/*.txt')


    def __getitem__(self, index,):
        
        img_path = self.imgs_path[index]
        data_array = pr.read_Raman(img_path)
        method = self.method
        wn_range= self.wn_range
        #print(data_array.shape)
        
        if method == 'Recurrence_plot':
            Raman2Dresult = pr.Recurrence_plot(wn_range,data_array)
        elif method == 'Gramian_angular':
            Raman2Dresult = pr.Gramian_angular(wn_range,data_array,transformation='s')
        elif method == 'Short_time_Fourier_transform':
            Raman2Dresult = pr.Short_time_Fourier_transform(wn_range,data_array,
                                                         fs =1044,window='hann',nperseg=10)
        elif method == 'Markov_transition_field':
            Raman2Dresult = pr.Markov_transition_field(wn_range,data_array,Q=150)
        #pil_img = Image.open(img_path).convert('L')  # 转换为灰度图
        #lable_path = self.lables_path[index]
        Raman2Dresult = cv2.resize(Raman2Dresult, (100,100))
        Raman2Dresult = Raman2Dresult.reshape(1,Raman2Dresult.shape[0],Raman2Dresult.shape[1])
        
        label = pr.read_label(img_path)
        Raman2Dresult_tensor = torch.from_numpy(Raman2Dresult)
        
        labels = torch.tensor(int(label))

        # 计算类别总数
        #num_classes = 2
        
        # 将整数标签转换为one-hot编码
        onehot_labels = torch.eye(self.num_classes)[labels]
        
        #label_tensor = torch.Tensor([label])
        return Raman2Dresult_tensor.float(),onehot_labels,img_path

    def __len__(self):
        return len(self.imgs_path)
    def inputs(self,data_array):
        method = self.method
        wn_range= self.wn_range
        if method == 'Recurrence_plot':
            Raman2Dresult = pr.Recurrence_plot(wn_range, data_array)
        elif method == 'Gramian_angular':
            Raman2Dresult = pr.Gramian_angular(wn_range, data_array, transformation='s')
        elif method == 'Short_time_Fourier_transform':
            Raman2Dresult = pr.Short_time_Fourier_transform(wn_range, data_array,
                                                            fs=1044, window='hann', nperseg=10)
        elif method == 'Markov_transition_field':
            Raman2Dresult = pr.Markov_transition_field(wn_range, data_array, Q=150)
        # pil_img = Image.open(img_path).convert('L')  # 转换为灰度图
        # lable_path = self.lables_path[index]
        Raman2Dresult = cv2.resize(Raman2Dresult, (100, 100))
        Raman2Dresult = Raman2Dresult.reshape(1, Raman2Dresult.shape[0], Raman2Dresult.shape[1])
        Raman2Dresult_tensor = torch.from_numpy(Raman2Dresult)



        # label_tensor = torch.Tensor([label])
        return Raman2Dresult_tensor.float()




#-------------------------------------#

"""
功能：光谱二维化+2分类CNN的训练程序
可以用Manually labeled witec spectra.py手动从witec采集的mapping数据中进行人工识别打标签制作训练数据。
"""



#--------CNN模型定义-------------#
class Net(nn.Module):
    """
    这个网络包含两个卷积层和三个全连接层。ReLU激活函数用于非线性变换，
    MaxPooling用于下采样。
    """
    
    def __init__(self):
        super(Net, self).__init__()
        
        self.covn1 = nn.Sequential(  # 构建Sequential，属于特殊的module，类似于forward前向传播函数，同样的方式调用执行
            nn.Conv2d(1, 32, 5, 1, 2),  # 输入通道为1（灰度图），输出通道为32(32个特征图)，卷积核大小为5
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.covn2 = nn.Sequential(  # 构建Sequential，属于特殊的module，类似于forward前向传播函数，同样的方式调用执行
            nn.Conv2d(32, 32, 4, 1, 2),  # 输入通道为1（灰度图），输出通道为32，卷积核大小为4
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),#池化2x2区域
        )
        
                
        # 全连接层，将16*6*6个节点连接到120个节点上
        self.fc1 = nn.Linear(20000, 120)
        # 全连接层，将120个节点连接到84个节点上
        self.fc2 = nn.Linear(120, 84)
        # 全连接层，将84个节点连接到10个节点上（10类输出）
        self.fc3 = nn.Linear(84, 2)


    def forward(self, x):
        
        x = self.covn1(x)
        x = self.covn2(x)
        
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # 所有维度，除了批量维度
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__ == '__main__':
    root = 'data/数据总和1标定前/手工打标签/240713_778/mix/'
    #数据集在root/data/raman/
    PATH = 'model_save/'
    train_ratio = 0.8
    test_ratio = 1 - train_ratio
    batch_size = 16
    num_epochs = 1000 #训练次数


    #trainset = MyDataset(root,'Recurrence_plot',[500,1800])
    dataset = MyDataset(root,'Gramian_angular',[500,1800])
    # 数据集划分比例


    # 计算训练集和测试集的大小
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size


    # 使用 random_split 划分数据集
    trainset, testset = random_split(dataset, [train_size, test_size])
    #trainset = MyDataset(root,'Gramian_angular',[500,1800])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    net = Net()

    print(net)

    # 定义损失函数，这里我们使用交叉熵损失函数
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.MultiLabelSoftMarginLoss()
    # 定义优化器，这里我们使用随机梯度下降优化器
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


    # 获取当前的时间戳
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    folder_path_totla = PATH+timestamp
    os.mkdir(folder_path_totla)
    writer = SummaryWriter(folder_path_totla+'/')

    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        net.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0):
            # 获取输入数据和标签
            inputs, labels, _ = data

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            _, labels_ = torch.max(labels, dim=1)
            total += labels.size(0)

            correct += (predicted == labels_ ).sum().item()

        train_loss /= len(trainloader)
        train_accuracy = 100 * correct / total

        # 记录训练集的损失和准确率
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_accuracy, epoch)

        # 验证模型
        net.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for i,data in enumerate(testloader, 0):
                inputs, labels, file_name_temp = data
            #for data, labels in test_loader:
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                _, labels_ = torch.max(labels, dim=1)
                total += labels.size(0)
                correct += (predicted == labels_).sum().item()
        test_loss /= len(testloader)
        test_accuracy = 100 * correct / total

        # 记录测试集的损失和准确率
        writer.add_scalar('Loss/Test', test_loss, epoch)
        writer.add_scalar('Accuracy/Test', test_accuracy, epoch)

    # 将网络结构添加到TensorBoard
    writer.add_graph(net, inputs)
    # 保存网络至时间戳文件夹， 名字为时间戳

    torch.save(net, folder_path_totla+'/'+timestamp+'.pth')


    #---------网络可视化----------------------#


    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    #trainset = datasets.MNIST('mnist_train', train=True, download=True, transform=transform)
    #trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    #model = torchvision.models.resnet50(False)
    # Have ResNet model take in grayscale rather than RGB
    #model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    #images, labels = next(iter(trainloader))
    #grid = torchvision.utils.make_grid(image_x_l)

    #writer.add_image('images', grid, 0)

    writer.close()






