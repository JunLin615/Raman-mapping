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
from torch.utils.data import Dataset
import sys
from torch.utils.tensorboard import SummaryWriter
import torchvision

import os
import datetime
#import time
from CNN_classification_train import Net,  MyDataset


"""
功能：光谱二维化+2分类CNN的调用程序，用于识别mapping中的点是否为目标物光谱。需要使用CNN_classification_train预先训练好光谱。
"""

def load_model(model_path):
    print(model_path)
    model = Net()
    model = torch.load(model_path)
    model.eval()  # 切换到评估模式
    return model



def CNN_reg(wavelengths,spectral_data,data_run,model):

    # 新建一个空的 DataFrame 用于存储计算结果，只有一行
    results_df = pd.DataFrame(index=['predicted', '0', '1'], columns=spectral_data.columns)
    # 使用tqdm添加进度条
    good_list = []
    good_list_xy = []
    # 确保模型在GPU上
    model = model.cuda() if torch.cuda.is_available() else model.cpu()

    # 确保输入数据在同一个设备上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for column in tqdm(spectral_data.columns, desc="Processing Spectral Data"):
        y = spectral_data[column].to_numpy()
        x = wavelengths.to_numpy()
        data_array = np.array([x, y]).T
        inputs = data_run.inputs(data_array)
        inputs = inputs.to(device)
        with torch.no_grad():
            output = model(inputs.unsqueeze(0))
            predicted = torch.argmax(output, dim=1)
            results_df.loc['predicted', column] = predicted
            results_df.loc['0', column] = output[0, 0].item()
            results_df.loc['1', column] = output[0, 1].item()
            if predicted == 1:
                good_list.append(y)
                label = column.split('(')[-1].split(')')[0]
                x, y = map(int, label.split('/'))
                good_list_xy.append((x, y))
    return results_df,good_list,good_list_xy



def make_hotmap(wavelengths,spectral_data,target_wavelength,results_df):

    # 查找特定波数或最近的波数
    #print(type(target_wavelength))
    #print(type(wavelengths))
    #print(wavelengths)
    closest_wavelength_idx = (np.abs(wavelengths - target_wavelength)).idxmin()
    # 提取列标签中的位置信息
    positions = []
    for col in tqdm(spectral_data.columns, desc="提取列标签信息"):
        label = col.split('(')[-1].split(')')[0]
        x, y = map(int, label.split('/'))
        positions.append((x, y))
    # 获取唯一的x和y值
    x_vals = sorted(set([pos[0] for pos in positions]))
    y_vals = sorted(set([pos[1] for pos in positions]))

    # 创建映射字典
    x_map = {val: i for i, val in enumerate(x_vals)}
    y_map = {val: i for i, val in enumerate(y_vals)}

    # 创建空的热图
    #hotmap = np.zeros((len(x_vals), len(y_vals)))

    hotmap_raw = np.zeros((len(x_vals), len(y_vals)))
    hotmap_p = np.zeros((len(x_vals), len(y_vals)))
    hotmap_0 = np.zeros((len(x_vals), len(y_vals)))
    hotmap_1 = np.zeros((len(x_vals), len(y_vals)))

    # 创建位置到列的映射
    position_to_col = dict(zip(positions, spectral_data.columns))
    for pos, col in tqdm(position_to_col.items(), desc="重整为热图"):
        x, y = pos
        hotmap_p[x_map[x], y_map[y]] = results_df.loc['predicted', col]
        hotmap_0[x_map[x], y_map[y]] = results_df.loc['0', col]
        hotmap_1[x_map[x], y_map[y]] = results_df.loc['1', col]
        hotmap_raw[x_map[x], y_map[y]] = spectral_data[col].iloc[closest_wavelength_idx]



    return hotmap_raw, hotmap_p,hotmap_0,hotmap_1

if __name__ == '__main__':
    # 加载模型
    folder_path_total = 'model_save/20240713224702/'
    folder_name = os.path.basename(folder_path_total.rstrip('/'))
    # 确保目录存在
    os.makedirs(folder_path_total, exist_ok=True)
    model_path = os.path.join(folder_path_total, folder_name + '.pth')
    model = load_model(model_path)


    input_directory = 'data/数据总和1标定前/二次镀膜'
    #output_directory = 'model_save/20240713224702/data'
    output_directory = 'model_save/20240713224702/data/xizhi/'
    #data_path = 'data/数据总和1标定前/二次镀膜去基线无平滑lam1e3/a,R6G,-5,0.5,15,20_reBaseLine.txt'
    #data_path = 'data/数据总和1标定前/二次镀膜去基线无平滑lam1e3/a,R6G,-15,0.5,60,60_reBaseLine.txt'
    data_path = 'data/xizhi/240729rebaseline_lam1e3/a1-1,R6G,-7,0.5,60,60_reBaseLine.txt'

    processor = pr.WitecRamanProcessor(input_directory, output_directory)

    wavelengths, spectral_data = processor.read_data(data_path, delimiter='\t')

    data_run = MyDataset(folder_path_total, 'Gramian_angular', [500, 1800])


    results_df, good_list, good_list_xy = CNN_reg(wavelengths,spectral_data,data_run,model)
    # 创建mapping图像
    target_wavelength = 778  # 例如550nm
    hotmap_raw, hotmap_p,hotmap_0,hotmap_1 = make_hotmap(wavelengths,spectral_data,target_wavelength,results_df)


            #print(f'Predicted class: {predicted.item()}')
    sum_good = np.sum(good_list, axis=0)
    # 计算平均值
    ave_good_good = sum_good / len(good_list)
    ave_good_all = sum_good / spectral_data.shape[1]

    save_ave_good_good = np.column_stack((wavelengths.to_numpy(), ave_good_good))
    save_ave_good_all = np.column_stack((wavelengths.to_numpy(), ave_good_all))
    #print(os.path.basename(data_path))
    concentration = processor.parse_filename(os.path.basename(data_path))
    #print(concentration)
    raw_rb = spectral_data.mean(axis=1)
    save_raw_rb = np.column_stack((wavelengths.to_numpy(), raw_rb.to_numpy()))



    print(model)


    #"""
    # 绘制热图
    os.makedirs(output_directory+f'/mapping/{concentration[2]}', exist_ok=True)
    #概率判别结果的二值图
    plt.figure()
    plt.imshow(hotmap_raw.T, cmap='hot', interpolation='nearest',origin='lower')
    plt.colorbar(label='Intensity')
    plt.title(f'raw')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    # 保存图像为PNG格式
    plt.savefig(output_directory+f'/mapping/{concentration[2]}/'+'mapping_raw.png')
    #plt.show()

    #概率判别结果的二值图
    plt.figure()
    plt.imshow(hotmap_p.T, cmap='hot', interpolation='nearest',origin='lower',vmin=0, vmax=1)
    plt.colorbar(label='Intensity')
    plt.title(f'predicted')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    # 保存图像为PNG格式
    plt.savefig(output_directory+f'/mapping/{concentration[2]}/'+'mapping_p.png')
    #plt.show()

    #判别为0的概率分布图
    plt.figure()
    plt.imshow(hotmap_0.T, cmap='hot', interpolation='nearest',origin='lower',vmin=0, vmax=np.max(hotmap_0))
    plt.colorbar(label='Intensity')
    plt.title(f'0')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    # 保存图像为PNG格式
    plt.savefig(output_directory+f'/mapping/{concentration[2]}/'+'mapping_0.png')
    #plt.show()

    #判别为1的概率分布图
    plt.figure()
    plt.imshow(hotmap_1.T, cmap='hot', interpolation='nearest',origin='lower',vmin=0, vmax=np.max(hotmap_1))
    plt.colorbar(label='Intensity')
    plt.title(f'1')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    # 保存图像为PNG格式
    plt.savefig(output_directory+f'/mapping/{concentration[2]}/'+'mapping_1.png')
    #plt.show()


    """
    
    plt.figure()
    plt.plot(wavelengths.to_numpy(),ave_good_good)
    plt.title(f'ave good good')
    plt.xlabel('Raman shift (cm-1)')
    plt.ylabel('Intensity (arb.units)')
    plt.xlim(500, 1800)
    #plt.ylim(0, 100)
    plt.show()
    """

    #"""


    #"""

    #保存数据
    os.makedirs(output_directory+f'/raw_rb/', exist_ok=True)
    np.savetxt(output_directory+f'/raw_rb/{concentration[2]}.txt', save_raw_rb, fmt='%.10f', delimiter=',')
    os.makedirs(output_directory+f'/ave_good_good/', exist_ok=True)
    np.savetxt(output_directory+f'/ave_good_good/{concentration[2]}.txt', save_ave_good_good, fmt='%.10f', delimiter=',')
    os.makedirs(output_directory+f'/ave_good_all/', exist_ok=True)
    np.savetxt(output_directory+f'/ave_good_all/{concentration[2]}.txt', save_ave_good_all, fmt='%.10f', delimiter=',')
    #"""