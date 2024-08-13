# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 22:32:53 2023

@author: ljjjun
"""

import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import pandas as pd

from scipy.ndimage import gaussian_filter1d

import os

import numpy as np
from scipy.signal import stft
import matplotlib.pyplot as plt
from scipy import interpolate
import re
from tqdm import tqdm
from scipy import sparse
from scipy.sparse.linalg import spsolve
#---------------数据读取---------------------#

def read_Raman(file_path):

    # 使用内置的open函数打开文件
    with open(file_path, 'r') as file:
        # 读取文件内容到字符串
        text = file.read()
    #-------------------------------------#
    #--------数据格式转换-------------#


    # 找到数据开始和结束的标记
    start_marker = "#shining_data"
    end_marker = "#shining_end"
    # 截取标记之间的文本
    data_text = text[text.index(start_marker) + len(start_marker) : text.index(end_marker)]
    # 将文本按制表符分割，并转换为浮点数数组
    data_text2 = data_text.split('\n')[1:-1]

    data_list = []
    for x in data_text2:
        aa = [float(x.split('\t')[0]),float(x.split('\t')[1])]
        data_list.append(aa)

    data_array = np.array(data_list)# data_array[0]是波数 data_array[0]是intensity
    return data_array

def read_label(file_path):
    # 用于读取lable
    with open(file_path, 'r') as file:
        # 读取文件内容到字符串
        text = file.read()


    end_marker = "\n#shining_data"
    #print(text)
    # 截取标记之间的文本
    try:
        try:
            start_marker = "#Lable:"
            data_text_concentration = text[text.index(start_marker) + len(start_marker) : text.index(end_marker)]
        except:
            start_marker = "#label "
            data_text_concentration = text[text.index(start_marker) + len(start_marker) : text.index(end_marker)]
    except:
        start_marker = "#label:"
        data_text_concentration = text[text.index(start_marker) + len(start_marker): text.index(end_marker)]


    lable = float(data_text_concentration)
    return lable

def read_conc(file_path):

    with open(file_path, 'r') as file:
        # 读取文件内容到字符串
        text = file.read()

    #start_marker = "Concentration:"
    #end_marker = "}"
    # 截取标记之间的文本
    #data_text_concentration = text[text.index(start_marker) + len(start_marker) : text.index(end_marker)]
    #lable = float(data_text_concentration)
    pattern = r'(?<=Concentration:)(.*?)(?=})'
    match1 = re.search(pattern, text)
    result = float(match1[0])
    return result


#---------------y变换方法--------------#
def conclog10_Normalization(y,miny,maxy):
    y=conclog10y(y)
    y=concNormalization(y,miny,maxy)
    return y

def Inverse_conclog10_Normalization(y,miny,maxy):
    y=Inverse_concNormalization(y,miny,maxy)
    y=Inverse_conclog10y(y)
    return y
def conclog10y(y):
    #输入应该大于0
    y=np.log10(y)
    return y
def Inverse_conclog10y(y):
    y = 10**y;
    return y
def concNormalization(y,miny,maxy):
    y1=(y-miny)/(maxy-miny)

    return y1

def Inverse_concNormalization(y,miny,maxy):
    y1 = y*(maxy-miny)+miny
    return y1

#-----------光谱二维化方法----------------------------#
def Recurrence_plot(wn_range,data_array,normalization=True):
    #光谱递推图方法
    #wn_range  截取的信号区域 (波数)
    #data_array: read_Raman返回的数据  data_array[0]是波数 data_array[0]是intensity
    x,y_normal = Data_Interception(wn_range,data_array,normalization=normalization)
    Raman2Dresult = np.abs(y_normal[:, None] - y_normal[None, :])
    return Raman2Dresult

def Gramian_angular(wn_range,data_array,transformation='s',normalization=True):
    # 格拉米角场
    #wn_range  截取的信号区域 (波数)
    #data_array: read_Raman返回的数据  data_array[0]是波数 data_array[0]是intensity
    #transformation: 转换方式，s:夹角和的余弦(GASF);d:夹角差的正弦(GADF),默认s方式,放大拉曼峰影响

    """
    GASF和GADF是两种将时间序列转化为二维图像的方法，它们都是基于Gramian angular field的概念。它们的区别在于：
    GASF是Gramian angular summation field的缩写，它是通过计算两个时间序列之间的夹角余弦之和来得到的。
    它可以保留时间序列的周期性和趋势性信息，但也可能放大噪声和异常值的影响。

    GADF是Gramian angular difference field的缩写，它是通过计算两个时间序列之间的夹角差值之正弦来得到的。
    它可以突出时间序列的差异性和变化性信息，但也可能损失周期性和趋势性信息。
    """

    x,y_normal = Data_Interception(wn_range,data_array,normalization=normalization)

    # 转换为极坐标
    theta = np.arccos(y_normal) # 角度
    #r = t # 半径
    if transformation == 's':

        # 计算Gramian angular summation field (GASF)[^3^][3]
        Raman2Dresult = np.cos(theta[:, None] + theta[None, :])
    elif transformation == 'd':

        # 计算Gramian angular difference field (GADF)[^2^][2]
        Raman2Dresult = np.sin(theta[:, None] - theta[None, :])
    return Raman2Dresult

def Short_time_Fourier_transform(wn_range,data_array,fs =1000,window='hann',nperseg=50,normalization=True):
    # 短时傅里叶变换

    #wn_range  截取的信号区域 (波数)
    #data_array: read_Raman返回的数据  data_array[0]是波数 data_array[0]是intensity
    # fs采样频率,window：窗函数选择 nperseg :每个段的窗口长度

    x,y_normal = Data_Interception(wn_range,data_array,normalization=normalization)

    frequencies, times, Zxx = stft(y_normal, fs = fs,window = window, nperseg=nperseg,)
    # 创建插值函数
    interpolator = interpolate.interp2d(times, frequencies, np.abs(Zxx), kind='cubic')
    new_f = np.linspace(frequencies.min(), frequencies.max(), y_normal.shape[0])
    new_t = np.linspace(times.min(), times.max(), y_normal.shape[0])
    # 使用插值函数计算新的STFT矩阵
    Raman2Dresult = interpolator(new_t, new_f)
    return Raman2Dresult


def Markov_transition_field(wn_range,data_array,Q,normalization=True):
    # 马尔科夫跃迁场
    #wn_range  截取的信号区域 (波数)
    #data_array: read_Raman返回的数据  data_array[0]是波数 data_array[0]是intensity

    x,y_normal = Data_Interception(wn_range,data_array,normalization=normalization)

    # 划分状态空间
    #Q =len(y_noraml) # 状态空间个数
    #Q =500
    bins = np.linspace(0, 1+0.1, Q + 1) # 状态空间边界  +0.1避免y0归一化后的1值被划分到状态空间外。
    #labels = np.arange(1, Q + 1) # 状态空间标签
    q = np.digitize(y_normal, bins) - 1 # 将时间序列值分配到状态空间

    # 构建转移矩阵
    V = np.zeros((Q, Q)) # 初始化转移矩阵
    for i in range(len(q) - 1):
        V[q[i], q[i + 1]] += 1 # 统计相邻两个状态出现的次数

    # 计算概率
    Raman2Dresult =V /np.sum(V)

    # 计算概率
    #Raman2Dresult =V /np.sum(V)
    return Raman2Dresult


def Heat_map(wn_range,data_array,num_intervals=100,normalization=False):
    #直接将曲线映射到二维图像。
    num_intervals = 100
    x,y_normal = Data_Interception(wn_range,data_array,normalization=normalization)
    data_array2 = np.column_stack((x, y_normal))

    x_range = [np.min(data_array2[:,0]),np.max(data_array2[:,0])]
    y_range = [np.min(data_array2[:,1]),np.max(data_array2[:,1])]
    step_x = (x_range[1]-x_range[0])/num_intervals
    step_y = (y_range[1]-y_range[0])/num_intervals
    # 创建一个二维数组，大小等于区间数
    heatmap = np.zeros((num_intervals, num_intervals))

    # 对于序列中的每一个点
    for x, y in data_array2:
        # 找到其在二维图像中对应的格子
        j = int((x-x_range[0]-1e-6) / step_x)
        i = int((y-y_range[0]-1e-6 )/ step_y)

        # 将该格子的值设为1
        heatmap[i, j] = heatmap[i, j]+1
    for i in range(heatmap.shape[1]):
        index = np.argmax(heatmap[:,i])
        heatmap[0:index,i]=np.max(heatmap[:,i])
    return heatmap

def Data_Interception(wn_range,data_array,normalization=True):

    #截取数据并归一化
    x0 = data_array[:,0]
    y0 = data_array[:,1]
    index_low = np.absolute(x0-wn_range[0]).argmin()
    index_high = np.absolute(x0-wn_range[1]).argmin()
    #x = x0[index_low:index_high]
    y = y0[index_low:index_high]
    x = x0[index_low:index_high]
    # 归一化y序列



    if normalization:
        y_normal = (y - np.min(y)) / (np.max(y) - np.min(y))
    else:
        y_normal = y
    return x,y_normal



def baseline_als(y, lam=10**4,p=0.001, niter=3):
    """
    偏最小二乘法生成基线
    """
    #L = len(y)
    L= np.size(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    baseline = 0
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        baseline = spsolve(Z, w * y)
        w = p * (y > baseline) + (1 - p) * (y < baseline)
    r_baseline = y-baseline
    return baseline,r_baseline
import math

class WitecRamanProcessor:
    def __init__(self, input_dir, output_dir, target_wavelength=1339.6, lam=1e5, p=0.01, niter=3,start_wavelength=500,end_wavelength=2500,sigma=3):
        """
        本类用于处理Witec的共聚焦拉曼显微镜所采集的拉曼mapping数据
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.target_wavelength = target_wavelength
        self.lam = lam#去基线时的平滑参数，越大越跟随低频，越小越跟随高频/
        self.p = p#权重更新参数。接近1，基线更可能低于大部分数据点，适用于数据中有较多的高值离群点。接近 0：基线更可能高于大部分数据点，适用于数据中有较多的低值离群点。
        self.niter = niter#基线校正循环次数
        self.start_wavelength = start_wavelength
        self.end_wavelength = end_wavelength
        self.sigma = sigma  #去噪窗口宽度
        self.reback = True
        self.Denoise = True

    def find_closest_point_index(self, spectral_data, x1, y1):
        """
        从spectral_data数据中找出指定(x1,y1)最近的索引
        """
        # 初始化最小距离为正无穷大
        l = self.extract_positions(spectral_data)
        min_distance = float('inf')
        closest_index = -1

        # 遍历列表中的每个点
        for i, (x, y) in enumerate(l):
            # 计算当前点与 (x1, y1) 之间的距离
            distance = math.sqrt((x - x1) ** 2 + (y - y1) ** 2)

            # 如果当前距离小于最小距离，则更新最小距离和最近点的索引
            if distance < min_distance:
                min_distance = distance
                closest_index = i
        data_column = spectral_data.iloc[:, closest_index]
        return data_column
    def parse_filename(self, file_path):
        """
        解析文件名，获取相关信息。
        """
        file_name = os.path.basename(file_path)
        self.file_name0 = file_name
        file_info = re.match(r'^(.*?),(.*?),(-?\d+),(\d+(\.\d+)?),(\d+),(\d+)(?:_.*)?\.txt$', file_name)
        if file_info:
            process, analyte, concentration, integration_time, _, x_points, y_points = file_info.groups()
            concentration = f"10^{int(concentration)} M"
            x_points, y_points = int(x_points), int(y_points)
            return process, analyte, concentration, integration_time, x_points, y_points
        return None

    def read_data(self, file_path,delimiter='\t'):
        """
        读取高光谱数据。
        """
        data = pd.read_csv(file_path, delimiter=delimiter)
        wavelengths = data.iloc[:, 0]
        spectral_data = data.iloc[:, 1:]
        return wavelengths, spectral_data

    def baseline_als(self, y):
        """
        偏最小二乘法生成基线。
        """
        L = np.size(y)
        D = sparse.csc_matrix(np.diff(np.eye(L), 2))
        w = np.ones(L)
        baseline = 0
        for i in range(self.niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + self.lam * D.dot(D.transpose())
            baseline = spsolve(Z, w * y)
            w = self.p * (y > baseline) + (1 - self.p) * (y < baseline)
        r_baseline = y - baseline
        return baseline, r_baseline
    def denoise_spectral_data(self, spectral_data, sigma=1):
        """
        对光谱数据进行去噪。
        """
        denoised_data = pd.DataFrame(index=spectral_data.index, columns=spectral_data.columns)
        for column in spectral_data.columns:
            denoised_data[column] = gaussian_filter1d(spectral_data[column].to_numpy(), sigma=self.sigma)
        return denoised_data

    def apply_baseline_correction(self, spectral_data):
        """
        对所有光谱数据进行优化。
        基线校正。self.reback = True 基线校正
        和去噪 self.Denoise = True  去噪开。
        """
        reback = self.reback
        denoise = self.Denoise
        baselines = pd.DataFrame(index=spectral_data.index, columns=spectral_data.columns)
        r_baselines = pd.DataFrame(index=spectral_data.index, columns=spectral_data.columns)
        if reback:
            for column in spectral_data.columns:
                y = spectral_data[column].to_numpy()
                baseline, r_baseline = self.baseline_als(y)
                baselines[column] = baseline
                r_baselines[column] = r_baseline
        else:
            r_baselines = spectral_data
        if denoise:
            denoise_data = self.denoise_spectral_data(r_baselines)
        else:
            denoise_data = r_baselines
        return denoise_data
    def apply_baseline_correction_SpectralLabelingApp(self, spectral_data):
        """
        SpectralLabelingApp类配合专用。
        对所有光谱数据进行优化。
        基线校正。self.reback = True 基线校正
        和去噪 self.Denoise = True  去噪开。
        """
        reback = self.reback
        denoise = self.Denoise
        baselines = pd.DataFrame(index=spectral_data.index, columns=spectral_data.columns)
        r_baselines = pd.DataFrame(index=spectral_data.index, columns=spectral_data.columns)
        if reback:
            for column in tqdm(spectral_data.columns,desc='Data preprocessing '):
                y = spectral_data[column].to_numpy()
                baseline, r_baseline = self.baseline_als(y)
                baselines[column] = baseline
                r_baselines[column] = r_baseline
        else:
            r_baselines = spectral_data
        if denoise:
            denoise_data = self.denoise_spectral_data(r_baselines)
        else:
            denoise_data = r_baselines
        return denoise_data
    def extract_positions(self, spectral_data):
        """
        从列标签中提取位置信息。
        """
        positions = []
        for col in spectral_data.columns:
            label = col.split('(')[-1].split(')')[0]
            x, y = map(int, label.split('/'))
            positions.append((x, y))
        return positions

    def create_position_to_col_map(self, spectral_data):
        """
        创建位置到列的映射。
        """
        positions = self.extract_positions(spectral_data)
        return dict(zip(positions, spectral_data.columns))

    def find_closest_wavelength_index(self, wavelengths):
        """
        查找最接近目标波长的索引。
        """
        return (np.abs(wavelengths - self.target_wavelength)).idxmin()

    def create_hotmap(self, r_baselines, positions, closest_wavelength_idx):
        """
        创建热图数据。
        """
        x_max = max([pos[0] for pos in positions])
        y_max = max([pos[1] for pos in positions])
        hotmap = np.zeros((x_max + 1, y_max + 1))

        for pos, col in self.create_position_to_col_map(r_baselines).items():
            x, y = pos
            hotmap[x, y] = r_baselines[col].iloc[closest_wavelength_idx]

        return hotmap

    def plot_and_save_hotmap(self, hotmap, wavelengths, closest_wavelength_idx, analyte, concentration, output_path):
        """
        绘制并保存热图。
        """
        plt.imshow(hotmap, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Intensity')
        plt.title(
            f'Hotmap for Wavelength {wavelengths.iloc[closest_wavelength_idx]} cm^-1\nAnalyte: {analyte}, Concentration: {concentration}')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.savefig(os.path.join(output_path, 'hotmap.png'))
        plt.close()

    def plot_stacked_spectra(self, denoised_r_baselines,wavelengths, top_10_cols, closest_wavelength_idx,analyte, concentration, output_path):
        """
        绘制并保存堆积曲线图。 还有问题
        """
        # 获取每个光谱的强度数据
        spectra = [denoised_r_baselines[col] for col in top_10_cols]

        # 转置数据，使得每行对应一个波长，每列对应一个光谱
        #spectra = np.array(spectra).T

        # 绘制堆积曲线图
        plt.figure()
        plt.stackplot(wavelengths, spectra, labels=top_10_cols)
        plt.title(
            f'Top 10 Spectra at Wavelength {wavelengths.iloc[closest_wavelength_idx]} cm^-1\nAnalyte: {analyte}, Concentration: {concentration}')
        plt.xlabel('Wavelength (cm^-1)')
        plt.ylabel('Intensity')
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(output_path, 'stacked_spectra.png'))
        plt.close()
    def plot_and_save_top_spectra(self, r_baselines, wavelengths, top_10_cols, closest_wavelength_idx, analyte,
                                  concentration, output_path):
        """
        绘制并保存最强的十个光谱图。
        """
        # 指定绘制的波长范围，例如 500 到 600 nm
        start_wavelength = self.start_wavelength
        end_wavelength = self.end_wavelength

        # 找到波长范围内的索引
        start_idx = np.argmax(wavelengths >= start_wavelength)
        end_idx = np.argmax(wavelengths > end_wavelength)

        plt.figure()
        for col in top_10_cols:
            label = col.split('(')[-1].split(')')[0]
            x, y = map(int, label.split('/'))
            # 选择指定范围内的波长和对应的光谱强度
            selected_wavelengths = wavelengths[start_idx:end_idx]
            selected_intensity = r_baselines[col].iloc[start_idx:end_idx]
            plt.plot(selected_wavelengths, selected_intensity, label=f'({x}, {y})')
        plt.title(
            f'Top 10 Spectra at Wavelength {wavelengths.iloc[closest_wavelength_idx]} cm^-1\nAnalyte: {analyte}, Concentration: {concentration}')
        plt.xlabel('Wavelength (cm^-1)')
        plt.ylabel('Intensity')
        plt.legend()
        plt.savefig(os.path.join(output_path, 'top_10_spectra.png'))
        plt.close()

    def process_file(self, file_path,delimiter='\t'):
        """
        处理单个文件，生成结果并保存。
        """
        # 从文件名中解析信息
        parsed_info = self.parse_filename(file_path)
        if parsed_info is None:
            return
        process, analyte, concentration, integration_time, x_points, y_points = parsed_info

        # 读取数据
        wavelengths, spectral_data = self.read_data(file_path,delimiter)

        # 应用基线校正和去噪
        r_baselines = self.apply_baseline_correction(spectral_data)

        # 查找特定波长或最近的波长

        closest_wavelength_idx = self.find_closest_wavelength_index(wavelengths)

        # 创建热图数据
        hotmap = self.create_hotmap(r_baselines, self.extract_positions(spectral_data), closest_wavelength_idx)

        # 找到在该波长处光谱强度最强的列
        top_10_cols = r_baselines.iloc[closest_wavelength_idx].nlargest(10).index

        # 创建输出目录
        output_path = os.path.join(self.output_dir, os.path.splitext(os.path.basename(file_path))[0])
        os.makedirs(output_path, exist_ok=True)

        # 绘制并保存热图
        self.plot_and_save_hotmap(hotmap, wavelengths, closest_wavelength_idx, analyte, concentration, output_path)

        # 绘制并保存最强的十个光谱图
        self.plot_and_save_top_spectra(r_baselines, wavelengths, top_10_cols, closest_wavelength_idx, analyte,
                                       concentration, output_path)
        #self.plot_stacked_spectra(r_baselines, wavelengths, top_10_cols, closest_wavelength_idx, analyte,
        #
        #concentration, output_path)
    def process_directory(self,delimiter='\t'):
        """
        批量处理文件夹中的所有文件。目的是绘制去基线后mapping，并选指定波长前十光谱绘图保存。
        """
        files = [file for file in os.listdir(self.input_dir) if file.endswith('.txt')]
        for file_name in tqdm(files, desc="Processing directory", position=0):
            file_path = os.path.join(self.input_dir, file_name)
            self.process_file(file_path,delimiter=delimiter)

    def process_file_reBaseLine(self, file_path, delimiter='\t'):
        """
        处理单个文件，生成结果并保存。
        """
        # 从文件名中解析信息
        parsed_info = self.parse_filename(file_path)
        #print(parsed_info)
        if parsed_info is None:
            return
        process, analyte, concentration, integration_time, x_points, y_points = parsed_info

        # 读取数据
        wavelengths, spectral_data = self.read_data(file_path, delimiter)


        # 应用基线校正和去噪
        r_baselines = self.apply_baseline_correction(spectral_data)
        # 将spectral_data和wavelengths_df按列合并
        data_reBaseLine = pd.concat([wavelengths, r_baselines], axis=1)


        # 创建输出目录
        output_path = os.path.join(self.output_dir, os.path.splitext(os.path.basename(file_path))[0])
        #os.makedirs(output_path, exist_ok=True)
        data_reBaseLine.to_csv(output_path+'_reBaseLine.txt', sep='\t', index=False)

    def process_directory_reBaseLine(self):
        """
        批量处理文件夹中的所有文件，目的是去基线并保存。
        """
        files = [file for file in os.listdir(self.input_dir) if file.endswith('.txt')]
        for file_name in tqdm(files, desc="Processing directory", position=0):
            file_path = os.path.join(self.input_dir, file_name)
            self.process_file_reBaseLine(file_path)


class SpectralLabelingApp:
    def __init__(self,master, spectral_data, template_file, output_dir, wavelengths, mark_list,file_name0, start_wavelength=500, end_wavelength=1800):
        self.master = master

        self.start_wavelength = start_wavelength
        self.end_wavelength = end_wavelength
        self.wavelengths = wavelengths
        self.spectral_data = spectral_data
        self.template_file = template_file
        self.output_dir = output_dir
        self.current_index = 0
        self.labels = []
        self.mark_list = mark_list
        self.file_name0 = file_name0


        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack()

        self.label_frame = tk.Frame(master)
        self.label_frame.pack()

        self.button_yes = tk.Button(self.label_frame, text="   Yes   ", command=lambda: self.label_spectrum(1))
        self.button_yes.pack(side=tk.LEFT)

        self.button_no = tk.Button(self.label_frame, text="   No   ", command=lambda: self.label_spectrum(0))
        self.button_no.pack(side=tk.LEFT)

        self.button_uncertainty = tk.Button(self.label_frame, text="uncertainty", command=lambda: self.label_spectrum(2))
        self.button_uncertainty.pack(side=tk.LEFT)


        self.update_plot()




        #self.update_plot()
    def read_template(self,template_file):
        with open(template_file, 'r') as file:
            lines = file.readlines()
        end_index = lines.index('#shining_end')
        label_index = lines.index('#Lable:1\n')
        self.header1 = lines[:label_index]
        self.header2 = lines[label_index+1:end_index]
        self.footer = lines[end_index:]

    def save_spectrum(self, wavelengths, spectrum, label, output_dir, index):
        labeled_data_str = "\n".join(
            f"{wavelength}\t{intensity}" for wavelength, intensity in zip(wavelengths, spectrum))


        output_file = os.path.join(output_dir, f"s_{self.file_name0}_n{index + 1}_l{label}.txt")

        with open(output_file, 'w') as file:
            file.writelines(self.header1)
            file.write(f"#label:{label}\n")
            file.writelines(self.header2)
            file.write(f"{labeled_data_str}\n")
            file.writelines(self.footer)
    def update_plot(self):
        start_wavelength = self.start_wavelength
        end_wavelength = self.end_wavelength
        wavelengths = self.wavelengths

        # 找到波长范围内的索引
        start_idx = np.argmax(wavelengths >= start_wavelength)
        end_idx = np.argmax(wavelengths > end_wavelength)

        self.ax.clear()
        current_spectrum = self.spectral_data.iloc[:, self.current_index]
        self.column_label = self.spectral_data.iloc[:, self.current_index].name
        selected_wavelengths = wavelengths[start_idx:end_idx]
        selected_intensity = current_spectrum.iloc[start_idx:end_idx]
        self.ax.plot(selected_wavelengths, selected_intensity)
        # 在每个位置添加竖线
        for mark in self.mark_list:
            plt.axvline(x=mark, color='r', linestyle='--')
            plt.text(mark, plt.ylim()[1] *0.8, mark, rotation=90, ha='right')
        self.ax.set_title(f'Spectrum{self.column_label}, {self.current_index + 1}/{self.spectral_data.shape[1]}')
        self.ax.set_xlabel('Raman shift (cm^-1)')
        self.ax.set_ylabel('Intensity')
        self.canvas.draw()


    def label_spectrum(self, label):

        current_spectrum = self.spectral_data.iloc[:, self.current_index]
        #processor = WitecRamanProcessor(self.template_file)
        self.read_template(self.template_file)
        if label != 2: #如果是2，不保存。
            self.save_spectrum(self.wavelengths, current_spectrum, label, self.output_dir, self.current_index)

        self.labels.append(label)
        self.current_index += 1
        if self.current_index < self.spectral_data.shape[1]:
            self.update_plot()
        else:
            messagebox.showinfo("Info", "All spectra have been labeled.")
            self.master.quit()




if __name__ == "__main__" :


    # 定义文件路径
    file_path = 'D:/北工博士阶段/论文冲冲冲/LIG-EDAg-LINE-SERS/AI/conc_txt/0.5_xy0.txt'

    #result1 = read_conc(file_path)
    data_array = read_Raman(file_path)
    Raman2Dresult1 = data_array[:,1].reshape(1,data_array.shape[0])
    conc=read_conc(file_path)
    #wn_range = [0,1000]
    #x,y_normal = Data_Interception(wn_range,data_array,normalization=True)
    #data_array1 = np.column_stack((x, y_normal))

    wn_range = [0,1000]
    x,y_normal = Data_Interception(wn_range,data_array,normalization=False)
    baseline,baseline_removed_data= baseline_als(y_normal, lam=10**4,p=0.001,niter=3)  # 调用之前实现的基线漂移去除方法
    data_array2 = np.column_stack((x, baseline_removed_data))
    #Raman2Dresult = Recurrence_plot(wn_range,data_array)
    #wn_range = [500,1800] # 截取的信号区域
    #Raman2Dresult = Recurrence_plot(wn_range,data_array)
    #Raman2Dresult = Gramian_angular(wn_range,data_array,transformation='s')
    Raman2Dresult = Short_time_Fourier_transform(wn_range,data_array,fs =1044,window='hann',nperseg=10)
    # = Markov_transition_field(wn_range,data_array,Q=150)

    #x,y_normal = Data_Interception(wn_range,data_array)

    #plt.imshow(Raman2Dresult, cmap='rainbow', interpolation='none', origin='lower',extent=(x.min(),x.max(),x.min(),x.max()))
    #plt.colorbar(fraction=0.0457, pad=0.04)
    #plt.show()


    #x = data_array[:,0]


    #plt.plot(x,y_normal)
    #plt.plot(x,baseline)
    #plt.plot(x,baseline_removed_data)
    #plt.show()
    #plt.imshow(Raman2Dresult, cmap='rainbow', origin='lower',vmin=0,vmax=1)
    #plt.title(file_path)
    #plt.show()

    # 假设我们有一个一维序列数据
    #sequence = np.random.rand(100, 2)
    # 我们首先需要确定x和y轴的区间数

    num_intervals = 1000
    x_range = [np.min(data_array2[:,0]),np.max(data_array2[:,0])]
    y_range = [np.min(data_array2[:,1]),np.max(data_array2[:,1])]
    step_x = (x_range[1]-x_range[0])/num_intervals
    step_y = (y_range[1]-y_range[0])/num_intervals
    # 创建一个二维数组，大小等于区间数
    heatmap = np.zeros((num_intervals, num_intervals))

    # 对于序列中的每一个点
    for x, y in data_array2:
        # 找到其在二维图像中对应的格子
        j = int((x-x_range[0]-1e-6) / step_x)
        i = int((y-y_range[0]-1e-6 )/ step_y)

        # 将该格子的值设为1
        heatmap[i, j] = heatmap[i, j]+1
    for i in range(heatmap.shape[1]):
        index = np.argmax(heatmap[:,i])
        heatmap[0:index,i]=np.max(heatmap[:,i])

    # 使用热图将这个二维数组可视化
    plt.plot(data_array2[:,0],data_array2[:,1])
    plt.show()
    plt.imshow(heatmap, cmap='rainbow', origin='lower')
    plt.show()







