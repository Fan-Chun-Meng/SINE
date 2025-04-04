import math
import os
import torch
from torch.utils.data import Dataset
import numpy as np

dict_total = {}

def Data_preprocessing(value):
    for i in range(value['input_data'].shape[0]):
        ori_list = value['input_data'][i, :]  # 取出第 i 行
        # 去势：用10次多项式拟合
        a = np.polyfit(range(len(ori_list)), ori_list, 10)
        b = np.poly1d(a)
        c = b(range(len(ori_list)))
        value['input_data'][i, :] = ori_list - c  # 去势后的结果
    return value

def load_npy_files_from_directory(directory, data_input_size, station_num, aligning_P, channel_select, std_dev, normalized):
    # 获取文件夹中的所有文件名
    datas = np.load(directory, allow_pickle=True).tolist()
    return datas

class MyDataset(Dataset):
    def __init__(self, dataPath, dataVluePath, data_input_size, station_num, aligning_P, channel_select, normalized,
                 std_dev):
        global dict_total
        if len(dict_total) == 0:
            dict_total = load_npy_files_from_directory(dataVluePath, data_input_size, station_num, aligning_P, channel_select, std_dev, normalized)
            self.data_value = dict_total
        else:
            self.data_value = dict_total
        self.data = open(dataPath, 'r').readlines()
        self.data_input_size = data_input_size
        print(len(self))

    def __getitem__(self, index):
        # 22821311	42.3247	11.5912	2.5	2.7
        key = self.data[index].strip().split('\t')[0]
        value = self.data_value[key]
        #Normalize labels according to target range
        label = np.array(value['label'])
        value = Data_preprocessing(value)
        label = label.tolist()
        return torch.tensor(value['input_data'][:, :self.data_input_size].astype(np.float32)), torch.tensor(value['input_data_fft'][:, :self.data_input_size].astype(np.float32)), torch.tensor(value['edge_index']), torch.tensor(
            value['edge_weight_dis']), torch.tensor(value['edge_weight_similar']), torch.tensor(
            label), torch.tensor(value['station_list']), torch.tensor(value['P_point']), torch.tensor(value['P_point_label'][:, :self.data_input_size])

    def __len__(self):
        return len(self.data)
