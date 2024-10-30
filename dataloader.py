import torch
import tokenizers
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):  # 继承Dataset模块的Dataset类
    # 初始化定义，得到数据内容
    def __init__(self, enc_inputs_vec, dec_inputs, dec_outputs):
        super(MyDataset, self).__init__()
        # self.data_set = data_set  # 加载数据集
        # self.length = len(data_set)  # 数据集长度
        self.enc_inputs_vec = enc_inputs_vec
        # self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs


    # 返回数据集大小
    def __len__(self):
        return self.dec_inputs.shape[0]

    # 数据预处理
    def __getitem__(self, idx):  # index(或item)不能少，这个参数是来挑选某条数据的
        return self.enc_inputs_vec[idx], self.dec_inputs[idx], self.dec_outputs[idx]
        # return self.enc_inputs_vec[idx], self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]
