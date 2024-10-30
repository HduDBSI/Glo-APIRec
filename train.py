# import pickle
# import torch
import torch.optim as optim
# from torch import nn
# from torch.utils.data import DataLoader
from model import *
from dataloader import *
from utils import make_data, get_tr_data, cal_loss
from model import Transformer
# from get3vec import get3vec
# from self_attention.similarity import vector_matrix_similarity
import numpy as np
import time
import os
from tqdm import tqdm

CUDA_NUM = "cuda:2"

# device = torch.device(CUDA_NUM if torch.cuda.is_available() else "cpu")

# device = torch.device("cpu")
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device = CUDA_NUM if torch.cuda.is_available() else 'cpu'
# model = Mymodel().to(device)    #模型创建并加载到第一个GUP上

batch_size = 64     # 4
src_len = 20
# tgt_len = 6943
epochs = 8

if __name__ == "__main__":
    start_time = time.time()
    # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    # with open('data/comment_API.pkl', 'rb') as file:
    # with open('data/mini_comment_API64.pkl', 'rb') as file:
    #     comment_API_data = pickle.load(file)

    # word3vec_dic = get3vec(comment_API_data)
    # enc_inputs_vec, dec_inputs, dec_outputs = make_data()
    enc_inputs_vec, dec_inputs, dec_outputs = torch.load("data/tensor_list.pt")

    # enc_inputs_vec, enc_inputs, dec_inputs, dec_outputs = get_tr_data(comment_API_data, word3vec_dic)
    # enc_inputs_vec, dec_inputs, dec_outputs = get_tr_data(comment_API_data, word3vec_dic)
    # enc_inputs = np.mean(enc_inputs, axis=2, dtype=np.float32)      # 3个向量求平均得到单词最终的词向量

    # enc_inputs_vec, enc_inputs, dec_inputs, dec_outputs = torch.tensor(enc_inputs_vec), torch.tensor(enc_inputs), \
    # enc_inputs_vec, dec_inputs, dec_outputs = torch.tensor(enc_inputs_vec), \
    #                                                       torch.tensor(dec_inputs), torch.tensor(dec_outputs)

    # loader = DataLoader(MyDataset(enc_inputs_vec, enc_inputs, dec_inputs, dec_outputs),
    loader = DataLoader(MyDataset(enc_inputs_vec, dec_inputs, dec_outputs),
                        batch_size=batch_size, shuffle=True)

    # attention = Attention(embedding_dim=100).to(device)

    model = Transformer().to(device)
    # 多GPU训练
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     # model = torch.nn.DataParallel(model, device_ids=[3, 1, 2])
    #     model = torch.nn.DataParallel(model)
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)         # 忽略 占位符 索引为0.
    # criterion = nn.CosineEmbeddingLoss(margin=0.2)
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)

    for epoch in range(epochs):
        # for enc_inputs_vec, enc_inputs, dec_inputs, dec_outputs in loader:  # enc_inputs : [batch_size, src_len]
        loop = tqdm(loader, desc='Train')
        for enc_inputs_vec, dec_inputs, dec_outputs in loop:  # enc_inputs : [batch_size, src_len]
                                                            # dec_inputs : [batch_size, tgt_len]
                                                            # dec_outputs: [batch_size, tgt_len]

            # enc_inputs_vec, enc_inputs, dec_inputs, dec_outputs = enc_inputs_vec.to(device), \
            #                                                       enc_inputs.to(device), \
            #                                                       dec_inputs.to(device), \
            #                                                       dec_outputs.to(device)
            enc_inputs_vec, dec_inputs, dec_outputs = enc_inputs_vec.to(device), dec_inputs.to(device), \
                                                                  dec_outputs.to(device)
            # enc_inputs_vec = attention(enc_inputs_vec)      # 左边的enc_inputs : [batch_size, src_len, 词向量维度100]

            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs_vec, dec_inputs)
                                                            # outputs: [batch_size * tgt_len, tgt_vocab_size]

            loss = criterion(outputs, dec_outputs.view(-1))
            # loss = cal_loss(outputs)
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_description(f'Epoch [{epoch + 1}/{epochs}]')
            loop.set_postfix(loss=loss)
    # torch.save(model.state_dict(), 'data/model1epoch.pt')
    torch.save(model.state_dict(), 'data/model8ep64bs12head.pt')
    # torch.save(model.state_dict(), 'data/model64.pt')
    end_time = time.time()
    tr_time = end_time - start_time
    print("训练时间为：", tr_time, "秒")
