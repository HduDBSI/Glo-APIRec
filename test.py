from dataloader import *
import torch
from torch.utils.data import DataLoader
# from utils import make_data
from utils import *
from model import Transformer
import pandas as pd
import time
from get3vec import get3vec


CUDA_NUM = "cuda:2"

device = torch.device(CUDA_NUM if torch.cuda.is_available() else "cpu")
tgt_len = 120   # 6943


def test(model, enc_input, start_symbol):
    # Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
    enc_outputs, enc_self_attns = model.Encoder(enc_input)
    dec_input = torch.zeros(1, tgt_len).type_as(enc_input.data)
    # dec_input = torch.zeros(1, tgt_len).type(torch.int).to(device)
    next_symbol = start_symbol
    for i in range(0, tgt_len):
        dec_input[0][i] = next_symbol
        dec_outputs, _, _ = model.Decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[i]
        next_symbol = next_word.item()
    return dec_input

'''
# 计算两个词的余弦相似度/相关程度
def get_similarity(word1, word2):
    # w2v = word2vec.Word2Vec.load('../data/w2v_model_comment_API')
    w2v = word2vec.Word2Vec.load('../data/w2v200/w2v_model_comment_API200')
    # print("计算两个词的相似度/相关程度")
    # word1 = 'java.io.File#mkdirs'
    # word2 = 'file'
    # result1 = model.wv.similarity(word1, word2)
    # print(word1 + "和" + word2 + "的相似度为：", result1)
    # print("\n================================")
    return w2v.wv.similarity(word1, word2)
'''

if __name__ == "__main__":
    print('hi')
    start_time = time.time()
    # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    # enc_inputs, dec_inputs, dec_outputs = make_data()
    with open("data/comment_dic_csv.json", "r", encoding='utf-8') as f:
        comment_dict = json.load(f)
    # 可以读csv输入查询
    df = pd.read_csv('data/query32.csv')
    queries = df['query']
    '''
    queries = [
        # 'append strings',
        # 'append text file',
        # 'binaryformatter',
        # 'connect to database',
        # 'convert int to string',
        # 'convert string to int',
        # 'copy file',
        # 'create file',
        'current time',
        'get current time',
        'download file from url',

        'execute sql statement',
        'generate md5 hash code',
        'get current directory',
        'get files in folder',
        'launch process',
        'load bitmap image',
        'load dll',
        'match regular expressions',
        'open file dialog',
        'parse datetime from string',

        'parse xml',
        'play sound',
        'random number',
        'generate random number',
        'read binary file',
        'read text file',
        'send mail',
        'serialize xml',
        'string split',
        'substring',
        'test file exists'
    ]
    '''
    data_len = len(queries)

    # with open("data/comment_dic.json", "r", encoding='utf-8') as f:
    # with open("data/comment_dic_csv.json", "r", encoding='utf-8') as f:
    #     comment_dict = json.load(f)

    # with open('data/mini_comment_API64.pkl', 'rb') as file:
    #     comment_API_data = pickle.load(file)
    # word3vec_dic = get3vec(comment_API_data)

    # enc_inputs = np.zeros(shape=(data_len, 20, 3, 100), dtype=np.float32)
    enc_inputs = np.zeros(shape=(data_len, 20), dtype=np.int64)
    i = 0
    # with torch.no_grad():
    for query in queries:
        # enc_input = []
        '''comment长度补齐0'''
        j = 0
        for key in query.split():
            if key not in comment_dict:
                key = '<unk>'
            enc_inputs[i][j] = comment_dict[key]
            j += 1

        # for key in query.split():
        #     if key in word3vec_dic:
        #         enc_inputs[i][j] = word3vec_dic[key]
        #     j += 1

        # enc_input = torch.tensor(enc_input)
        # if len(enc_input) < MAX_COMMENT_LEN:
        #     for i in range(MAX_COMMENT_LEN - len(enc_input)):
        #         enc_input.append(0)
        # enc_inputs.extend([enc_input])
        i += 1

    enc_inputs = torch.tensor(enc_inputs)
    enc_inputs = enc_inputs.to(device)

    # loader = DataLoader(MyDataset(enc_inputs, dec_inputs, dec_outputs), 2, True)
    # loader = DataLoader(MyDataset(enc_inputs, enc_inputs, enc_inputs), 2, True)
    # enc_inputs, _, _ = next(iter(loader))

    model = Transformer().to(device)
    model.to(device)
    # model.load_state_dict(torch.load("data/model10epoch.pt"))  # model.load_state_dict()函数把加载的权重复制到模型的权重中去
    model.load_state_dict(torch.load("data/model8epoch64batch_size.pt"))  # model.load_state_dict()函数把加载的权重复制到模型的权重中去
    # model.load_state_dict(torch.load("data/model8ep64bs12head.pt"))  # model.load_state_dict()函数把加载的权重复制到模型的权重中去

    """用训练集测试
    enc_inputs_vec, dec_inputs, dec_outputs = torch.load("data/tensor_list.pt")

    for i in range(len(enc_inputs_vec)):
        predict_dec_input = test(model, enc_inputs_vec[i].view(1, -1).to(device), start_symbol=1)
        predict, _, _, _ = model(enc_inputs_vec[i].view(1, -1).to(device), predict_dec_input)
        predict = predict.data.max(1, keepdim=True)[1]

        print([src_idx2word(int(i)) for i in enc_inputs_vec[i]], '->',
              [set(idx2word(n.item()) for n in predict.squeeze())])
    """

    for num in range(len(enc_inputs)):
        # predict_dec_input = test(model, enc_inputs[num].view(1, -1).to(device), start_symbol=api_dict["<start>"]) # 1 * 655
        # predict_dec_input = test(model, enc_inputs[num].view(1, -1).to(device), start_symbol=1) # 1 * 655
        # enc_inputs[num] = enc_inputs[num].to(device)
        # enc_input = attention(enc_inputs[num].unsqueeze(0))

        enc_input = enc_inputs[num].unsqueeze(0)

        # predict_dec_input = test(model, enc_input.to(device), start_symbol=1) # 1 * 655
        # # predict, _, _, _ = model(enc_input.view(1, -1).to(device), predict_dec_input)   # 655 * 2657
        # predict, _, _, _ = model(enc_input.to(device), predict_dec_input)   # 655 * 2657
        # predict = predict.data.max(1, keepdim=True)[1]  # 655 * 1

        # recom = [idx2word(n.item()) for n in predict.squeeze()]
        # recom = sorted(set(recom), key=recom.index)

        # w2v = word2vec.Word2Vec.load('data/w2v_model_comment_API')
        # recomm = []
        # for api in recom:
        #     if api in w2v.wv.key_to_index.keys():
        #         recom.remove(api)
        recom = []
        while len(recom) < 10:
            predict_dec_input = test(model, enc_input.to(device), start_symbol=1)  # 1 * 655
            # predict, _, _, _ = model(enc_input.view(1, -1).to(device), predict_dec_input)   # 655 * 2657
            predict, _, _, _ = model(enc_input.to(device), predict_dec_input)  # 655 * 2657
            predict = predict.data.max(1, keepdim=True)[1]  # 655 * 1
            recom_add = [idx2word(n.item()) for n in predict.squeeze()]
            for api in recom_add:
                if api == '<end>':
                    break
                if api not in recom:
                    recom.append(api)
            recom = sorted(set(recom), key=recom.index)

        # 不用相似度排序
        # i = 0
        # for api in recom:
        #     line = pd.DataFrame({'API': [api]})
        #     pre = pre._append(line, ignore_index=True)
        #     i += 1
        #     if i >= 10:
        #         break

        # 用相似度排序
        # w2v = word2vec.Word2Vec.load('data/w2v_model_comment_API')
        w2v = word2vec.Word2Vec.load('data/' + 'w2v50' + '/w2v_model_comment_API')
        recom_dic = dict()
        for api in recom:
            query_li = queries[num].split()
            sim_sum = 0
            for query in query_li:
                if query in w2v.wv.key_to_index.keys():
                    sim = w2v.wv.similarity(query, api)
                    sim_sum += sim
            recom_dic[api] = sim_sum
        recom_dic = sorted(recom_dic.items(), key=lambda x: x[1], reverse=True)
        i = 0
        pre = pd.DataFrame(columns=['API', 'rel'])
        for api, p in recom_dic:
            # line = pd.DataFrame({'API': [api], 'pro': [p]})
            line = pd.DataFrame({'API': [api]})
            pre = pre._append(line, ignore_index=True)
            i += 1
            if i >= 10:
                break

        pre.to_csv('result/' + '8ep64bs8head50vec/' + str(queries[num]) + '.csv', index=None)
        # pre.to_csv('result/transformer+sim/' + str(queries[num]) + '.csv', index=None)

        # print(queries[num], '->',
        #       set([idx2word(n.item()) for n in predict.squeeze()]))

    end_time = time.time()
    tr_time = end_time - start_time
    print("测试时间为：", tr_time, "秒")
