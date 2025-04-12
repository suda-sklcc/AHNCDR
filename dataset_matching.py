import os
import random
from typing import DefaultDict
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from collections import defaultdict
import json


# def seq_padding(seq, length_enc, long_length, pad_id):
#     if len(seq) >= long_length:
#         long_mask = 1
#     else:
#         long_mask = 0
#     if len(seq) >= length_enc:
#         enc_in = seq[-length_enc + 1:]
#     else:
#         enc_in = [pad_id] * (length_enc - len(seq) - 1) + seq
#
#     return enc_in, long_mask
def seq_padding(seq, length_enc, pad_id):  # 序列填充
    if len(seq) >= length_enc:
        enc_in = seq[-length_enc + 1:]
    else:
        enc_in = [pad_id] * (length_enc - len(seq) - 1) + seq

    # 生成标记矩阵：1 表示实际数据，0 表示填充
    mask_matrix = [1] * len(seq[-length_enc + 1:]) if len(seq) >= length_enc else \
                  [0] * (length_enc - len(seq) - 1) + [1] * len(seq)

    return enc_in, mask_matrix

class DualDomainSeqDataset(data.Dataset):
    def __init__(self, seq_len, isTrain, neg_nums, pad_id, csv_path=''):
        super(DualDomainSeqDataset, self).__init__()
        self.user_item_data = pd.read_csv(csv_path)
        print(self.user_item_data['user_id'].max())
        self.user_nodes = self.user_item_data['user_id'].tolist()
        # self.user_nodes = self.__encode_uid__(self.user_nodes_old)
        self.seq_d1 = self.user_item_data['seq_d1'].tolist()
        self.seq_d2 = self.user_item_data['seq_d2'].tolist()
        self.domain_id = self.user_item_data['domain_id'].tolist()
        self.isoverlap = self.user_item_data['overlap'].tolist()
        self.item_pool_d1 = self.__build_i_set__(self.seq_d1)
        self.item_pool_d2 = self.__build_i_set__(self.seq_d2)
        print("domain 1 len:{}".format(len(self.item_pool_d1)))
        print("domain 2 len:{}".format(len(self.item_pool_d2)))
        self.seq_len = seq_len
        self.isTrain = isTrain
        self.neg_nums = neg_nums
        self.pad_id = pad_id

    def __build_i_set__(self, seq1):
        item_d1 = list()
        for item_seq in seq1:
            item_seq_list = json.loads(item_seq)
            for i_tmp in item_seq_list:
                item_d1.append(i_tmp)
        item_pool_d1 = set(item_d1)
        return item_pool_d1

    def __encode_uid__(self, user_nodes):
        u_node_dict = defaultdict(list)
        i = 0
        u_node_new = list()
        for u_node_tmp in user_nodes:
            if len(u_node_dict[u_node_tmp]) == 0:
                u_node_dict[u_node_tmp].append(i)
                i += 1
        for u_node_tmp in user_nodes:
            u_node_new.append(u_node_dict[u_node_tmp][0])
        print("u_id len:{}".format(len(u_node_dict)))
        return u_node_new

    def __len__(self):
        print("dataset len:{}\n".format(len(self.user_nodes)))
        return len(self.user_nodes)

    def __getitem__(self, idx):
        user_node = self.user_nodes[idx]
        seq_d1_tmp = json.loads(self.seq_d1[idx])
        seq_d2_tmp = json.loads(self.seq_d2[idx])
        domain_id_old = self.domain_id[idx]
        label = list()
        isoverlap = self.isoverlap[idx]
        if domain_id_old == 0:
            neg_items_set = self.item_pool_d1 - set(seq_d1_tmp)
            item = seq_d1_tmp[-1]
            seq_d1_tmp = seq_d1_tmp[:-1]
            label.append(1)
            while (item in seq_d1_tmp):
                seq_d1_tmp.remove(item)
            if self.isTrain:
                neg_samples = random.sample(neg_items_set, 1)
                label.append(0)
            else:
                neg_samples = random.sample(neg_items_set, self.neg_nums)
                for _ in range(self.neg_nums):
                    label.append(0)
            domain_id = 0
        else:
            neg_items_set = self.item_pool_d2 - set(seq_d2_tmp)
            item = seq_d2_tmp[-1]
            seq_d2_tmp = seq_d2_tmp[:-1]
            label.append(1)
            while (item in seq_d2_tmp):
                seq_d2_tmp.remove(item)
            if self.isTrain:
                neg_samples = random.sample(neg_items_set, 1)
                label.append(0)
            else:
                neg_samples = random.sample(neg_items_set, self.neg_nums)
                for _ in range(self.neg_nums):
                    label.append(0)
            domain_id = 1
        seq_d1_tmp,pad_d1 = seq_padding(seq_d1_tmp,self.seq_len+1,self.pad_id)
        seq_d2_tmp,pad_d2 = seq_padding(seq_d2_tmp,self.seq_len+1,self.pad_id)
        sample = dict()
        sample['user_node'] = np.array([user_node])
        sample['i_node'] = np.array([item])
        sample['seq_d1'] = np.array([seq_d1_tmp])
        sample['seq_d2'] = np.array([seq_d2_tmp])
        sample['pad_d1'] = np.array([pad_d1])
        sample['pad_d2'] = np.array([pad_d2])
        sample['domain_id'] = np.array([domain_id])
        sample['label'] = np.array(label)  # no need copy
        sample['neg_samples'] = np.array(neg_samples)
        sample['isoverlap'] = np.array([isoverlap])

        return sample


def collate_fn_enhance(batch):
    user_node = torch.cat([ torch.Tensor(sample['user_node']) for sample in batch],dim=0)
    i_node = torch.cat([ torch.Tensor(sample['i_node']) for sample in batch],dim=0)
    seq_d1 = torch.cat([ torch.Tensor(sample['seq_d1']) for sample in batch],dim=0)
    seq_d2 = torch.cat([ torch.Tensor(sample['seq_d2']) for sample in batch],dim=0)
    pad_d1 = torch.cat([torch.Tensor(sample['pad_d1']) for sample in batch], dim=0)
    pad_d2 = torch.cat([torch.Tensor(sample['pad_d2']) for sample in batch], dim=0)
    label = torch.stack([ torch.Tensor(sample['label']) for sample in batch],dim=0)
    domain_id = torch.cat([ torch.Tensor(sample['domain_id']) for sample in batch],dim=0)
    isoverlap = torch.cat([ torch.Tensor(sample['isoverlap']) for sample in batch],dim=0)
    neg_samples = torch.stack([ torch.Tensor(sample['neg_samples']) for sample in batch],dim=0)
    data = {'user_node' : user_node,
            'i_node': i_node,
            'seq_d1' : seq_d1,
            'seq_d2': seq_d2,
            'pad_d1':pad_d1,
            'pad_d2':pad_d2,
            'label':label,
            'domain_id' : domain_id,
            'isoverlap':isoverlap,
            'neg_samples':neg_samples
            }
    return data




