import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from sklearn.cluster import DBSCAN
from utils import *


class embUserLayerEnhance(nn.Module):
    def __init__(self, user_length, emb_dim):
        super(embUserLayerEnhance, self).__init__()
        self.emb_user_share = nn.Embedding(user_length, emb_dim)
        self.transd1 = nn.Linear(emb_dim, emb_dim)
        self.transd2 = nn.Linear(emb_dim, emb_dim)

    def forward(self, user_id):
        user_nomarl = self.emb_user_share(user_id)
        user_spf1 = self.transd1(user_nomarl)
        user_spf2 = self.transd2(user_nomarl)
        return user_spf1, user_spf2  # , user_nomarl


class embItemLayerEnhance(nn.Module):
    def __init__(self, item_length, emb_dim):
        super(embItemLayerEnhance, self).__init__()
        self.emb_item = nn.Embedding(item_length, emb_dim)

    def forward(self, item_id):
        item_f = self.emb_item(item_id)
        return item_f

class GraphSAGELayer(nn.Module):
    def __init__(self, item_length, emb_dim, neighbor_size):
        super(GraphSAGELayer, self).__init__()
        self.emb_item = nn.Embedding(item_length, emb_dim)
        self.neighbor_size = neighbor_size
        self.linear = nn.Linear(emb_dim, emb_dim)  # 用于特征的线性变换
        self.self_linear = nn.Linear(emb_dim, emb_dim)  # 自身特征的线性变换

    def sample_neighbors(self, adj_matrix, node_indices, mask):
        sampled_neighbors = []

        # 获取每个节点的邻居并进行采样
        for i, neighbors in enumerate(adj_matrix):
            # 获取当前节点的有效邻居（根据掩码）
            valid_neighbors = neighbors[mask[i].bool()]  # 仅选择有效的邻居

            if valid_neighbors.size(0) > self.neighbor_size:
                # 如果有效邻居数量足够，从有效邻居中采样
                sampled_indices = torch.multinomial(valid_neighbors.float(), self.neighbor_size, replacement=False)
                sampled_neighbors.append(valid_neighbors[sampled_indices])
            else:
                # 如果有效邻居不足，从所有邻居中采样
                sampled_indices = torch.multinomial(neighbors.float(), self.neighbor_size, replacement=False)
                sampled_neighbors.append(neighbors[sampled_indices])

        # 将结果转换为张量并返回，返回形状为 (batch_size, neighbor_size)
        return torch.stack(sampled_neighbors)

    def forward(self, user_indices, item_indices, seq, mask):
        # 获取物品嵌入, seq 的形状是 (batch_size, seq_length)
        seq_embeddings = self.emb_item(seq)  # (batch_size, seq_length, emb_dim)
        # 采样邻居
        # `seq` 作为邻接矩阵, `user_indices` 是节点索引, `mask` 是有效节点的掩码
        sampled_neighbors_item = self.sample_neighbors(seq, user_indices, mask)
        # 获取用户邻居和物品邻居的嵌入
        neighbor_item_embeddings = self.emb_item(sampled_neighbors_item)  # (batch_size, neighbor_size, emb_dim)
        # 计算邻居特征的平均聚合 (GraphSAGE的mean aggregator)
        agg_item_neighbors = torch.mean(neighbor_item_embeddings, dim=1)  # (batch_size, emb_dim)
        agg_item_neighbors = agg_item_neighbors.unsqueeze(1).expand(-1, seq_embeddings.size(1), -1)

        # 对聚合的邻居特征和自身特征进行线性变换
        item_agg_features = F.relu(self.self_linear(seq_embeddings) + self.linear(agg_item_neighbors))  # (batch_size, seq_length, emb_dim)

        return item_agg_features

class IUGraphLayer(nn.Module):
    def __init__(self, emb_dim):
        super(IUGraphLayer, self).__init__()
        self.i_to_u = nn.Linear(emb_dim, emb_dim)

    def forward(self, user_feat_d1, seq_d):
        seq_mess = torch.mean(self.i_to_u(seq_d), dim=1)
        user_feat_d1 = user_feat_d1 + seq_mess
        return user_feat_d1




class RefineIUGraphLayer(nn.Module):
    def __init__(self, emb_dim):
        super(RefineIUGraphLayer, self).__init__()
        self.trans = nn.Linear(emb_dim, emb_dim)
        self.num_heads = 16
        self.embed_dim = emb_dim
        self.head_dim = emb_dim // self.num_heads

    def forward(self, user_feat, seq_feat):
        device = user_feat.device

        # 初始化多头注意力层，确保它在正确的设备上
        attention_layer = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads,batch_first=True).to(device)
        # 使用多头注意力
        seq_feat, attn_weights = attention_layer(seq_feat, seq_feat, seq_feat)
        # 现在进行其他的计算
        seq_feat = seq_feat.reshape(-1, seq_feat.shape[-1])  # [N*Seq_len, D]
        # 计算用户和序列特征的相似度
        att = torch.matmul(user_feat, seq_feat.T)  # [batch_size, N*seq_len]
        att = F.normalize(att, p=1, dim=1)  # 按照行归一化

        # 注意力加权并通过线性变换
        message = self.trans(att.unsqueeze(-1) * seq_feat.unsqueeze(0))  # (batch_size, seq_len, embed_dim)

        # 最终更新用户特征

        user_feat = user_feat + torch.mean(message, dim=1).squeeze()
        return user_feat


# class SpecificGrapModule(nn.Module):
#     def __init__(self, emb_dim, layer_nums,item_length):
#         super(SpecificGrapModule, self).__init__()
#         self.gat_module_u = nn.ModuleList()
#         self.gat_module_i = nn.ModuleList()  # For item update
#         self.module = nn.ModuleList()
#         self.layer_nums = layer_nums
#         self.relu = nn.ReLU()
#         self.gnn_layer = GraphSAGELayer(item_length, emb_dim, neighbor_size=4)  # GNN层
#         # self.gat_module_i.append(UIGraphLayer(emb_dim))
#         for i in range(layer_nums):
#             self.gat_module_u.append(IUGraphLayer(emb_dim))
#             # self.gat_module_i.append(UIGraphLayer(emb_dim))
#             self.module.append(RefineIUGraphLayer(emb_dim))
#
#
#     def forward(self, user_feat, seq, mask, user_indices, item_indices):
#         seq_feat = self.gnn_layer(user_indices, item_indices, seq, mask)
#         user_feat = self.relu(self.gat_module_u[0](user_feat, seq_feat))
#         for i in range(self.layer_nums-1):
#             user_feat = self.relu(self.gat_module_u[i+1](user_feat, seq_feat))
#             user_feat = self.relu(self.module[i](user_feat, seq_feat))
#
#         return user_feat,seq_feat
class SpecificGrapModule(nn.Module):
    def __init__(self, emb_dim, layer_nums,item_length):
        super(SpecificGrapModule, self).__init__()
        self.gat_module_u = nn.ModuleList()
        self.module = nn.ModuleList()
        self.layer_nums = layer_nums
        self.relu = nn.ReLU()
        self.gnn_layer = GraphSAGELayer(item_length, emb_dim, neighbor_size=4)  # GNN层
        for i in range(layer_nums):
            self.gat_module_u.append(IUGraphLayer(emb_dim))
            self.module.append(RefineIUGraphLayer(emb_dim))


    def forward(self, user_feat, seq, mask, user_indices, item_indices):
        seq_feat = self.gnn_layer(user_indices, item_indices, seq, mask)
        for i in range(self.layer_nums):
            user_feat = self.relu(self.gat_module_u[i](user_feat, seq_feat))
            user_feat = self.relu(self.module[i](user_feat, seq_feat))
        return user_feat,seq_feat

class GateFuseCell(nn.Module):
    def __init__(self, emb_dim):
        super(GateFuseCell, self).__init__()
        self.trans1 = nn.Linear(emb_dim, emb_dim)
        self.trans2 = nn.Linear(emb_dim, emb_dim)
        self.act = nn.ReLU()

    def forward(self, message_feat, self_feat):  # user_s = shared
        gate_att = torch.sigmoid(self.trans1(message_feat) + self.trans2(self_feat))
        message = gate_att * message_feat + (1 - gate_att) * self_feat
        return message


class InterGraphMatchingLayer(nn.Module):
    def __init__(self, emb_dim):
        super(InterGraphMatchingLayer, self).__init__()
        self.trans_d1 = nn.Linear(emb_dim, emb_dim)
        self.trans_d2 = nn.Linear(emb_dim, emb_dim)
        self.self_d1 = nn.Linear(emb_dim, emb_dim)
        self.self_d2 = nn.Linear(emb_dim, emb_dim)
        self.relu = nn.ReLU()  # need to test
        self.gru_fuse1 = GateFuseCell(emb_dim)
        self.gru_fuse2 = GateFuseCell(emb_dim)

    def forward(self, user_feat_d1, user_feat_d2):
        # pass message d2 to d1

        att_1 = torch.matmul(user_feat_d1, user_feat_d2.T)
        att_1 = att_1 - torch.diag_embed(torch.diag(att_1))
        att_1 = F.normalize(att_1, p=1, dim=1)
        message_d2_d1 = torch.mean(self.trans_d1(att_1.unsqueeze(-1) * user_feat_d2.unsqueeze(0)),dim=1).squeeze()
        message_self = self.self_d1(user_feat_d2)
        message_sum = self.gru_fuse1(message_d2_d1, message_self)

        user_two_d1 = torch.matmul(user_feat_d1, user_feat_d1.T)
        user_two_d1 = torch.matmul(user_two_d1, user_feat_d1)
        user_d1 = F.normalize(user_two_d1, p=1, dim=1)
        user_feat_d1 = user_d1 + message_sum

        # pass message d1 to d2
        att_2 = torch.matmul(user_feat_d2, user_feat_d1.T)
        att_2 = att_2 - torch.diag_embed(torch.diag(att_2))
        att_2 = F.normalize(att_2, p=1, dim=1)
        message_d1_d2 = torch.mean(self.trans_d2(att_2.unsqueeze(-1) * user_feat_d1.unsqueeze(0)), dim=1).squeeze()
        message_self = self.self_d2(user_feat_d1)
        message_sum = self.gru_fuse2(message_d1_d2, message_self)

        user_two_d2 = torch.matmul(user_feat_d2, user_feat_d2.T)
        user_two_d2 = torch.matmul(user_two_d2, user_feat_d2)
        user_feat_d2 = F.normalize(user_two_d2, p=1, dim=1)

        user_d2 = user_feat_d2 + message_sum

        return user_d1, user_d2


class InterGraphMatchingModule(nn.Module):
    def __init__(self, emb_dim, layer_nums):
        super(InterGraphMatchingModule, self).__init__()
        self.module = nn.ModuleList()
        self.layer_nums = layer_nums
        self.relu = nn.ReLU()
        for i in range(layer_nums):
            self.module.append(InterGraphMatchingLayer(emb_dim))

    def forward(self, user_feat_d1, user_feat_d2):
        for i in range(self.layer_nums):
            user_feat_d1, user_feat_d2 = self.module[i](user_feat_d1, user_feat_d2)
            user_feat_d1 = self.relu(user_feat_d1)
            user_feat_d2 = self.relu(user_feat_d2)
        return user_feat_d1, user_feat_d2


# class RefineUIGraphLayer(nn.Module):
#     def __init__(self, emb_dim):
#         super(RefineUIGraphLayer, self).__init__()
#         self.trans = nn.Linear(emb_dim, emb_dim)
#
#     def forward(self, user_feat, seq_feat):
#         seq_feat = seq_feat.reshape(-1, seq_feat.shape[-1])  # [N*Seq_len,D]
#         att = torch.matmul(user_feat, seq_feat.T)  # [bs,bs]
#         att = F.normalize(att, p=1, dim=1)
#         message = self.trans(att.unsqueeze(-1) * seq_feat.unsqueeze(0))
#         user_feat = user_feat + torch.mean(message, dim=1).squeeze()
#         return user_feat
#





class RefineUIGraphModule(nn.Module):
    def __init__(self, emb_dim, layer_nums):
        super(RefineUIGraphModule, self).__init__()
        self.module = nn.ModuleList()
        self.layer_nums = layer_nums
        self.relu = nn.ReLU()
        for i in range(layer_nums):
            self.module.append(RefineIUGraphLayer(emb_dim))

    def forward(self, user_feat, seq_feat):
        for i in range(self.layer_nums):
            user_feat = self.relu(self.module[i](user_feat, seq_feat))
        return user_feat


class predictModule(nn.Module):
    def __init__(self, emb_dim, hid_dim):
        super(predictModule, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(emb_dim * 2, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, 1))

    def forward(self, user_spf1, user_spf2, i_feat):
        '''
            user_spf : [bs,dim]
            i_feat : [bs,dim]
            neg_samples_feat: [bs,1/99,dim] 1 for train, 99 for test
        '''
        user_spf1 = user_spf1.unsqueeze(1).expand_as(i_feat)
        user_item_concat_feat_d1 = torch.cat((user_spf1, i_feat), -1)
        logits_d1 = torch.sigmoid(self.fc(user_item_concat_feat_d1))

        user_spf2 = user_spf2.unsqueeze(1).expand_as(i_feat)
        user_item_concat_feat_d2 = torch.cat((user_spf2, i_feat), -1)
        logits_d2 = torch.sigmoid(self.fc(user_item_concat_feat_d2))

        return logits_d1.squeeze(), logits_d2.squeeze()


class AHNCDR(nn.Module):

    def __init__(self, user_length, user_emb_dim, item_length, item_emb_dim, seq_len, m1_layers, m3_layers,m4_layers,hid_dim,
                 user_node_train, user_node_val, train_id_d1, train_id_d2, val_id_d1, val_id_d2, device):
        super(AHNCDR, self).__init__()
        self.user_emb_layer = embUserLayerEnhance(user_length, user_emb_dim)
        self.item_emb_layer = embItemLayerEnhance(item_length, item_emb_dim)

        self.m1_layers = m1_layers

        self.m3_layers = m3_layers
        self.m4_layers = m4_layers
        self.device = device
        self.UIGraph_m1_d1 = SpecificGrapModule(user_emb_dim,m4_layers,item_length)  # GATGraphModule(user_emb_dim,head_nums,layers)#nn.ModuleList()
        self.UIGraph_m1_d2 = SpecificGrapModule(user_emb_dim,m4_layers,item_length)  # GATGraphModule(user_emb_dim,head_nums,layers)#nn.ModuleList()

        self.interGM_m3 = InterGraphMatchingModule(user_emb_dim, m3_layers)
        # self.refine_m4_d1 = RefineUIGraphModule(user_emb_dim, m4_layers)
        # self.refine_m4_d2 = RefineUIGraphModule(user_emb_dim, m4_layers)
        self.seq_len = seq_len
        self.relu = nn.ReLU()
        self.w1 = nn.ParameterList([nn.Parameter(torch.rand(user_emb_dim, user_emb_dim), requires_grad=True)])
        self.w2 = nn.ParameterList([nn.Parameter(torch.rand(user_emb_dim, user_emb_dim), requires_grad=True)])

        self.predictModule = predictModule(user_emb_dim, hid_dim)
        self.predictModule2 = predictModule(user_emb_dim, hid_dim)
        self.predictModule3 = predictModule(user_emb_dim, hid_dim)
        self.predictModule4 = predictModule(user_emb_dim, hid_dim)
        self.predictModule5 = predictModule(user_emb_dim, hid_dim)
        self.user_node_train_d1, self.user_node_train_d2 = self.user_emb_layer(user_node_train)

        self.centroids_d1, self.node2cluster_d1, self.node2cluster_dist_d1 = self.run_dbscan(self.user_node_train_d1,train_id_d1)
        self.centroids_d2, self.node2cluster_d2, self.node2cluster_dist_d2 = self.run_dbscan(self.user_node_train_d2,train_id_d2)


        self.sp_loss = SPLoss(n_samples=256)
        self.proto_reg =  1e-3#8e-8#5e-7


    def run_dbscan(self, x, indexes):
        """get DBSCAN clusters"""
        # 转换为 NumPy 数组
        x = x.detach().cpu().numpy()
        use_emb = np.take(x, indexes, axis=0)

        # 执行 DBSCAN 聚类
        clustering = DBSCAN(eps=7.9, min_samples=10)
        labels = clustering.fit_predict(use_emb)  # `labels` 是形状为 `(num_samples,)` 的聚类标签数组

        # 找到所有唯一的聚类标签（忽略噪声点标签 -1）
        unique_labels = np.unique(labels[labels != -1])

        # 计算聚类中心
        cluster_cents = []
        for label in unique_labels:
            cluster_points = use_emb[labels == label]
            centroid = np.mean(cluster_points, axis=0)
            cluster_cents.append(centroid)
        cluster_cents = np.array(cluster_cents)

        # 检查是否有有效聚类
        if cluster_cents.size == 0:
            raise ValueError("DBSCAN did not find any valid clusters")

        D = np.dot(x, cluster_cents.T) / (np.linalg.norm(x, axis=1, keepdims=True) * np.linalg.norm(cluster_cents.T, axis=0, keepdims=True))
        sim = 1 - D
        # 转换为 CUDA 张量
        centroids = torch.Tensor(cluster_cents).to(self.device)

        # 将标签转换为张量（注意：DBSCAN 标签可能包含 -1，表示噪声点）
        node2cluster = torch.LongTensor(labels).squeeze().to(self.device)

        # 直接返回未正则化的相似度张量
        node2cluster_dist = torch.from_numpy(sim).to(self.device)

        return centroids, node2cluster, node2cluster_dist


    # def spilt_overlap(self, index, domain_ids):
    #     domain_d1, domain_d2 = [], []
    #     # user = [idx for idx, u in enumerate(index)]
    #     for i, idx in enumerate(index):
    #     # 针对每个用户的domain_id进行操作
    #         if domain_ids[i] == 0:
    #             domain_d1.append(i)
    #         else:
    #             domain_d2.append(i)
    #         # domain_d1.append(idx)
    #         # domain_d2.append(idx)
    #     domain_d1 = torch.LongTensor(domain_d1).to(self.device)
    #     domain_d2 = torch.LongTensor(domain_d2).to(self.device)
    #     return domain_d1, domain_d2

    def spilt_overlap(self, index, domain_ids):
        user = [idx for idx, u in enumerate(index) if u == 1]
        domain_d1, domain_d2 = [], []
        for i, idx in enumerate(user):
            # 针对每个用户的domain_id进行操作
            if domain_ids[idx] == 0:domain_d1.append(idx)
            else:domain_d2.append(idx)
            # domain_d1.append(idx)
            # domain_d2.append(idx)
        domain_d1 = torch.LongTensor(domain_d1).to(self.device)
        domain_d2 = torch.LongTensor(domain_d2).to(self.device)
        return domain_d1, domain_d2

    def proto_nce_loss(self, user_embeddings_src_all,index, node2cluster_dist, user_centroids_src):
        """global structural loss for overlapping users with positive and filtered negative samples"""

        user_embeddings_src = user_embeddings_src_all[index]  # B, e
        norm_user_embedding_src = F.normalize(user_embeddings_src)

        # 正样本：重叠用户在目标域中的表示
        user2cluster_dist_tgt = node2cluster_dist[index]  # B, c
        user_embeddings_tgt_new = torch.matmul(user2cluster_dist_tgt, user_centroids_src)  # B, e
        # norm_user_embeddings_tgt_new = F.normalize(user_embeddings_tgt_new)

        # 正样本得分
        pos_score_user = torch.mul(norm_user_embedding_src, user_embeddings_tgt_new).sum(dim=1)
        pos_score_user = torch.exp(pos_score_user / 0.1)  # ssl_temp = 0.1

        ttl_score_user = torch.matmul(norm_user_embedding_src, user_embeddings_tgt_new.transpose(0,1))
        ttl_score_user = torch.exp(ttl_score_user /  0.1).sum(dim=1)

        # 计算 NCE loss
        proto_nce_loss_user = self.proto_reg*self.sp_loss(-torch.log(pos_score_user / ttl_score_user), index)
        # # 更新嵌入
        # user_embeddings_src_all = user_embeddings_src_all.clone()
        # user_embeddings_src_all[index] = user_embeddings_tgt_new

        return proto_nce_loss_user

    def forward(self, u_node, i_node, neg_samples, seq_d1, seq_d2, mask_d1, mask_d2, overlap_user, domain_id,isTrain=True):
        user_spf1, user_spf2 = self.user_emb_layer(u_node)
        i_feat = self.item_emb_layer(i_node).unsqueeze(1)
        neg_samples_feat = self.item_emb_layer(neg_samples)
        u_feat_enhance_m1_d1,seq_d1_feat = self.UIGraph_m1_d1(user_spf1, seq_d1, mask_d1, u_node, i_node)
        u_feat_enhance_m1_d2,seq_d2_feat = self.UIGraph_m1_d2(user_spf2, seq_d2, mask_d2, u_node, i_node)

        u_feat_enhance_m2_d1 = u_feat_enhance_m1_d1
        u_feat_enhance_m2_d2 = u_feat_enhance_m1_d2

        domain_d1_index, domain_d2_index = self.spilt_overlap(overlap_user, domain_id)

        if isTrain :
            if len(domain_d1_index) > 0:
                proto_nce_loss_d1 = self.proto_nce_loss(u_feat_enhance_m2_d1,domain_d1_index,
                                                                              self.node2cluster_dist_d1.to(self.device),
                                                                              self.centroids_d2.to(self.device))
            else: proto_nce_loss_d1 = torch.tensor(0.0).to(self.device)
            if len(domain_d2_index) > 0:
                proto_nce_loss_d2 = self.proto_nce_loss(u_feat_enhance_m2_d2,domain_d2_index,
                                                                              self.node2cluster_dist_d2.to(self.device),
                                                                              self.centroids_d1.to(self.device))
            else:
                proto_nce_loss_d2 = torch.tensor(0.0).to(self.device)
        u_feat_enhance_m3_d1, u_feat_enhance_m3_d2 = self.interGM_m3(u_feat_enhance_m2_d1, u_feat_enhance_m2_d2)

        i_feat = torch.cat((i_feat, neg_samples_feat), 1)
        user_feat_d1 = torch.matmul(u_feat_enhance_m3_d1, self.w1[0]) + torch.matmul(u_feat_enhance_m3_d2,1 - self.w1[0])
        user_feat_d2 = torch.matmul(u_feat_enhance_m3_d2, self.w2[0]) + torch.matmul(u_feat_enhance_m3_d1,1 - self.w2[0])

        u_feat_enhance_m4_d1 = user_feat_d1
        u_feat_enhance_m4_d2 = user_feat_d2

        logits_d1, logits_d2 = self.predictModule(u_feat_enhance_m4_d1, u_feat_enhance_m4_d2, i_feat)

        if isTrain:
            logits_m2_d1, logits_m2_d2 = self.predictModule2(u_feat_enhance_m3_d1, u_feat_enhance_m3_d2, i_feat)
            # logits_m2_d1, logits_m2_d2 = self.predictModule3(u_feat_enhance_m2_d1, u_feat_enhance_m2_d2, i_feat)
            logits_m1_d1, logits_m1_d2 = self.predictModule4(u_feat_enhance_m1_d1, u_feat_enhance_m1_d2, i_feat)
            logits_m0_d1, logits_m0_d2 = self.predictModule5(user_spf1, user_spf2, i_feat)

            return logits_d1, logits_d2, logits_m0_d1, logits_m0_d2, logits_m1_d1, logits_m1_d2, logits_m2_d1, logits_m2_d2,proto_nce_loss_d1,proto_nce_loss_d2
        else:
            return logits_d1, logits_d2#,user_spf1, user_spf2, u_feat_enhance_m4_d1, u_feat_enhance_m4_d2

