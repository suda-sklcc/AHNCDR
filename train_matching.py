import numpy as np
import random
import os
import torch
# import pickle
import time
from collections import defaultdict
from dataset_matching import *
# from dataset import *
import pandas as pd
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import argparse
import torch.nn.functional as F
# from model import *
from model_matching import *
from sklearn.metrics import roc_auc_score
from pathlib import Path
# from pypai.model import upload_model
from tqdm import tqdm
from functools import partial
import logging
from utils import *
# from thop import profile
from sklearn.model_selection import train_test_split  # 划分数据集
from torch.utils.tensorboard import SummaryWriter

# 初始化SummaryWriter
# writer = SummaryWriter('runs/nmcdr')
logger = logging.getLogger()


def test(model, args, valLoader):
    model.eval()
    stats = AverageMeter('loss')
    # stats = AverageMeter('loss','ndcg_1_d1','ndcg_5_d1','ndcg_10_d1','ndcg_1_d2','ndcg_5_d2','ndcg_10_d2','hit_1_d1','hit_5_d1','hit_10_d1','hit_1_d2','hit_5_d2','hit_10_d2','MRR_d1','MRR_d2')
    pred_d1_list = None
    pred_d2_list = None
    criterion_cls = nn.BCELoss(reduce=False)
    fix_value = 1e-7  # fix the same value
    for k, sample in enumerate(tqdm(valLoader)):
        u_node = torch.LongTensor(sample['user_node'].long()).cuda()
        i_node = torch.LongTensor(sample['i_node'].long()).cuda()
        neg_samples = torch.LongTensor(sample['neg_samples'].long()).cuda()
        seq_d1 = torch.LongTensor(sample['seq_d1'].long()).cuda()
        seq_d2 = torch.LongTensor(sample['seq_d2'].long()).cuda()
        pad_d1 = torch.LongTensor(sample['pad_d1'].long()).to(device)
        pad_d2 = torch.LongTensor(sample['pad_d2'].long()).to(device)
        domain_id = torch.LongTensor(sample['domain_id'].long()).to(device)
        overlap_user = torch.LongTensor(sample['isoverlap'].long()).to(device)
        labels = torch.LongTensor(sample['label'].long()).cuda()
        labels = labels.float()
        with torch.no_grad():
            predict_d1, predict_d2 = model(u_node, i_node, neg_samples, seq_d1, seq_d2,pad_d1,pad_d2,overlap_user,domain_id,False)
        predict_d1 = predict_d1.squeeze()
        predict_d2 = predict_d2.squeeze()
        one_value = torch.LongTensor(torch.ones(domain_id.shape[0]).long()).cuda()
        mask_d1 = torch.LongTensor((one_value.cpu() - domain_id.cpu()).long()).cuda()
        mask_d2 = torch.LongTensor((domain_id.cpu()).long()).cuda()
        loss_cls = criterion_cls(predict_d1, labels) * mask_d1.unsqueeze(1) + criterion_cls(predict_d2,labels) * mask_d2.unsqueeze(1)
        loss_cls=torch.mean(loss_cls)

        loss = loss_cls
        stats.update(loss=loss.item())  # ,loss_cl=loss_cl.item())
        domain_id = domain_id.unsqueeze(1).expand_as(predict_d1)
        predict_d1 = predict_d1.view(-1, args.neg_nums + 1).cpu().detach().numpy().copy()
        predict_d2 = predict_d2.view(-1, args.neg_nums + 1).cpu().detach().numpy().copy()
        domain_id = domain_id.view(-1, args.neg_nums + 1).cpu().detach().numpy().copy()
        predict_d1_cse, predict_d2_cse = choose_predict(predict_d1, predict_d2, domain_id)
        if pred_d1_list is None and not isinstance(predict_d1_cse, list):
            pred_d1_list = predict_d1_cse
        elif pred_d1_list is not None and not isinstance(predict_d1_cse, list):
            pred_d1_list = np.append(pred_d1_list, predict_d1_cse, axis=0)
        if pred_d2_list is None and not isinstance(predict_d2_cse, list):
            pred_d2_list = predict_d2_cse
        elif pred_d2_list is not None and not isinstance(predict_d2_cse, list):
            pred_d2_list = np.append(pred_d2_list, predict_d2_cse, axis=0)
    pred_d1_list[:, 0] = pred_d1_list[:, 0] - fix_value
    pred_d2_list[:, 0] = pred_d2_list[:, 0] - fix_value
    HIT_1_d1, NDCG_1_d1, HIT_5_d1, NDCG_5_d1, HIT_10_d1, NDCG_10_d1, MRR_d1 = get_sample_scores(pred_d1_list)
    HIT_1_d2, NDCG_1_d2, HIT_5_d2, NDCG_5_d2, HIT_10_d2, NDCG_10_d2, MRR_d2 = get_sample_scores(pred_d2_list)
    return stats.loss,HIT_1_d1, NDCG_1_d1, HIT_5_d1, NDCG_5_d1, HIT_10_d1, NDCG_10_d1, MRR_d1, HIT_1_d2, NDCG_1_d2, HIT_5_d2, NDCG_5_d2, HIT_10_d2, NDCG_10_d2, MRR_d2


def train(model, trainLoader, args, valLoader, early_stop):
    best_hit_1_d1 = 0
    best_hit_5_d1 = 0
    best_hit_10_d1 = 0
    best_hit_1_d2 = 0
    best_hit_5_d2 = 0
    best_hit_10_d2 = 0

    best_ndcg_1_d1 = 0
    best_ndcg_5_d1 = 0
    best_ndcg_10_d1 = 0
    best_ndcg_1_d2 = 0
    best_ndcg_5_d2 = 0
    best_ndcg_10_d2 = 0

    best_mrr_d1 = 0
    best_mrr_d2 = 0
    save_path1 = Path(args.model_dir) / 'checkpoint' / ( args.domain +str(int(args.overlap_ratio * 100)) + 'best_d1.pt')
    save_path2 = Path(args.model_dir) / 'checkpoint' / ( args.domain +str(int(args.overlap_ratio*100))+'best_d2.pt')
    criterion_recon = partial(sce_loss, alpha=args.alpha_l)
    criterion_cls = nn.BCELoss(reduce=False)
    if not os.path.exists(os.path.join(Path(args.model_dir), 'checkpoint')):
        os.mkdir(os.path.join(Path(args.model_dir), 'checkpoint'))
    for epoch in range(args.epoch):
        stats = AverageMeter('loss', 'loss_cls', 'loss_cls_m1', 'loss_cls_m2','nce_loss')
        model.train()
        for i, sample in enumerate(tqdm(trainLoader)):
            u_node = torch.LongTensor(sample['user_node'].long()).to(device)
            i_node = torch.LongTensor(sample['i_node'].long()).to(device)
            neg_samples = torch.LongTensor(sample['neg_samples'].long()).to(device)
            seq_d1 = torch.LongTensor(sample['seq_d1'].long()).to(device)
            seq_d2 = torch.LongTensor(sample['seq_d2'].long()).to(device)
            pad_d1 = torch.LongTensor(sample['pad_d1'].long()).to(device)
            pad_d2 = torch.LongTensor(sample['pad_d2'].long()).to(device)
            labels = torch.LongTensor(sample['label'].long()).to(device)
            labels = labels.float()
            domain_id = torch.LongTensor(sample['domain_id'].long()).to(device)
            overlap_user = torch.LongTensor(sample['isoverlap'].long()).to(device)

            predict_d1, predict_d2, predict_m0_d1, predict_m0_d2, predict_m1_d1, predict_m1_d2, predict_m2_d1, predict_m2_d2 \
                ,proto_nce_loss_d1,proto_nce_loss_d2= model( u_node, i_node, neg_samples, seq_d1, seq_d2,pad_d1,pad_d2,overlap_user,domain_id)
            predict_d1 = predict_d1.squeeze()
            predict_d2 = predict_d2.squeeze()
            one_value = torch.LongTensor(torch.ones(domain_id.shape[0]).long()).to(device)
            mask_d1 = torch.LongTensor((one_value.cpu() - domain_id.cpu()).long()).to(device)
            mask_d2 = torch.LongTensor((domain_id.cpu()).long()).to(device)

            loss_d1=criterion_cls(predict_d1,labels) * mask_d1.unsqueeze(1)
            loss_d2=criterion_cls(predict_d2,labels) * mask_d2.unsqueeze(1)
            l1 = torch.mean(loss_d1)
            l2 = torch.mean(loss_d2)
            loss_ratio_d1 = 1 - (l1 / (l1 + l2))
            loss_ratio_d2 = 1 - (l2 / (l1 + l2))
            loss_cls = loss_d1 * (1 + loss_ratio_d1) + loss_d2 * (1 + loss_ratio_d2)
            # loss_cls = criterion_cls(predict_d1, labels) * mask_d1.unsqueeze(1) + criterion_cls(predict_d2,labels) * mask_d2.unsqueeze(1)  # * 2
            loss_cls = torch.mean(loss_cls)

            loss_d1=criterion_cls(predict_m2_d1,labels) * mask_d1.unsqueeze(1)
            loss_d2=criterion_cls(predict_m2_d2,labels) * mask_d2.unsqueeze(1)
            l1 = torch.mean(loss_d1)
            l2 = torch.mean(loss_d2)
            loss_ratio_d1 = 1 - (l1 / (l1 + l2))
            loss_ratio_d2 = 1 - (l2 / (l1 + l2))
            loss_cls_m2 = loss_d1 * (1 + loss_ratio_d1) + loss_d2 * (1 + loss_ratio_d2)
            # loss_cls_m2 = criterion_cls(predict_m2_d1, labels) * mask_d1.unsqueeze(1) + criterion_cls(predict_m2_d2,labels) * mask_d2.unsqueeze(1)  # * 2
            loss_cls_m2 = torch.mean(loss_cls_m2)

            loss_d1=criterion_cls(predict_m1_d1,labels) * mask_d1.unsqueeze(1)
            loss_d2=criterion_cls(predict_m1_d2,labels) * mask_d2.unsqueeze(1)
            l1 = torch.mean(loss_d1)
            l2 = torch.mean(loss_d2)
            loss_ratio_d1 = 1 - (l1 / (l1 + l2))
            loss_ratio_d2 = 1 - (l2 / (l1 + l2))
            loss_cls_m1 = loss_d1 * (1 + loss_ratio_d1) + loss_d2 * (1 + loss_ratio_d2)
            # loss_cls_m1 = criterion_cls(predict_m1_d1, labels) * mask_d1.unsqueeze(1) + criterion_cls(predict_m1_d2, labels) * mask_d2.unsqueeze(1)  # * 2
            loss_cls_m1 = torch.mean(loss_cls_m1)

            eps = 1e-8
            l1 = proto_nce_loss_d1
            l2 = proto_nce_loss_d2

            loss_ratio_d1 = 1 - (l1 / (l1 + l2+ eps))
            loss_ratio_d2 = 1 - (l2 / (l1 + l2+ eps))

            nce_loss = l1 * (1 + loss_ratio_d1) + l2 * (1 + loss_ratio_d2)
            nce_loss = torch.mean(nce_loss)


            loss = loss_cls + loss_cls_m1 + loss_cls_m2  +nce_loss # + loss_cls_m0 #+ loss_cl * 0.05
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # auc_avg_domain = roc_auc_score(labels.detach().cpu().numpy(),predict.detach().cpu().numpy())
            stats.update(loss=loss.item(), loss_cls=loss_cls.item(),loss_cls_m2=loss_cls_m2.item(), loss_cls_m1=loss_cls_m1.item(),nce_loss=nce_loss.item())
            if i % 20 == 0:
                logger.info(
                    f'train total loss:{stats.loss}, cls loss:{stats.loss_cls}, cls m1 loss:{stats.loss_cls_m1}, '
                    f'cls m2 loss:{stats.loss_cls_m2} ,nce_loss:{stats.nce_loss}\t')
        val_loss, HIT_1_d1, NDCG_1_d1, HIT_5_d1, NDCG_5_d1, HIT_10_d1, NDCG_10_d1, \
        MRR_d1, HIT_1_d2, NDCG_1_d2, HIT_5_d2, NDCG_5_d2, HIT_10_d2, NDCG_10_d2, MRR_d2 = test(model, args, valLoader)

        best_hit_1_d1 = max(HIT_1_d1, best_hit_1_d1)
        best_hit_5_d1 = max(HIT_5_d1, best_hit_5_d1)
        best_hit_10_d1 = max(HIT_10_d1, best_hit_10_d1)
        best_hit_1_d2 = max(HIT_1_d2, best_hit_1_d2)
        best_hit_5_d2 = max(HIT_5_d2, best_hit_5_d2)
        best_hit_10_d2 = max(HIT_10_d2, best_hit_10_d2)

        best_ndcg_1_d1 = max(best_ndcg_1_d1, NDCG_1_d1)
        best_ndcg_5_d1 = max(best_ndcg_5_d1, NDCG_5_d1)
        best_ndcg_10_d1 = max(best_ndcg_10_d1, NDCG_10_d1)
        best_ndcg_1_d2 = max(best_ndcg_1_d2, NDCG_1_d2)
        best_ndcg_5_d2 = max(best_ndcg_5_d2, NDCG_5_d2)
        best_ndcg_10_d2 = max(best_ndcg_10_d2, NDCG_10_d2)
        if MRR_d1 >= best_mrr_d1:
            #     best_auc1 = auc_testd1
            torch.save(model.state_dict(), str(save_path1))
        if MRR_d2 >= best_mrr_d2:
            #     best_auc2 = auc_testd2
            torch.save(model.state_dict(), str(save_path2))
        best_mrr_d1 = max(best_mrr_d1, MRR_d1)
        best_mrr_d2 = max(best_mrr_d2, MRR_d2)

        # writer.add_scalar('Train Loss', stats.loss, epoch)
        # writer.add_scalar('Val loss', val_loss, epoch)
        #
        # writer.add_scalar('HR@1_1', HIT_1_d1 * 100, epoch)
        # writer.add_scalar('HR@5_1', HIT_5_d1 * 100, epoch)
        # writer.add_scalar('HR@10_1', HIT_10_d1 * 100, epoch)
        # writer.add_scalar('NDCG@5_1', NDCG_5_d1 * 100, epoch)
        # writer.add_scalar('NDCG@10_1', NDCG_10_d1 * 100, epoch)
        # writer.add_scalar('MRR_d1', MRR_d1 * 100, epoch)
        #
        # writer.add_scalar('HR@1_2', HIT_1_d2 * 100, epoch)
        # writer.add_scalar('HR@5_2', HIT_5_d2 * 100, epoch)
        # writer.add_scalar('HR@10_2', HIT_10_d2 * 100, epoch)
        # writer.add_scalar('NDCG@5_2', NDCG_5_d2 * 100, epoch)
        # writer.add_scalar('NDCG@10_2', NDCG_10_d2 * 100, epoch)
        # writer.add_scalar('MRR_d2', MRR_d2 * 100, epoch)

        if epoch > 20 and not early_stop.update(epoch, HIT_10_d1, HIT_10_d2, NDCG_10_d1, NDCG_10_d2):
            print(f"Best epoch get, epoch = {epoch}")
            break
        logger.info(f'Epoch: {epoch}/{args.epoch} \t'
                    f'Train Loss: {stats.loss:.4f} \t'
                    f'Val loss: {val_loss:.4f}\n'
                    f'val domain1 cur/max HR@1: {HIT_1_d1:.4f}/{best_hit_1_d1:.4f} \n,'
                    f'HR@5: {HIT_5_d1:.4f}/{best_hit_5_d1:.4f} \n, '
                    f'HR@10: {HIT_10_d1:.4f}/{best_hit_10_d1:.4f} \n'
                    # f'val domain1 cur/max NDCG@1: {NDCG_1_d1:.4f}/{best_ndcg_1_d1:.4f} \n, '
                    f'NDCG@5: {NDCG_5_d1:.4f}/{best_ndcg_5_d1:.4f} \n, '
                    f'NDCG@10: {NDCG_10_d1:.4f}/{best_ndcg_10_d1:.4f}, \n'
                    f'val domain1 cur/max MRR: {MRR_d1:.4f}/{best_mrr_d1:.4f} \n'
                    f'val domain2 cur/max HR@1: {HIT_1_d2:.4f}/{best_hit_1_d2:.4f} \n, '
                    f'HR@5: {HIT_5_d2:.4f}/{best_hit_5_d2:.4f} \n, '
                    f'HR@10: {HIT_10_d2:.4f}/{best_hit_10_d2:.4f} \n'
                    # f'val domain2 cur/max NDCG@1: {NDCG_1_d2:.4f}/{best_ndcg_1_d2:.4f} \n, '
                    f'NDCG@5: {NDCG_5_d2:.4f}/{best_ndcg_5_d2:.4f} \n, '
                    f'NDCG@10: {NDCG_10_d2:.4f}/{best_ndcg_10_d2:.4f}, \n'
                    f'val domain2 cur/max MRR: {MRR_d2:.4f}/{best_mrr_d2:.4f} \n')
        # upload_model(save_name,name="MRAGN_vis")
    return best_hit_1_d1, best_hit_5_d1, best_hit_10_d1, best_ndcg_5_d1, best_ndcg_10_d1, best_mrr_d1, best_hit_1_d2, best_hit_5_d2, best_hit_10_d2, best_ndcg_5_d2, best_ndcg_10_d2, best_mrr_d2


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Multi-edge multi-domain training')
    parser.add_argument('--epoch', type=int, default=200, help='# of epoch')
    parser.add_argument('--bs', type=int, default=256, help='# images in batch')
    parser.add_argument('--use_gpu', type=bool, default=True, help='gpu flag, true for GPU and false for CPU')
    parser.add_argument('--lr', type=float, default=1e-3,help='initial learning rate for adam')  # 1e-3 for cdr23 3e-4 for cdr12
    parser.add_argument('--emb_dim', type=int, default=128, help='embedding size')
    parser.add_argument('--hid_dim', type=int, default=32, help='hidden layer dim')
    parser.add_argument('--seq_len', type=int, default=20,help='the length of the sequence')  # 20 for mybank 150 for amazon
    parser.add_argument('--m1_layers', type=int, default=3, help='m1 layer nums')
    parser.add_argument('--m2_layers', type=int, default=3, help='m2 layer nums')
    parser.add_argument('--m3_layers', type=int, default=4, help='m3 layer nums')
    parser.add_argument('--m4_layers', type=int, default=2, help='m4 layer nums')
    parser.add_argument('--alpha_l', type=int, default=3, help='sce loss')
    parser.add_argument('--neg_nums', type=int, default=199, help='sample negative numbers')
    parser.add_argument('--overlap_ratio', type=float, default=0.1, help='overlap ratio for choose dataset ')
    parser.add_argument('-md', '--model-dir', type=str, default='model/')
    parser.add_argument('--dataset', type=str, default='data2/', help='data1/,data2/,data3/')
    parser.add_argument('--domain', type=str, default='cloth_sports', help='music_movie,cloth_sports,phone_elec')
    parser.add_argument('--log-file', type=str, default='log')

    args = parser.parse_args()

    hit_1_d1 = []
    hit_5_d1 = []
    hit_10_d1 = []
    hit_1_d2 = []
    hit_5_d2 = []
    hit_10_d2 = []

    ndcg_5_d1 = []
    ndcg_10_d1 = []
    ndcg_5_d2 = []
    ndcg_10_d2 = []

    mrr_d1 = []
    mrr_d2 = []

    for i in range(1):
        SEED = i
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
        early_stop = EarlyStopping()
        args.log_file = "log" + str(i) + "_" + args.domain + str(int(args.overlap_ratio * 100)) + ".txt"
          #music_movie
        # user_length = 94034#92782#142189
        # item_length = 80378
        # # cloth_sports
        user_length = 20976#23930#113938
        item_length = 29789#45426

        datasetTrain = DualDomainSeqDataset(seq_len=args.seq_len,isTrain=True,neg_nums=args.neg_nums,pad_id=item_length+1,
                                            csv_path=args.dataset+args.domain+"_train"+str(int(args.overlap_ratio*100))+".csv")#+str(int(args.overlap_ratio*100))+".csv")
        user_node_train = datasetTrain.user_nodes
        train_id_d1 = [user for user, domain in zip(range(len(user_node_train)), datasetTrain.domain_id) if domain == 0]
        train_id_d2 = [user for user, domain in zip(range(len(user_node_train)), datasetTrain.domain_id) if domain == 1]
        trainLoader = data.DataLoader(datasetTrain, batch_size=args.bs, shuffle=True, num_workers=8,collate_fn=collate_fn_enhance)

        datasetVal = DualDomainSeqDataset(seq_len=args.seq_len,isTrain=False,neg_nums=args.neg_nums,pad_id=item_length+1,
                                          csv_path=args.dataset+args.domain+"_test.csv")
        user_node_val = datasetVal.user_nodes
        val_id_d1 = [user for user, domain in zip(range(len(user_node_val)), datasetVal.domain_id) if domain == 0]
        val_id_d2 = [user for user, domain in zip(range(len(user_node_val)), datasetVal.domain_id) if domain == 1]
        valLoader = data.DataLoader(datasetVal, batch_size=args.bs, shuffle=False, num_workers=8,collate_fn=collate_fn_enhance)
        user_node_train = torch.LongTensor(user_node_train)
        user_node_val = torch.LongTensor(user_node_val)
        train_id_d1 = torch.LongTensor(train_id_d1)
        train_id_d2 = torch.LongTensor(train_id_d2)
        val_id_d1 = torch.LongTensor(val_id_d1)
        val_id_d2 = torch.LongTensor(val_id_d2)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        item_length = item_length + 2  # for pad id
        # user_length = user_length * 2


        model = AHNCDR(user_length=user_length, user_emb_dim=args.emb_dim, item_length=item_length, item_emb_dim=args.emb_dim,
                       seq_len=args.seq_len, m1_layers=args.m1_layers, m3_layers=args.m3_layers,m4_layers=args.m4_layers,hid_dim=args.hid_dim,
                       user_node_train=user_node_train,user_node_val=user_node_val, train_id_d1=train_id_d1,
                       train_id_d2=train_id_d2, val_id_d1=val_id_d1, val_id_d2=val_id_d2,device=device)

        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            model.to(device)
        else:
            model = model.cpu()  # 如果没有 CUDA，将模型移动到 CPU

        print(f"Using device: {device}")

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        init_logger(args.model_dir, args.log_file)
        logger.info(vars(args))

        best_hit_1_d1, best_hit_5_d1, best_hit_10_d1, best_ndcg_5_d1, best_ndcg_10_d1, best_mrr_d1, best_hit_1_d2, best_hit_5_d2, \
        best_hit_10_d2, best_ndcg_5_d2, best_ndcg_10_d2, best_mrr_d2 = train(model, trainLoader, args, valLoader,early_stop)
        # test(model,args,valLoader)
        hit_1_d1.append(best_hit_1_d1)
        hit_5_d1.append(best_hit_5_d1)
        hit_10_d1.append(best_hit_10_d1)
        ndcg_5_d1.append(best_ndcg_5_d1)
        ndcg_10_d1.append(best_ndcg_10_d1)
        mrr_d1.append(best_mrr_d1)

        hit_1_d2.append(best_hit_1_d2)
        hit_5_d2.append(best_hit_5_d2)
        hit_10_d2.append(best_hit_10_d2)
        ndcg_5_d2.append(best_ndcg_5_d2)
        ndcg_10_d2.append(best_ndcg_10_d2)
        mrr_d2.append(best_mrr_d2)
        # break

    log_all_txt = "log_all"+args.domain+str(int(args.overlap_ratio*100))+".txt"
    init_logger(args.model_dir, log_all_txt)
    logger.info(f'domain1 HR@1: {np.mean(hit_1_d1):.4f}/{np.std(hit_1_d1):.4f} \n,'
                f'HR@5: {np.mean(hit_5_d1):.4f}/{np.std(hit_5_d1):.4f} \n, '
                f'HR@10: {np.mean(hit_10_d1):.4f}/{np.std(hit_10_d1):.4f} \n'
                f'NDCG@5: {np.mean(ndcg_5_d1):.4f}/{np.std(ndcg_5_d1):.4f} \n, '
                f'NDCG@10: {np.mean(ndcg_10_d1):.4f}/{np.std(ndcg_10_d1):.4f}, \n'
                f'MRR: {np.mean(mrr_d1):.4f}/{np.std(mrr_d1):.4f} \n'
                f'domain2 HR@1: {np.mean(hit_1_d2):.4f}/{np.std(hit_1_d2):.4f} \n,'
                f'HR@5: {np.mean(hit_5_d2):.4f}/{np.std(hit_5_d2):.4f} \n, '
                f'HR@10: {np.mean(hit_10_d2):.4f}/{np.std(hit_10_d2):.4f} \n'
                f'NDCG@5: {np.mean(ndcg_5_d2):.4f}/{np.std(ndcg_5_d2):.4f} \n, '
                f'NDCG@10: {np.mean(ndcg_10_d2):.4f}/{np.std(ndcg_10_d2):.4f}, \n'
                f'MRR: {np.mean(mrr_d2):.4f}/{np.std(mrr_d2):.4f} \n')

    # writer.close()