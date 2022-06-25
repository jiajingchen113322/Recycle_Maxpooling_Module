import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx




def cal_loss_raw(pred, gold):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    eps = 0.2
    n_class = pred.size(1)

    one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
    #one_hot = F.one_hot(gold, pred.shape[1]).float()

    one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
    log_prb = F.log_softmax(pred, dim=1)

    loss_raw = -(one_hot * log_prb).sum(dim=1)


    loss = loss_raw.mean()

    return loss,loss_raw











def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature      # (batch_size, 2*num_dims, num_points, k)




class DGCNN_semseg_ref(nn.Module):
    def __init__(self,num_cls,inpt_length):
        super(DGCNN_semseg_ref, self).__init__()
        # self.args = args
        self.k = 20
        self.num_cls=num_cls
        self.inpt_length=inpt_length
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        # self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn6 = nn.BatchNorm1d(1024)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)

        self.conv1 = nn.Sequential(nn.Conv2d(2*self.inpt_length, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        # self.conv6 = nn.Sequential(nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False),
        #                            self.bn6,
        #                            nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, 1024, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(1216, 512, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=0.5)
        self.conv9 = nn.Conv1d(256, self.num_cls, kernel_size=1, bias=False)
        
       
        self.refine_time=2
        self.NUM=1.2

        self.head=nn.Sequential(self.conv7,
                                self.conv8,
                                self.dp1,
                                self.conv9)


    def get_legal_id(self,used_index_list,data_index,num_point):
        mask=torch.zeros(num_point)
        used_index=used_index_list[data_index]
        mask[used_index]=1
        legal_index=torch.where(mask==0)[0]
        return legal_index

    def feature_refinement(self,point_feat):
            device=point_feat.device
            num_point=point_feat.shape[2]
            batch_size=point_feat.shape[0]

            feat_list=[]
            used_index_list=[torch.LongTensor([]).to(device) for _ in range(batch_size)]

            for i in range(self.refine_time):
                hie_feat_list=[]
                for data_index,single_data in enumerate(point_feat):
                    legal_index=self.get_legal_id(used_index_list,data_index,num_point)
                    legal_feat=single_data[:,legal_index]

                    max_feat,max_index=torch.max(legal_feat,-1)
                    max_index=torch.unique(max_index).detach()
                    hie_feat_list.append(max_feat)
                    used_index_list[data_index]=torch.cat((used_index_list[data_index],max_index))
                
                hie_feat_list=torch.stack(hie_feat_list,0)
                feat_list.append(hie_feat_list)
            
            feat_list=torch.stack(feat_list,0)
            # feat_list=feat_list.permute(1,0,2)
            
            return feat_list


    def get_aug_loss(self,inv_feat,point_feat,y):
        
        _,_,num_point=point_feat.shape
        device=inv_feat.device


        pred_1=self.head(torch.cat([inv_feat[:,:,0].unsqueeze(-1).repeat(1,1,num_point),point_feat],1)).permute(0,2,1)
        pred_2=self.head(torch.cat([inv_feat[:,:,1].unsqueeze(-1).repeat(1,1,num_point),point_feat],1)).permute(0,2,1)
        
        pred1_loss,pred1_row_loss=cal_loss_raw(pred_1.reshape(-1,pred_1.shape[-1]),y.reshape(-1))
        pred2_loss,pred2_row_loss=cal_loss_raw(pred_2.reshape(-1,pred_2.shape[-1]),y.reshape(-1))
        
        pc_con = F.softmax(pred_1, dim=-1)#.max(dim=1)[0]
        one_hot = F.one_hot(y, pred_1.shape[-1]).float()
        pc_con = (pc_con*one_hot).max(dim=-1)[0]

        parameters = torch.max(torch.tensor(self.NUM).to(device), torch.exp(pc_con) * self.NUM).reshape(-1).to(device)
        aug_diff = torch.abs(1.0 - torch.exp(pred2_row_loss - pred1_row_loss * parameters)).mean()

        loss=pred1_loss+pred2_loss+aug_diff
        return pred_1,loss









    def forward(self, x,y):
        x=x.permute(0,2,1)

        batch_size = x.size(0)
        num_points = x.size(2)

        x = get_graph_feature(x, k=self.k, dim9=True)   # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 9*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        point_feat = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

        x = self.conv6(point_feat)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        

        inv_feat=self.feature_refinement(x).permute(1,2,0)
        pred1,loss = self.get_aug_loss(inv_feat,point_feat,y)



        return pred1,loss


if __name__=='__main__':
    inpt=torch.randn((5,4096,9))
    label=torch.randint(low=0,high=13,size=(5,4096))
    
    net=DGCNN_semseg_ref(num_cls=13,inpt_length=9)
    out=net(inpt,label)
    s