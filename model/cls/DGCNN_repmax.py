import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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




def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
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
  
    return feature




class DGCNN_ref(nn.Module):
    def __init__(self, output_channels=40,lamda=None,alpha=2.1):
        super(DGCNN_ref, self).__init__()
        self.k = 20
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)

        self.head=nn.Sequential(self.linear1,
                                self.bn6,
                                nn.LeakyReLU(0.2),
                                self.dp1,
                                self.linear2,
                                self.bn7,
                                nn.LeakyReLU(0.2),
                                self.dp2,
                                self.linear3)
    

        self.refine_time=2
        self.NUM=alpha
        self.lamda=lamda
    
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


    def get_aug_loss(self,inv_feat,y):
    
        device=inv_feat.device

        pred_1=self.head(inv_feat[:,:,0])
        pred_2=self.head(inv_feat[:,:,1])
        
        pred1_loss,pred1_row_loss=cal_loss_raw(pred_1,y)
        pred2_loss,pred2_row_loss=cal_loss_raw(pred_2,y)
        
        pc_con = F.softmax(pred_1, dim=-1)#.max(dim=1)[0]
        one_hot = F.one_hot(y, pred_1.shape[1]).float()
        pc_con = (pc_con*one_hot).max(dim=1)[0]

        parameters = torch.max(torch.tensor(self.NUM).to(device), torch.exp(pc_con) * self.NUM).to(device)
        aug_diff = torch.abs(1.0 - torch.exp(pred2_row_loss - pred1_row_loss * parameters)).mean()

        if self.lamda==None:
            loss=pred1_loss+pred2_loss+aug_diff
        else:
            loss=(1-self.lamda)*(pred1_loss+pred2_loss)+self.lamda*aug_diff
        
        return pred_1,loss






    def forward(self,x,y):
        # y=torch.LongTensor([0,0]).cuda()

        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
   

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]
    


        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        inv_feat=self.feature_refinement(x).permute(1,2,0)
        pred1,loss = self.get_aug_loss(inv_feat,y)
        return pred1,loss








class DGCNN_hie_ref(nn.Module):
    def __init__(self, output_channels=40):
        super(DGCNN_hie_ref, self).__init__()
        self.k = 20
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)

        self.head=nn.Sequential(self.linear1,
                                self.bn6,
                                nn.LeakyReLU(0.2),
                                self.dp1,
                                self.linear2,
                                self.bn7,
                                nn.LeakyReLU(0.2),
                                self.dp2,
                                self.linear3)
    

        self.refine_time=4
        self.NUM=1.5

        ####### if hie, the previous level works for next level,
        ####### if not hie, all level serves for the first level
        self.hie=True


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


    def get_aug_loss(self,inv_feat,y):
        if self.hie:
            level_num=-2
        else:
            level_num=0


        device=inv_feat.device
        iter_time=inv_feat.shape[-1]

        pred_list=[]
        # pred_loss_list=[]
        pred_row_loss_list=[]

        self.num_list=[self.NUM,self.NUM]

        pred_loss_total=0
        aug_diff=0
        for i in range(iter_time):
            pred=self.head(inv_feat[:,:,i])
            pred_list.append(pred)
            
            pred_loss,pred_row_loss=cal_loss_raw(pred,y)
            # pred_loss_list.append(pred_loss)
            pred_loss_total+=pred_loss
            pred_row_loss_list.append(pred_row_loss)

            if i!=0:
                pc_con=F.softmax(pred_list[level_num],dim=-1)
                one_hot=F.one_hot(y,pred_list[level_num].shape[1]).float()
                pc_con = (pc_con*one_hot).max(dim=1)[0]
                parameters = torch.max(torch.tensor(self.num_list[i-1]).to(device), torch.exp(pc_con) * self.num_list[i-1]).to(device)
                aug_diff += torch.abs(1.0 - torch.exp(pred_row_loss_list[-1] - pred_row_loss_list[level_num] * parameters)).mean()

        total_loss=aug_diff+pred_loss_total
        return pred_list[0],total_loss





    def forward(self, x,y):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
   

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]
    


        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        inv_feat=self.feature_refinement(x).permute(1,2,0)
        pred1,loss = self.get_aug_loss(inv_feat,y)
        return pred1,loss









if __name__=='__main__':
    inpt=torch.rand((5,3,1024))
    network=DGCNN_ref(output_channels=40)
    label=torch.LongTensor([0,0,0,0,0])
    
    out=network(inpt,label)
