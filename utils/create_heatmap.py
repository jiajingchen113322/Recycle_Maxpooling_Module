import os
import sys
from numpy import random

from numpy.core.defchararray import mod
sys.path.append(os.getcwd())


import numpy as np
import torch
from model.PointNet import PointNet_cls_CAM
from Dataloader.ModelNet40 import ModuleNet40

import open3d as o3d

def vis_point_imp(datset,label_index):
    model=PointNet_cls_CAM(inpt_dim=3,num_cls=40)
    model.eval()
    
    model_param_path='./Exp/PointNet_CAM/pth/epoch_190'
    dic=torch.load(model_param_path)
    model.load_state_dict(dic['model_state'])
    params=list(model.parameters())
    softmaxs_weight=params[-2].detach().cpu().numpy()
    
    
    
    point_feat = 'mlp1'
    features_blobs = []
    def hook_feature(module, input, output):
        outdata=output.squeeze(0).permute(1,0)
        features_blobs.append(outdata.data.cpu().numpy())
    
    model._modules.get(point_feat).register_forward_hook(hook_feature)
    
    # label_index=2
    all_data=dataset.data
    all_label=dataset.label.squeeze()
    target_index=np.where(all_label==label_index)[0]
    for index in target_index:
        inpt=torch.FloatTensor(all_data[index]).permute(1,0).unsqueeze(0)
        out=model(inpt)
        point_importance=get_point_importance(features_blobs[0],softmaxs_weight[label_index])
        vis_point(all_data[index],point_importance)



def vis_point(point,importance):
    num_point=point.shape[0]
    
    pointcloud=o3d.geometry.PointCloud()
    pointcloud.points=o3d.utility.Vector3dVector(point)
    
    point_color=np.array([[0,0,255]])
    point_color=np.repeat(point_color,num_point,0)
    
    for index in range(point_color.shape[0]):
        imp_factor=importance[index]
        point_color[index,0]=255*imp_factor
        point_color[index,-1]=255*(1-imp_factor)
        
    
    
    affect_index=np.where(importance!=0)[0]
    print(affect_index.shape[0])
    # point_color[affect_index]=np.array([255,0,0])
    pointcloud.colors=o3d.utility.Vector3dVector(point_color)
    
    
    ske_point=o3d.geometry.PointCloud()
    ske_point.points=o3d.utility.Vector3dVector(point[affect_index]+np.array([2,0,0]))
    
    
    random_point_index=np.random.permutation(np.arange(num_point))[:affect_index.shape[0]]
    random_point=o3d.geometry.PointCloud()
    random_point.points=o3d.utility.Vector3dVector(point[random_point_index]+np.array([-2,0,0]))
    random_color=np.repeat(np.array([[0,255,0]]),random_point_index.shape[0],0)
    random_point.colors=o3d.utility.Vector3dVector(random_color)
    
    o3d.visualization.draw_geometries([pointcloud,ske_point,random_point])
    
    

    
    

def get_point_importance(feature,weight):
    num_point=feature.shape[0]
    point_imp=np.zeros(num_point)
    
    point_source=np.argmax(feature,0)
    pinv_feat=np.max(feature,0)
    matter_point=np.unique(point_source)
    for mater_index in matter_point:
        source=np.where(point_source==mater_index)[0]
        
        picked_feat=pinv_feat[source]
        picked_weight=weight[source]
        
        point_imp[mater_index]=np.sum(picked_feat*picked_weight)
    # cam = cam - np.min(cam)
    # cam_img = cam / np.max(cam)
    point_imp=np.clip(point_imp,0,None)
    point_imp=(point_imp-np.min(point_imp))/(np.max(point_imp)-np.min(point_imp))
    
    return point_imp




if __name__=='__main__':
    target_cls=6
    datapath='D:/Computer_vision/Dataset/Modulenet40/ModelNet40/data'
    dataset=ModuleNet40(datapath,'test')
    # train_loader,test_loader,valid_loader=get_sets(datapath,batch_size=10)
    vis_point_imp(dataset,label_index=target_cls)
    