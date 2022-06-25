import os
import sys
from numpy import random
import numpy
import torch
from numpy.core.defchararray import array, mod
from torch.autograd.grad_mode import enable_grad
sys.path.append(os.getcwd())

import scipy.spatial as spa

import numpy as np
import torch
from model.PointNet import PointNet_cls_CAM
from Dataloader.ModelNet40 import ModuleNet40

import open3d as o3d
from sklearn.cluster import KMeans
np.random.seed(1)




def get_point_cluster(point):
    num_point=point.shape[0]
    
    color=np.tile(np.array([[0,0,255]]),(num_point,1))
    
    point_cloud=o3d.geometry.PointCloud()
    point_cloud.points=o3d.utility.Vector3dVector(point)
    point_cloud.colors=o3d.utility.Vector3dVector(color)
    # o3d.visualization.draw_geometries([point_cloud])
    
    voxel=o3d.geometry.voxel_down_sample(point_cloud,0.05)
    voxel_coor=np.array(voxel.points)
    voxel_color=np.array(voxel.colors)
    
    
    neighbor_size=20
    num_voxel=voxel_coor.shape[0]
    dist_matrix=spa.distance_matrix(voxel_coor,voxel_coor)
    index=np.argsort(dist_matrix,1)
    index=index[:,:neighbor_size]
    
    index=index.reshape(-1)
    normalized_coor=voxel_coor[index].reshape(num_voxel,neighbor_size,-1)
    normalized_coor=normalized_coor-np.mean(normalized_coor,1,keepdims=True)
    neighbor_coor=torch.FloatTensor(normalized_coor)
    coor_mat=torch.bmm(neighbor_coor.permute(0,2,1),neighbor_coor)
    
    e,v = torch.symeig(coor_mat, eigenvectors=True)
    labels = get_cluster_result(e,v)
    # print(np.unique(labels,return_counts=True))
    
    sample_index=[] 
    
    for l in np.unique(labels):
        ind=np.where(labels==l)[0]
        picked_ind=np.random.permutation(ind)[:50]
        voxel_color[ind]=np.random.randint(low=0,high=255,size=(1,3))
        sample_index.append(picked_ind)
    
    sample_index=np.concatenate(sample_index)
    sample_coor=voxel_coor[sample_index]
    sample_color=voxel_color[sample_index]
    
    sample_point=o3d.geometry.PointCloud()
    sample_point.points=o3d.utility.Vector3dVector(sample_coor)
    sample_point.colors=o3d.utility.Vector3dVector(sample_color/255)
    
      
    # voxel.colors=o3d.utility.Vector3dVector(voxel_color.astype(np.int)/255)
    o3d.visualization.draw_geometries([sample_point])
        
    
    # voxel_color[index[10]]=np.array([255,0,0])
    # voxel.colors=o3d.utility.Vector3dVector(voxel_color)
    
    # o3d.visualization.draw_geometries([voxel])
    

def get_cluster_result(value,vector):
    eig_value=np.sort(value,1)[:,::-1]
    
    linearity=(eig_value[:,0]-eig_value[:,1])/eig_value[:,0]
    planarity=(eig_value[:,1]-eig_value[:,2])/eig_value[:,1]
    scaterring=eig_value[:,2]/eig_value[:,1]
    
    neighbor_feat=np.stack((linearity,planarity,scaterring),1)
    
    kmeans = KMeans(n_clusters=3, random_state=0).fit(neighbor_feat)
    labels=kmeans.labels_
    
    
    
    return labels
    
    # nei_point=o3d.geometry.PointCloud()
    # nei_point.points=o3d.utility.Vector3dVector(neighbor_feat)
    # o3d.visualization.draw_geometries([nei_point])
    
    
    # voxel=point_cloud.voxel_down_sample(voxel_size=0.05)
    # o3d.visualization.draw_geometries([voxel])
    
    

if __name__=='__main__':
    target_cls=1
    datapath='D:/Computer_vision/Dataset/Modulenet40/ModelNet40/data'
    dataset=ModuleNet40(datapath,'test')
    
    data,label=dataset.data,dataset.label
    index=np.where(label==target_cls)[0]
    target_data=data[index]
    for i in range(len(data)):
        get_point_cluster(target_data[i])
    # train_loader,test_loader,valid_loader=get_sets(datapath,batch_size=10)
    
    