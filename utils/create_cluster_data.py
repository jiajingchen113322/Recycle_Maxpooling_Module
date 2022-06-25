import torch
import numpy as np
import os
import torch.utils.data as data
import open3d as o3d

import sys
import glob
import h5py
import scipy.spatial as spa

from sklearn.cluster import KMeans
from tqdm import tqdm


np.random.seed(0)
class ModuleNet40(data.Dataset):
    def __init__(self,root,split):
        if split=='train':
            self.split=split
        else:
            self.split='test'
        self.root=root
        self.data,self.label=self.get_datalabel()
        self.num_points=1024

    def get_datalabel(self):
        all_data = []
        all_label = []
        for h5_name in glob.glob(os.path.join(self.root, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%self.split)):
            f = h5py.File(h5_name)
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            f.close()
            all_data.append(data)
            all_label.append(label)
        all_data = np.concatenate(all_data, axis=0)
        all_label = np.concatenate(all_label, axis=0)
        return all_data, all_label

    def translate_pointcloud(self,pointcloud):
        xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
        xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
        
        translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
        return translated_pointcloud


    def get_point_cluster(self,point):
        #Create Point Cloud
        num_point=point.shape[0]
        point_cloud=o3d.geometry.PointCloud()
        point_cloud.points=o3d.utility.Vector3dVector(point)

        #Downsample PointCloud into Voxel
        voxel=o3d.geometry.voxel_down_sample(point_cloud,0.1)
        voxel_coor=np.array(voxel.points)
        
        # Get Each Voxel's coordinate covairance matrix
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
        
        ## get each feature cluster's coordinate ###
        e,v = torch.symeig(coor_mat, eigenvectors=True)
        labels = self.get_cluster_result(e,v)
        # coor_list=[]
        coor_list=np.zeros((0,50,3))
        for l in np.unique(labels):
            l_index=np.where(labels==l)[0]
            picked_ind=np.random.permutation(l_index)[:50]
            while len(picked_ind)<50:
                make_up=50-len(picked_ind)
                make_up_index=np.random.permutation(l_index)[:make_up]
                picked_ind=np.append(picked_ind,make_up_index)
            
            vo_co=np.expand_dims(voxel_coor[picked_ind],0)
            
            coor_list=np.append(coor_list,vo_co,0)
            
            # coor_list.append(voxel_coor[picked_ind])
        # coor_list=np.array(coor_list)
        return coor_list
    
    

    
    
    def get_cluster_result(self,value,vector):
        eig_value=np.sort(value,1)[:,::-1]
        
        linearity=(eig_value[:,0]-eig_value[:,1])/eig_value[:,0]
        planarity=(eig_value[:,1]-eig_value[:,2])/eig_value[:,1]
        scaterring=eig_value[:,2]/eig_value[:,1]
        
        neighbor_feat=np.stack((linearity,planarity,scaterring),1)
        
        kmeans = KMeans(n_clusters=3, random_state=0).fit(neighbor_feat)
        labels=kmeans.labels_
        
        
        
        return labels
    
    
    
    
    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.split == 'train':
            pointcloud = self.translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)

        
        ### get feature cluster ###
        cluster_coord=self.get_point_cluster(pointcloud)
        
        ### visulization ###
        # sampled_voxel=cluster_coord.reshape(150,-1)
        # pointcloud_o3d=o3d.geometry.PointCloud()
        # pointcloud_o3d.points=o3d.utility.Vector3dVector(cluster_coord[2].reshape(-1,3))
        
        # origi_point=o3d.geometry.PointCloud()
        # origi_point.points=o3d.utility.Vector3dVector(pointcloud+2)
        
        # o3d.visualization.draw_geometries([pointcloud_o3d,origi_point])
        #### visulization ###
        
        pointcloud=torch.FloatTensor(pointcloud)
        label=torch.LongTensor(label)

        pointcloud=pointcloud.permute(1,0)

        return pointcloud, torch.FloatTensor(cluster_coord), label

    def __len__(self):
        return self.data.shape[0]





def get_sets(data_path,batch_size):
    train_data=ModuleNet40(data_path,split='train')
    train_loader=data.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True,num_workers=0)

    test_data=ModuleNet40(data_path,split='test')
    test_loader=data.DataLoader(dataset=test_data,batch_size=batch_size,shuffle=True,num_workers=0)

    valid_dataset=ModuleNet40(data_path,split='valid')
    valid_loader=data.DataLoader(dataset=valid_dataset,batch_size=batch_size,shuffle=True,num_workers=0)
    
    return train_loader,test_loader,valid_loader






if __name__=='__main__':
    data_path='D:\Computer_vision\Dataset\Modulenet40\ModelNet40\data'
    dataset=ModuleNet40(data_path,'test')
    
    save_path='D:/Computer_vision/Dataset/Modulenet40/ModelNet40_cluster/Test'
    # point_cloud,point_cluster,label=dataset[0]
   
    for i in tqdm(range(len(dataset))):
        point_cloud,point_cluster,label=dataset[i]
        
        point_cloud=point_cloud.numpy()
        point_cluster=point_cluster.numpy()
        label=label.numpy()

        file_name='data_{}'.format(i)
        file_save_path=os.path.join(save_path,file_name)
        np.savez(file_save_path,pc=point_cloud,p_clster=point_cluster,label=label)
        
    # target_cls=4
    # label=dataset.label
    # index=np.where(label==target_cls)[0]
    # # for i in index:
    # #     inpt,label=dataset[i]
    # a,b,c=get_sets(data_path,10)
    # for (point,cluster_point,label) in tqdm(a):
    #     aaa=1
        
        
        
    
    