import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset, dataset
import torch.utils.data as data
# import open3d as o3d
import torch


class ScanObjectNN(Dataset):
    def __init__(self,data_path,split='train',num_points=1024):
        self.split = split
        self.BASE_DIR=data_path
        self.data, self.label = self.load_scanobjectnn_data()
        self.num_points = num_points
               

    
    
    def translate_pointcloud(self,pointcloud):
        xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
        xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
        
        translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
        return translated_pointcloud
    
    
    
    
    
    def load_scanobjectnn_data(self):
        # self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        # self.BASE_DIR='D:/Computer_vision/Dataset/ScanObjectNN/h5_files/h5_files/main_split'
        # DATA_DIR = os.path.join(self.BASE_DIR, 'data')
        all_data = []
        all_label = []

        if self.split=='train':
            partition='training'
        else:
            partition='test'
            
        
        # h5_name = self.BASE_DIR + '/data/' + partition + '_objectdataset_augmentedrot_scale75.h5'
        h5_name = os.path.join(self.BASE_DIR, partition + '_objectdataset.h5')
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
        all_data = np.concatenate(all_data, axis=0)
        all_label = np.concatenate(all_label, axis=0)
        return all_data, all_label
    
    
    
    
    def __len__(self):
        return self.data.shape[0]
    
    
    
    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points,:].astype(np.float)
        label = self.label[item]
        # if self.split == 'train':
        #     pointcloud = self.translate_pointcloud(pointcloud)
        #     np.random.shuffle(pointcloud)
        
        pointcloud=torch.FloatTensor(pointcloud)
        label=torch.LongTensor([label])

        pointcloud=pointcloud.permute(1,0)

        return pointcloud, label

    


def get_sets(data_path,train_batch_size,test_batch_size):
    train_data=ScanObjectNN(data_path,split='train')
    train_loader=data.DataLoader(dataset=train_data,batch_size=train_batch_size,shuffle=True,num_workers=2)

    test_data=ScanObjectNN(data_path,split='test')
    test_loader=data.DataLoader(dataset=test_data,batch_size=test_batch_size,shuffle=True,num_workers=2)

    valid_dataset=ScanObjectNN(data_path,split='valid')
    valid_loader=data.DataLoader(dataset=valid_dataset,batch_size=test_batch_size,shuffle=True,num_workers=2)
    
    return train_loader,test_loader,valid_loader










if __name__=='__main__':
    data_path='/data1/jiajing/dataset/scanobjectnn/main_split_nobg'
    dataset=ScanObjectNN(data_path,split='train')
    
    picked_index=100
    picked_data=dataset[picked_index]
    
    a,b,c=get_sets(data_path,10,10)
    
    # for (x,y) in a:
    #     s
    # pointcloud=o3d.geometry.PointCloud()
    # # pc=o3d.geometry.PointCloud()
    # # pc.points=o3d.utility.Vector3dVector(point_cloud.transpose(1,0)+2)
    # pointcloud.points=o3d.utility.Vector3dVector(picked_data[0].reshape(-1,3))
    # o3d.visualization.draw_geometries([pointcloud]) 