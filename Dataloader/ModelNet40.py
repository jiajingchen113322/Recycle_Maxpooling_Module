import torch
import numpy as np
import os
import torch.utils.data as data


import sys
import glob
import h5py



########## label meaning #########
### 0 is looking forward ###
### 1 is looking left ###
### 2 is looking right ###


# np.random.seed(0)
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



    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.split == 'train':
            pointcloud = self.translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        
        pointcloud=torch.FloatTensor(pointcloud)
        label=torch.LongTensor(label)

        pointcloud=pointcloud.permute(1,0)

        return pointcloud, label

    def __len__(self):
        # return 32
        return self.data.shape[0]





def get_sets(data_path,train_batch_size,test_batch_size):
    train_data=ModuleNet40(data_path,split='train')
    train_loader=data.DataLoader(dataset=train_data,batch_size=train_batch_size,shuffle=True,num_workers=2)

    test_data=ModuleNet40(data_path,split='test')
    test_loader=data.DataLoader(dataset=test_data,batch_size=test_batch_size,shuffle=True,num_workers=2)

    valid_dataset=ModuleNet40(data_path,split='valid')
    valid_loader=data.DataLoader(dataset=valid_dataset,batch_size=test_batch_size,shuffle=True,num_workers=2)
    
    return train_loader,test_loader,valid_loader






if __name__=='__main__':
    data_path='/data1/jiajing/dataset/ModelNet40/data'
    dataset=ModuleNet40(data_path,'train')

    #### inpt shape is (3,1024) #####
    inpt,label=dataset[2]
    a,b,c=get_sets(data_path,10)
    
    