import torch
import numpy as np
# import open3d as o3d
import torch.utils.data as data
import os






# cls_list = ['clutter', 'ceiling', 'floor', 
# 'wall', 'beam', 'column', 'door','window', 
# 'table', 'chair', 'sofa', 'bookcase', 'board']


np.random.seed(0)
class S3DISDataset(data.Dataset):
    def __init__(self,root,split,test_area):
        self.test_area=test_area
        self.split=split
        total_area_list=['Area_1','Area_2','Area_3','Area_4','Area_5','Area_6']
        # self.area_list=[]
        assert test_area in np.arange(1,7)
        if split=='train':
            self.area_list=[i for i in total_area_list if int(i.split('_')[1])!=test_area]
            # self.area_list=['Area_6','Area_1','Area_2','Area_3','Area_4']
        else:
              self.area_list=[i for i in total_area_list if int(i.split('_')[1])==test_area]
          
        self.root=root
        self.batch_list=self.create_batch_list()
        
        
    def create_batch_list(self):
        all_batch_list=[]
        for area in self.area_list:
            area_path=os.path.join(self.root,area)
            room_list=os.listdir(area_path)
            for room in room_list:
                if (self.test_area==2) and (self.split!='train') and (room=='auditorium_2'):
                    continue

                batch_folder_path=os.path.join(area_path,room,'Batch_Folder')
                batch_list=os.listdir(batch_folder_path)
                for batch in batch_list:
                    batch_path=os.path.join(batch_folder_path,batch)
                    all_batch_list.append(batch_path)


        if (self.split=='train') and (self.test_area==2):
            auditorium1_path=os.path.join(self.root,'Area_2','auditorium_2')
            batch_folder_path=os.path.join(auditorium1_path,'Batch_Folder')
            batch_list=os.listdir(batch_folder_path)
            for batch in batch_list:
                batch_path=os.path.join(batch_folder_path,batch)
                all_batch_list.append(batch_path)

        
        return all_batch_list
    
    def __getitem__(self,batch_index):
        np_file=self.batch_list[batch_index]
        data=np.load(np_file)
        # inpt=torch.FloatTensor(data[:,0:6])
        # inpt_color=torch.FloatTensor(data[:,6:9])
        inpt=torch.FloatTensor(data[:,:-1])

        # index=[6,7,8,3,4,5,0,1,2]
        # inpt=inpt[:,index]
        label=torch.LongTensor(data[:,-1])
    

        return inpt,label

    def __len__(self):
        # return 20
        return len(self.batch_list)
        

    

def get_sets(data_path,batch_size,test_batch,test_area):
    train_data=S3DISDataset(data_path,split='train',test_area=test_area)
    train_loader=data.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True,num_workers=2)

    test_data=S3DISDataset(data_path,split='test',test_area=test_area)
    test_loader=data.DataLoader(dataset=test_data,batch_size=test_batch,shuffle=True,num_workers=2)

    valid_loader=S3DISDataset(data_path,split='valid',test_area=test_area)
    valid_loader=data.DataLoader(dataset=valid_loader,batch_size=test_batch,shuffle=True,num_workers=2)
    
    return train_loader,test_loader,valid_loader





# def visulize_point(point_path,cls=0):
#     data=np.load(point_path)
#     if cls!=None:
#         pos=(data[:,-2]==cls)
#         data[pos,3:6]=np.array([255,0,0])
    
    
#     points_info=o3d.geometry.PointCloud()
#     points_info.points=o3d.utility.Vector3dVector(data[:,0:3])
#     points_info.colors=o3d.utility.Vector3dVector(data[:,3:6]/255)
#     o3d.visualization.draw_geometries([points_info])



if __name__=='__main__':
    data_path='/data1/jiajing/dataset/S3DIS_area/data'
    dataset=S3DISDataset(data_path,split='test',test_area=2)
    inpt,label=dataset[0]


    get_sets(data_path,5,5,2)
    # for i in range(len(dataset)):
    #     inpt,label=dataset[i]
    #     s
        
    
    
    # point_path='D:/Computer_vision/3D_Dataset\Stanford_Large_Scale/plane_seg_sample/Area_1/conferenceRoom_1/whole_room_point.npy'
    # visulize_point(point_path)
    
