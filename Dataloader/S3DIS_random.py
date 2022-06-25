import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
import torch.utils.data as data

class S3DISDataset(Dataset):
    def __init__(self, split='train', data_root='trainval_fullarea', num_point=4096, test_area=5, block_size=1.0, sample_rate=1.0, transform=None):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        rooms = sorted(os.listdir(data_root))
        rooms = [room for room in rooms if 'Area_' in room]
        if split == 'train':
            rooms_split = [room for room in rooms if not 'Area_{}'.format(test_area) in room]
        else:
            rooms_split = [room for room in rooms if 'Area_{}'.format(test_area) in room]

        self.room_points, self.room_labels = [], []
        self.room_coord_min, self.room_coord_max = [], []
        num_point_all = []
        labelweights = np.zeros(13)

        for room_name in tqdm(rooms_split, total=len(rooms_split)):
            room_path = os.path.join(data_root, room_name)
            room_data = np.load(room_path)  # xyzrgbl, N*7
            points, labels = room_data[:, 0:6], room_data[:, 6]  # xyzrgb, N*6; l, N
            tmp, _ = np.histogram(labels, range(14))
            labelweights += tmp
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_points.append(points), self.room_labels.append(labels)
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
            num_point_all.append(labels.size)
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        print(self.labelweights)
        sample_prob = num_point_all / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point)
        room_idxs = []
        for index in range(len(rooms_split)):
            room_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
        self.room_idxs = np.array(room_idxs)
        print("Totally {} samples in {} set.".format(len(self.room_idxs), split))

    def __getitem__(self, idx):
        room_idx = self.room_idxs[idx]
        points = self.room_points[room_idx]   # N * 6
        labels = self.room_labels[room_idx]   # N
        N_points = points.shape[0]

        while (True):
            center = points[np.random.choice(N_points)][:3]
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
            if point_idxs.size > 1024:
                break

        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)

        # normalize
        selected_points = points[selected_point_idxs, :]  # num_point * 6
        current_points = np.zeros((self.num_point, 9))  # num_point * 9
        current_points[:, 6] = selected_points[:, 0] / self.room_coord_max[room_idx][0]
        current_points[:, 7] = selected_points[:, 1] / self.room_coord_max[room_idx][1]
        current_points[:, 8] = selected_points[:, 2] / self.room_coord_max[room_idx][2]
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        selected_points[:, 3:6] /= 255.0
        current_points[:, 0:6] = selected_points
        current_labels = labels[selected_point_idxs]
        if self.transform is not None:
            current_points, current_labels = self.transform(current_points, current_labels)


        current_points=torch.FloatTensor(current_points)
        current_labels=torch.LongTensor(current_labels)

        return current_points, current_labels

    def __len__(self):
        return len(self.room_idxs)



def get_sets(data_path,batch_size,test_batch,test_area):
    num_point, block_size, sample_rate = 4096, 1.0, 0.01

    train_data=S3DISDataset(split='train', data_root=data_root, num_point=num_point, test_area=test_area, 
    block_size=block_size, sample_rate=sample_rate, transform=None)
    train_loader=data.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True,num_workers=2)

    test_data=S3DISDataset(split='test', data_root=data_root, num_point=num_point, test_area=test_area, 
    block_size=block_size, sample_rate=sample_rate, transform=None)
    test_loader=data.DataLoader(dataset=test_data,batch_size=test_batch,shuffle=True,num_workers=2)

    valid_loader=S3DISDataset(split='valid', data_root=data_root, num_point=num_point, test_area=test_area, 
    block_size=block_size, sample_rate=sample_rate, transform=None)
    valid_loader=data.DataLoader(dataset=valid_loader,batch_size=test_batch,shuffle=True,num_workers=2)
    
    return train_loader,test_loader,valid_loader






if __name__=='__main__':
    data_root = '/data1/jiajing/worksapce/Algorithm/PointNet/Pointnet_Pointnet2_pytorch/data/stanford_indoor3d/'
    num_point, test_area, block_size, sample_rate = 4096, 1, 1.0, 0.01
    point_data = S3DISDataset(split='valid', data_root=data_root, num_point=num_point, test_area=test_area, 
    block_size=block_size, sample_rate=sample_rate, transform=None)
    s
    

