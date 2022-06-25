from ast import parse
from model.seg.DPFA_seg_repmax import DPFA_repmax
import numpy as np
import torch

from utils.test_perform_cal import get_mean_accuracy
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from utils.cal_final_result import accuracy_calculation

from model.seg.PointNet_seg_repmax import PointNet_seg_ref 
from model.seg.DPFA_seg_repmax import DPFA_repmax
from model.seg.DGCNN_seg_repmax import DGCNN_semseg_ref




from sklearn.metrics import confusion_matrix
import argparse


TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())




def get_parse():
    parser=argparse.ArgumentParser(description='argumment')
    parser.add_argument('--model',default='dgcnn_rmp',choices=['dgcnn','pointnet','dpfa'])
    parser.add_argument('--seed',default=0)
    parser.add_argument('--cuda',default=0,type=int)
    parser.add_argument('--test_area',default=6,type=int)
    parser.add_argument('--exp_name',default='DGCNN_area6_ref',type=str)
    parser.add_argument('--batch_size',default=10,type=int)
    parser.add_argument('--lr',default=0.001)
    parser.add_argument('--neighbor',default=20)
    parser.add_argument('--train',default=True,type=bool)
    parser.add_argument('--data_path',default='/home/jchen152/workspace/Data/S3DIS_room/data/stanford_indoor3d')
    parser.add_argument('--epoch',default=100,type=int)
    parser.add_argument('--multi_gpu',default=0,type=int)
    parser.add_argument('--max_iter',default=6,type=int)

    return parser.parse_args()

cfg=get_parse()


def main():
    seed=cfg.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled=False
    
    cuda=0
    datapath=cfg.data_path
    datapath=cfg.data_path

    if cfg.model=='pointnet':
        model=PointNet_seg_ref(inpt_dim=9,num_cls=13)
        from Dataloader.S3DIS_random import get_sets


    elif cfg.model=='dgcnn':
        model=DGCNN_semseg_ref(num_cls=13,inpt_length=9)
        from Dataloader.S3DIS_random import get_sets
    
    elif cfg.model=='dpfa':
        model=DPFA_repmax(num_cls=13,inpt_length=9)
        from Dataloader.S3DIS import get_sets

    else:
        raise ValueError('Unknown Model')



    train_loader,test_loader,valid_loader=get_sets(datapath,batch_size=cfg.batch_size,test_batch=cfg.batch_size,test_area=cfg.test_area)
    if cfg.train:
        train_model(model,train_loader,valid_loader,cfg.exp_name,cuda)
    
    else:
        test(model,test_loader,pth_folder=cfg.exp_name)
        
def test(model,data_loader,pth_folder):
    pth_file=os.path.join('Exp',pth_folder,'./pth_file')
    # pth_list=os.listdir(pth_file)
    # index_list=np.array([int(i.split('_')[-1]) for i in pth_list])
    # target_pth=pth_list[np.argmax(index_list)]
    target_pth_path=os.path.join(pth_file,'epoch_66')

    dic=torch.load(target_pth_path)
    model.eval()
    model.load_state_dict(dic['model_state'])
    model.cuda()
    device=next(model.parameters()).device

    num_cls=13
    confusion_matrix=np.zeros((num_cls,num_cls))
    tq_it=tqdm(data_loader,ncols=100,unit='batch')

    for indice,(x_cpu,y_cpu) in enumerate(tq_it):
        x=x_cpu.to(device)
        y=y_cpu.to(device)
        out,_=model(x,y)
        out=out.cpu().detach().numpy()
        out=out.reshape(-1,num_cls)
        out=np.argmax(out,axis=1).reshape(-1)
        label=y_cpu.view(-1).numpy()
        for i in range(out.shape[0]):
            truth=label[i]
            prediction=out[i]
            confusion_matrix[truth,prediction]+=1
    
    calcula_class=accuracy_calculation(confusion_matrix)
    # np.save('./result/{}'.format(pth_folder),calcula_class)
    print(calcula_class.get_over_all_accuracy())
    print(calcula_class.get_average_intersection_union())
    print(calcula_class.get_intersection_union_per_class())
    







def train_model(model,train_loader,valid_loader,exp_name,cuda_n):
    assert torch.cuda.is_available()
    device=torch.device('cuda:{}'.format(cuda_n))
    #这里应该用GPU
    
    if cfg.multi_gpu:
        model = nn.DataParallel(model).to(device)
    else:
        model=model.to(device)
    # device=torch.device('cpu')
    # model=model.to(device)
    initial_epoch=0
    training_epoch=cfg.epoch

    loss_func=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg.lr)
    lr_schedule=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=np.arange(10,training_epoch,20),gamma=0.7)
    



    #here we define train_one_epoch
    def train_one_epoch():
        iterations=tqdm(train_loader,ncols=100,unit='batch',leave=False)
        epsum=run_one_epoch(model,iterations,"train",loss_func=loss_func,optimizer=optimizer,loss_interval=10)
        
        summary={"loss/train":np.mean(epsum['losses'])}
        return summary


    def eval_one_epoch():
        iteration=tqdm(valid_loader,ncols=100,unit='batch',leave=False)
        epsum=run_one_epoch(model,iteration,"valid",loss_func=loss_func)
        mean_acc=np.mean(epsum['acc'])
        summary={'meac':mean_acc}
        summary["loss/valid"]=np.mean(epsum['losses'])
        return summary,epsum['conf_mat']



  
    exp_path=os.path.join('Exp',exp_name)
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
    
    tensorboard=SummaryWriter(log_dir=os.path.join(exp_path,'TB'))
    tqdm_epoch=tqdm(range(initial_epoch,training_epoch),unit='epoch',ncols=100)

    pth_save_path=os.path.join(exp_path,'pth_file')
    if not os.path.exists(pth_save_path):
        os.mkdir(pth_save_path)

    acc_list=[]
    for e in tqdm_epoch:
        train_summary=train_one_epoch()
        valid_summary,conf_mat=eval_one_epoch()
        summary={**train_summary,**valid_summary}
        acc_list.append(summary['meac'])
        lr_schedule.step()
       
        if np.max(acc_list)==acc_list[-1]:

            if cfg.multi_gpu:
                summary_saved={**train_summary,
                                'conf_mat':confusion_matrix,
                                'model_state':model.module.state_dict(),
                                'optimizer_state':optimizer.state_dict()}
            else:
                summary_saved={**train_summary,
                                'conf_mat':confusion_matrix,
                                'model_state':model.state_dict(),
                                'optimizer_state':optimizer.state_dict()}

            # torch.save(summary_saved,'./pth_file/{0}/epoch_{1}'.format(exp_name,e))
            torch.save(summary_saved,os.path.join(pth_save_path,'epoch_{}'.format(e)))


           
        for name,val in summary.items():
            tensorboard.add_scalar(name,val,e)
    


def run_one_epoch(model,tqdm_iter,mode,loss_func=None,optimizer=None,loss_interval=10):
    if mode=='train':
        model.train()
    else:
        model.eval()
        param_grads=[]
        for param in model.parameters():
            param_grads+=[param.requires_grad]
            param.requires_grad=False
    
    summary={"losses":[],"acc":[]}
    device=next(model.parameters()).device
    confusion_martrix=np.zeros((13,13))



    for i,(x_cpu,y_cpu) in enumerate(tqdm_iter):  
        x,y=x_cpu.to(device),y_cpu.to(device)

        if mode=='train':
            optimizer.zero_grad()

       
        logits,loss=model(x,y)
        if loss_func is not None:
            summary['losses']+=[loss.item()]
            
        
        if mode=='train':
            loss.backward()
            optimizer.step()

            #display
            if loss_func is not None and i%loss_interval==0:
                tqdm_iter.set_description("Loss: %.3f"%(np.mean(summary['losses'])))

        else:
            log=logits.cpu().detach().numpy()
            lab=y_cpu.numpy()
            # num_cls=model.num_cls
            num_cls=13

            mean_acc=get_mean_accuracy(log,lab,num_cls)
            summary['acc'].append(mean_acc)


            label=lab.reshape(-1)
            prediction=log.reshape(-1,num_cls)
            prediction=np.argmax(prediction,1)
            confusion_martrix+=confusion_matrix(label,prediction,labels=np.arange(13))
            


            if i%loss_interval==0:
                tqdm_iter.set_description("mea_ac: %.3f"%(np.mean(summary['acc'])))


    if mode!='train':
        for param,value in zip(model.parameters(),param_grads):
                param.requires_grad=value
        
        summary['conf_mat']=confusion_martrix

    return summary


if __name__=='__main__':
    main()
