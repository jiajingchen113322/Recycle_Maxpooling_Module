import numpy as np
import torch
from torch.nn.modules import module

from Dataloader.scanobjectnn import get_sets

from utils.test_perform_cal import get_cls_accuracy
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import shutil
from utils.cal_final_result import accuracy_calculation
from utils.all_utils import smooth_loss


# from model.cls.DGCNN_repmax import DGCNN_ref
# from model.cls.PointNet_pp_repmax import PointNet_pp
# from model.cls.PointNet_repmax import PointNet_hie_ref
# from model.cls.curvenet_repmax import CurveNet_repmax

import random


import time
import argparse



def get_parse():
    parser=argparse.ArgumentParser(description='argumment')
    parser.add_argument('--exp_name',type=str,default='DGCNN_ref_scanobj_exp')
    parser.add_argument('--seed',default=0)
    parser.add_argument('--batch_size',default=16)
    parser.add_argument('--data_path',default='/data1/jiajing/dataset/scanobjectnn/main_split_nobg')
    parser.add_argument('--lr',default=0.001)
    return parser.parse_args()


cfg=get_parse()






def main():
    seed=cfg.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled=False
  
    
    cuda=0
    datapath=cfg.data_path


    from model.cls.DGCNN_repmax import DGCNN_ref
    model=DGCNN_ref(40,0.8,2.1)
         
    train_loader,test_loader,valid_loader=get_sets(datapath,train_batch_size=cfg.batch_size,test_batch_size=cfg.batch_size)
    
    train_model(model,train_loader,valid_loader,cfg.exp_name,cuda)
    

    



def train_model(model,train_loader,valid_loader,exp_name,cuda_n):
    assert torch.cuda.is_available()
    epoch_acc=[]

    #这里应该用GPU
    device=torch.device('cuda:{}'.format(cuda_n))
    model=model.to(device)


    initial_epoch=0
    training_epoch=350

    loss_func=smooth_loss
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
    lr_schedule=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=np.arange(10,training_epoch,40),gamma=0.7)



    #here we define train_one_epoch
    def train_one_epoch():
        iterations=tqdm(train_loader,ncols=100,unit='batch',leave=False)
        
        #真正训练这里应该解封
        epsum=run_one_epoch(model,iterations,"train",loss_func=loss_func,optimizer=optimizer,loss_interval=10)
        
        summary={"loss/train":np.mean(epsum['losses'])}
        return summary


    def eval_one_epoch():
        iteration=tqdm(valid_loader,ncols=100,unit='batch',leave=False)
        #epsum only have logit and labes
        #epsum['logti'] is (batch,4096,13)
        #epsum['labels] is (batch,4096)
        
        epsum=run_one_epoch(model,iteration,"valid",loss_func=loss_func)
        mean_acc=np.mean(epsum['acc'])
        
        epoch_acc.append(mean_acc)
        
        summary={'meac':mean_acc}
        summary["loss/valid"]=np.mean(epsum['losses'])
        return summary



    #build tensorboard
    
    # tensorboard=SummaryWriter(log_dir='.Exp/{}/TB'.format(exp_name))
    tqdm_epoch=tqdm(range(initial_epoch,training_epoch),unit='epoch',ncols=100)

    #build folder for pth_file
    exp_path=os.path.join('./Exp',exp_name)
    pth_path=os.path.join(exp_path,'pth_file')
    tensorboard_path=os.path.join(exp_path,'TB')
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
        os.mkdir(pth_path)
        os.mkdir(tensorboard_path)
    # pth_save_path=os.path.join('Exp',exp_name,'pth_file')
    # if not os.path.exists(pth_save_path):
    #     os.mkdir(pth_save_path)
    
    tensorboard=SummaryWriter(log_dir=tensorboard_path)


    for e in tqdm_epoch:
        train_summary=train_one_epoch()
        valid_summary=eval_one_epoch()
        summary={**train_summary,**valid_summary}
        lr_schedule.step()
        #save checkpoint
        if np.max(epoch_acc)==epoch_acc[-1]:
            summary_saved={**summary,
                            'model_state':model.state_dict(),
                            'optimizer_state':optimizer.state_dict()}


            # torch.save(summary_saved,'./pth_file/{0}/epoch_{1}'.format(exp_name,e))
            torch.save(summary_saved,os.path.join(pth_path,'epoch_{}'.format(e)))
        
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

    for i,(x_cpu,y_cpu) in enumerate(tqdm_iter):
        x,y=x_cpu.to(device),y_cpu.to(device)

        if mode=='train':
            optimizer.zero_grad()
            
        #logtis' shape is [batch,40]
        #y size is [batch,1]
        if mode=='train':
            logits,loss=model(x,y.view(-1))
        else:
            logits,loss=model(x,y.view(-1))


        if loss_func is not None:
            summary['losses']+=[loss.item()]
        
        if mode=='train':
            loss.backward(retain_graph=True)
            optimizer.step()

            #display
            if loss_func is not None and i%loss_interval==0:
                tqdm_iter.set_description("Loss: {:.3f}".format(np.mean(summary['losses'])))

        else:
            log=logits.cpu().detach().numpy()
            lab=y_cpu.numpy()
            
            mean_acc=get_cls_accuracy(log,lab)
            summary['acc'].append(mean_acc)

            if i%loss_interval==0:
                tqdm_iter.set_description("mea_ac: %.3f"%(np.mean(summary['acc'])))


    if mode!='train':
        for param,value in zip(model.parameters(),param_grads):
                param.requires_grad=value


    return summary


if __name__=='__main__':
    main()
