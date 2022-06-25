import torch
import numpy as np




def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx, pairwise_distance




def GDM_repli(x):
    k = 64  # number of neighbors to decide the range of j in Eq.(5)
    tau = 0.2  # threshold in Eq.(2)
    sigma = 2  # parameters of f (Gaussian function in Eq.(2))
    ###############
    """Graph Construction:"""
    device = x.device
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    idx, p = knn(x, k=k)  # p: -[(x1-x2)^2+...]

    # here we add a tau
    p1 = torch.abs(p)
    p1 = torch.sqrt(p1)
    mask = p1 < tau

    # here we add a sigma
    p = p / (sigma * sigma)
    w = torch.exp(p)  # b,n,n
    w = torch.mul(mask.float(), w)

    b = 1/torch.sum(w, dim=1)
    b = b.reshape(batch_size, num_points, 1).repeat(1, 1, num_points)
    c = torch.eye(num_points, num_points, device=device)
    c = c.expand(batch_size, num_points, num_points)
    D = b * c  # b,n,n

    A = torch.matmul(D, w)  # normalized adjacency matrix A_hat

    # Get Aij in a local area:
    idx2 = idx.view(batch_size * num_points, -1)
    idx_base2 = torch.arange(0, batch_size * num_points, device=device).view(-1, 1) * num_points
    idx2 = idx2 + idx_base2

    idx2 = idx2.reshape(batch_size * num_points, k)[:, 1:k]
    idx2 = idx2.reshape(batch_size * num_points * (k - 1))
    idx2 = idx2.view(-1)

    A = A.view(-1)
    A = A[idx2].reshape(batch_size, num_points, k - 1)  # Aij: b,n,k
    ###############
    """Disentangling Point Clouds into Sharp(xs) and Gentle(xg) Variation Components:"""
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.reshape(batch_size * num_points, k)[:, 1:k]
    idx = idx.reshape(batch_size * num_points * (k - 1))

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # b,n,c
    neighbor = x.view(batch_size * num_points, -1)[idx, :]
    neighbor = neighbor.view(batch_size, num_points, k - 1, num_dims)  # b,n,k,c
    A = A.reshape(batch_size, num_points, k - 1, 1)  # b,n,k,1
    n = A.mul(neighbor)  # b,n,k,c
    n = torch.sum(n, dim=2)  # b,n,c

    pai = torch.norm(x - n, dim=-1).pow(2)  # Eq.(5)
    return pai



def create_cluster(xyz,point_feat,num_hie):
    batch_size,feat_dim,num_point=point_feat.shape
    
    # pow_list=np.arange(1,num_hie+1)
    num_point_list=np.array([64,128])
    feat_list=[]
    
        
    pi=GDM_repli(xyz)
    sort_index=torch.argsort(pi,-1)
    repeat_index=sort_index.unsqueeze(1).repeat(1,feat_dim,1)
    gather_feat=torch.gather(point_feat,-1,repeat_index)

    gather_feat=gather_feat.reshape(batch_size,feat_dim,num_hie,-1)
    gather_feat=torch.max(gather_feat,-1,keepdim=False)[0]

    

    # for num_point in num_point_list:
    #     sf=gather_feat[:,:,:num_point]
    #     sf_max_feat=torch.max(sf,-1,keepdim=True)[0]
        
    #     gf=gather_feat[:,:,-num_point:]
    #     gf_max_feat=torch.max(gf,-1,keepdim=True)[0]
        
    #     feat_list.append(sf_max_feat)
    #     feat_list.append(gf_max_feat)
    
    # cluster_max_feat=torch.cat(feat_list,-1)
    
    return gather_feat
    

    
   