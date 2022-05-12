import os
import numpy as np
import torch
import torch.nn.functional as F
import random
cuda = True if torch.cuda.is_available() else False


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def train_test_split(data_index_slice,train_scale):

    num = len(data_index_slice)
    num_train = int(train_scale*num)
    train_index = np.array(random.sample(data_index_slice,num_train))
    test_index =  np.setdiff1d(data_index_slice,train_index)

    return  train_index,test_index

def cal_sample_weight(labels, num_class, use_sample_weight=True):
    if not use_sample_weight:
        return np.ones(len(labels)) / len(labels)
    count = np.zeros(num_class)
    for i in range(num_class):
        count[i] = np.sum(labels==i)
    sample_weight = np.zeros(labels.shape)
    for i in range(num_class):
        sample_weight[np.where(labels==i)[0]] = count[i]/np.sum(count)

    return sample_weight

def one_hot_tensor(y, num_dim):
    num_dim = torch.tensor(num_dim)
    if cuda:
        num_dim = num_dim.cuda()
        y = y.cuda()
    y_onehot = torch.zeros(y.shape[0], num_dim)
    if cuda:
        y_onehot = y_onehot.cuda()
    y_onehot.scatter_(1, y.view(-1,1), 1)
    
    return y_onehot


def cosine_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    temp = 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)
    return temp



def to_sparse(x):
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)
    indices = torch.nonzero(x)
    if len(indices.shape) == 0:
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())

def cal_adj_mat_parameter(edge_per_node, data,eps, metric="cosine"):
    assert metric == "cosine", "Only cosine distance implemented"
    dist = cosine_distance_torch(data, data,eps=eps)
    parameter = torch.sort(dist.reshape(-1,)).values[edge_per_node*data.shape[0]]
    return np.asscalar(parameter.data.cpu().numpy())

def graph_from_dist_tensor(dist, parameter, self_dist=True):
    if self_dist:
        assert dist.shape[0]==dist.shape[1], "Input is not pairwise dist matrix"
    g = (dist <= parameter).float()
    if self_dist:
        diag_idx = np.diag_indices(g.shape[0])
        g[diag_idx[0], diag_idx[1]] = 0
        
    return g

def gen_adj_mat_tensor(data, parameter,eps, metric="cosine"):
    assert metric == "cosine", "Only cosine distance implemented"
    dist = cosine_distance_torch(data, data,eps)
    g = graph_from_dist_tensor(dist, parameter, self_dist=True)
    if metric == "cosine":
        adj = 1-dist
    else:
        raise NotImplementedError
    adj = adj*g 
    adj_T = adj.transpose(0,1)
    I = torch.eye(adj.shape[0])
    if cuda:
        I = I.cuda()
    adj = adj + adj_T*(adj_T > adj).float() - adj*(adj_T > adj).float()
    adj = F.normalize(adj + I, p=1)
    adj = to_sparse(adj)
    
    return adj

def gen_test_adj_mat_tensor(data, trte_idx, parameter, eps,metric="cosine"):
    assert metric == "cosine", "Only cosine distance implemented"
    k = len(trte_idx["tr"]) + len(trte_idx["te"])
    adj = torch.zeros((k,k))
    if cuda:
        adj = adj.cuda()
    num_tr = len(trte_idx["tr"])
    
    dist_tr2te = cosine_distance_torch(data[trte_idx["tr"]], data[trte_idx["te"]],eps)
    g_tr2te = graph_from_dist_tensor(dist_tr2te, parameter, self_dist=False)
    if metric == "cosine":
        adj[:num_tr,num_tr:] = 1-dist_tr2te
    else:
        raise NotImplementedError
    adj[:num_tr,num_tr:] = adj[:num_tr,num_tr:]*g_tr2te
    
    dist_te2tr = cosine_distance_torch(data[trte_idx["te"]], data[trte_idx["tr"]],eps)
    g_te2tr = graph_from_dist_tensor(dist_te2tr, parameter, self_dist=False)
    if metric == "cosine":
        adj[num_tr:,:num_tr] = 1-dist_te2tr
    else:
        raise NotImplementedError
    adj[num_tr:,:num_tr] = adj[num_tr:,:num_tr]*g_te2tr
    
    adj_T = adj.transpose(0,1)
    I = torch.eye(adj.shape[0])
    if cuda:
        I = I.cuda()
    adj = adj + adj_T*(adj_T > adj).float() - adj*(adj_T > adj).float()
    adj = F.normalize(adj + I, p=1)
    adj = to_sparse(adj)
    
    return adj

def gen_test_adj_mat_tensor(data_train,data_test, parameter, eps,metric="cosine"):
    assert metric == "cosine", "Only cosine distance implemented"
    k = len(data_train) + len(data_test)
    adj = torch.zeros((k,k))
    if cuda:
        adj = adj.cuda()
    num_tr = len(data_train)

    dist_tr2te = cosine_distance_torch(data_train, data_test,eps)
    g_tr2te = graph_from_dist_tensor(dist_tr2te, parameter, self_dist=False)
    if metric == "cosine":
        adj[:num_tr, num_tr:] = 1 - dist_tr2te
    else:
        raise NotImplementedError
    adj[:num_tr, num_tr:] = adj[:num_tr, num_tr:] * g_tr2te

    dist_te2tr = cosine_distance_torch(data_test,data_train,eps)
    g_te2tr = graph_from_dist_tensor(dist_te2tr, parameter, self_dist=False)
    if metric == "cosine":
        adj[num_tr:, :num_tr] = 1 - dist_te2tr
    else:
        raise NotImplementedError
    adj[num_tr:, :num_tr] = adj[num_tr:, :num_tr] * g_te2tr

    adj_T = adj.transpose(0, 1)
    I = torch.eye(adj.shape[0])
    if cuda:
        I = I.cuda()
    adj = adj + adj_T * (adj_T > adj).float() - adj * (adj_T > adj).float()
    adj = F.normalize(adj + I, p=1)
    adj = to_sparse(adj)

    return adj

def save_model_dict(folder, model_dict):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for module in model_dict:
        torch.save(model_dict[module].state_dict(), os.path.join(folder, module+".pth"))

def load_model_dict(folder, model_dict):
    for module in model_dict:
        if os.path.exists(os.path.join(folder, module+".pth")):

            model_dict[module].load_state_dict(torch.load(os.path.join(folder, module+".pth"), map_location="cuda:{:}".format(torch.cuda.current_device())))
        else:
            print("WARNING: Module {:} from model_dict is not loaded!".format(module))
        if cuda:
            model_dict[module].cuda()    
    return model_dict