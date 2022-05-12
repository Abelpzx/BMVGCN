import torch.nn as nn
import torch
import torch.nn.functional as F
class Classifier_1(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim),
                                 nn.Softmax(dim=1))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)

        return x
def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)
           

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_normal_(self.weight.data)
        if self.bias is not None:
            self.bias.data.fill_(0.0)
    
    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    

class GCN_E(nn.Module):
    def __init__(self, in_dim, hgcn_dim, dropout=0.5):
        super().__init__()
        self.gc1 = GraphConvolution(in_dim, hgcn_dim[0])
        self.gc2 = GraphConvolution(hgcn_dim[0], hgcn_dim[1])
        self.gc3 = GraphConvolution(hgcn_dim[1], hgcn_dim[2])
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x, adj)
        x = F.leaky_relu(x, 0.25)
        return x

def get_sizes_list(dim, chunks):
    split_size = (dim + chunks - 1) // chunks
    sizes_list = [split_size] * chunks
    sizes_list[-1] = sizes_list[-1] - (sum(sizes_list) - dim)
    assert sum(sizes_list) == dim
    if sizes_list[-1]<0:
        n_miss = sizes_list[-2] - sizes_list[-1]
        sizes_list[-1] = sizes_list[-2]
        for j in range(n_miss):
            sizes_list[-j-1] -= 1
        assert sum(sizes_list) == dim
        assert min(sizes_list) > 0
    return sizes_list

def get_chunks(x,sizes):
    out = []
    begin = 0
    for s in sizes:
        y = x.narrow(1,begin,s)
        out.append(y)
        begin += s
    return out

class Block(nn.Module):

    def __init__(self,
            input_dims,
            output_dim,
            mm_dim=1600,
            chunks=20,
            rank=15,
            shared=False,
            dropout_input=0,
            dropout_pre_lin=0,
            dropout_output=0,
            pos_norm='before_cat'):
        super(Block, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.mm_dim = mm_dim
        self.chunks = chunks
        self.rank = rank
        self.shared = shared
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        assert(pos_norm in ['before_cat', 'after_cat'])
        self.pos_norm = pos_norm
        self.linear0 = nn.Linear(input_dims[0], mm_dim)
        if shared:
            self.linear1 = self.linear0
        else:
            self.linear1 = nn.Linear(input_dims[1], mm_dim)
        merge_linears0, merge_linears1 = [], []
        self.sizes_list = get_sizes_list(mm_dim, chunks)
        for size in self.sizes_list:
            ml0 = nn.Linear(size, size*rank)
            merge_linears0.append(ml0)
            if self.shared:
                ml1 = ml0
            else:
                ml1 = nn.Linear(size, size*rank)
            merge_linears1.append(ml1)
        self.merge_linears0 = nn.ModuleList(merge_linears0)
        self.merge_linears1 = nn.ModuleList(merge_linears1)
        self.linear_out = nn.Linear(mm_dim, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.sm = nn.Sigmoid()

    def forward(self, x):
        x0 = self.linear0(x[0])
        x1 = self.linear1(x[1])
        bsize = x1.size(0)
        if self.dropout_input > 0:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)
        x0_chunks = get_chunks(x0, self.sizes_list)
        x1_chunks = get_chunks(x1, self.sizes_list)
        zs = []
        for chunk_id, m0, m1 in zip(range(len(self.sizes_list)),
                                    self.merge_linears0,
                                    self.merge_linears1):
            x0_c = x0_chunks[chunk_id]
            x1_c = x1_chunks[chunk_id]
            m = m0(x0_c) * m1(x1_c)
            m = m.view(bsize, self.rank, -1)
            z = torch.sum(m, 1)
            if self.pos_norm == 'before_cat':
                z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
                z = F.normalize(z,p=2)
            zs.append(z)
        z = torch.cat(zs,1)
        if self.pos_norm == 'after_cat':
            z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
            z = F.normalize(z,p=2)

        if self.dropout_pre_lin > 0:
            z = F.dropout(z, p=self.dropout_pre_lin, training=self.training)
        z = self.linear_out(z)
        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        z = self.sm(z)
        return z

def init_model_dict_mymodel(num_view, num_class, dim_list, dim_he_list, gcn_dopout=0.5):
    model_dict = {}
    for i in range(num_view):
        model_dict["E{:}".format(i + 1)] = GCN_E(dim_list[i], dim_he_list, gcn_dopout)
        model_dict["C{:}".format(i+1)] = Classifier_1(dim_he_list[-1],num_class)
    model_dict["F"] = Block([dim_he_list[2],dim_he_list[2]],num_class)
    return model_dict


def init_optim_mymodel(num_view, model_dict, lr_e=1e-4):
    optim_dict = {}
    for i in range(num_view):
        optim_dict["C{:}".format(i + 1)] = torch.optim.Adam(
            list(model_dict["E{:}".format(i + 1)].parameters()) + list(model_dict["C{:}".format(i + 1)].parameters()),
            lr=lr_e)
    optim_dict["F"] = torch.optim.Adam(model_dict["F"].parameters(),lr=lr_e)
    return optim_dict