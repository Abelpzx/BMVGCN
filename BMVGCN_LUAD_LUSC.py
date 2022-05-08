import numpy as np
import os
from train_test import prepare_trte_data,gen_trvalte_adj_mat, train_test_mymodel
from utils import one_hot_tensor, cal_sample_weight,set_seed
import torch
import argparse
import warnings
from sklearn.model_selection import KFold
import datetime
import glob
from scipy.io import savemat
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score,recall_score,precision_score
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--num_class',type=int,default=2,
                    help="num of tumor classes")
parser.add_argument('--seed',type=int,default=0,
                    help='random seed')
parser.add_argument('--epoch_pretrain',type=int,default=0,
                    help="num of pretrain epoch")
parser.add_argument('--epoch', type=int, default=85,
                    help='maximum number of epochs')
parser.add_argument('--src_folder',type=str,default=".\\tasks\\LUAD_LUSC",
                    help='source data path for classification')
parser.add_argument('--tar_folder',type=str,default=".\\data_LUAD_LUSC",
                    help='processed data for classification')
parser.add_argument('--center_folder',type=str,default=".\\center_LUAD_LUSC",
                    help='cluster center for feature')
parser.add_argument('--lr_e_pretrain',type=float,default=1e-3,
                    help='pretain learning rate')
parser.add_argument('--lr_e',type=float,default=1e-3,
                    help='GCN lerarning rate')
parser.add_argument('--lr_c',type=float,default=1e-3,
                    help='Classification learning rate')
parser.add_argument('--dim_he_list',default=[200,200,100],help="dim of the hidden layers in GCN")
if __name__ == "__main__":
    fid = open('results_LUAD_LUSC.txt', 'w')
    args = parser.parse_args()
    set_seed(args.seed)
    path_haralick = os.path.join(args.src_folder, 'haralick')
    num = len(os.listdir(path_haralick))



    X = np.arange(num).tolist()
    kf = KFold(num,shuffle=True,random_state=args.seed)




    result = []
    ii = 0
    for train_index,test_index in kf.split(X):
        data_train_list, data_all_list, idx_dict, labels = prepare_trte_data(args.src_folder,
                                                                             args.tar_folder,
                                                                             args.center_folder,
                                                                             train_index,
                                                                             test_index)

        data_train = np.concatenate((data_train_list[0], data_train_list[1]), axis=1)
        test_1 = data_all_list[0][idx_dict['te'], :]
        test_2 = data_all_list[1][idx_dict['te'], :]
        data_test = np.concatenate((test_1, test_2), axis=1)
        label_train = labels[idx_dict['tr']]
        label_test = labels[idx_dict['te']]


        k = 0.2
        val_index = np.random.choice(train_index, int(k * len(train_index)), replace=False)
        train_index = np.setdiff1d(train_index, val_index)

        data_all_list_1 = data_train_list.copy()

        data_train_list[0] = data_all_list[0][train_index]
        data_train_list[1] = data_all_list[1][train_index]
        label_train = labels[train_index]

        # 准备验证集数据、label
        data_val_list = data_train_list.copy()
        data_val_list[0] = data_all_list[0][val_index]
        data_val_list[1] = data_all_list[1][val_index]
        label_val = labels[val_index]

        # 准备测试集数据
        data_test_list = data_train_list.copy()
        data_test_list[0] = data_all_list[0][idx_dict['te']]
        data_test_list[1] = data_all_list[1][idx_dict['te']]
        label_test = labels[idx_dict['te']]

        # 将数据转成tensor
        train_tensor_list = []
        for i in range(len(data_train_list)):
            train_tensor_list.append(torch.FloatTensor(data_train_list[i]))
            if torch.cuda.is_available():
                train_tensor_list[i] = train_tensor_list[i].cuda()

        val_tensor_list = []
        for i in range(len(data_test_list)):
            val_tensor_list.append(torch.FloatTensor(data_val_list[i]))
            if torch.cuda.is_available():
                val_tensor_list[i] = val_tensor_list[i].cuda()

        test_tensor_list = []
        for i in range(len(data_test_list)):
            test_tensor_list.append(torch.FloatTensor(data_test_list[i]))
            if torch.cuda.is_available():
                test_tensor_list[i] = test_tensor_list[i].cuda()

        labels_tr_tensor = torch.LongTensor(label_train)
        labels_val_tensor = torch.LongTensor(label_val)
        labels_te_tensor = torch.LongTensor(label_test)
        onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, args.num_class)
        sample_weight_tr = cal_sample_weight(label_train, 2)
        sample_weight_tr = torch.FloatTensor(sample_weight_tr)
        if torch.cuda.is_available():
            labels_tr_tensor = labels_tr_tensor.cuda()
            labels_val_tensor = labels_val_tensor.cuda()
            labels_te_tensor = labels_te_tensor.cuda()
            onehot_labels_tr_tensor = onehot_labels_tr_tensor.cuda()
            sample_weight_tr = sample_weight_tr.cuda()
        adj_tr_list, adj_val_list, adj_te_list = gen_trvalte_adj_mat(train_tensor_list, val_tensor_list,
                                                                     test_tensor_list,
                                                                     adj_parameter=2)
        view_list = [1, 2]
        num_view = len(view_list)
        dim_list = [x.shape[1] for x in data_train_list]



        result_attention = train_test_mymodel(train_tensor_list,
                                              val_tensor_list,
                                              test_tensor_list,
                                              adj_tr_list,
                                              adj_val_list,
                                              adj_te_list,
                                              labels_tr_tensor,
                                              labels_val_tensor,
                                              labels_te_tensor,
                                              view_list=[1, 2],
                                              num_class=args.num_class,
                                              lr_e_pretrain=args.lr_e_pretrain,
                                              lr_e=args.lr_e,
                                              num_epoch=args.epoch,
                                              dim_he_list=args.dim_he_list)

        result.append([result_attention["truth"],result_attention["predict"],result_attention["score"]])
        if not result_attention["truth"] == result_attention["predict"]:
            nn = glob.glob(os.path.join(args.tar_folder, 'test', 'haralick', '*'))[0]
            nn = nn.split('\\')[-1]
            print(nn)

        print()


        result_temp = result.copy()
        temp = np.array(result_temp)
        if ii > 2:
            try:
                print("{:}/{:}  time:{:}  acc:{:} f1:{:} auc:{:}  recall_score{:} ps {:}".format(
                    ii,
                    len(X),
                    datetime.datetime.now(),
                    round(accuracy_score(temp[:, 0], temp[:, 1]), 3),
                    round(f1_score(temp[:, 0], temp[:, 1],average='weighted'), 3),
                    round(roc_auc_score(temp[:, 0], temp[:, 2],average='weighted'), 3),
                    round(recall_score(temp[:, 0], temp[:, 1],average='weighted'), 3),
                    round(precision_score(temp[:, 0], temp[:, 1],average='weighted'), 3)
                )
                )
            except ValueError:
                print(
                    "acc:{:} f1:{:} auc:{:} recall_score{:} ps {:}".format(round(accuracy_score(temp[:, 0], temp[:, 1]), 3),
                                                                           round(f1_score(temp[:, 0], temp[:, 1],average='weighted'), 3),
                                                                           0,
                                                                           round(recall_score(temp[:, 0], temp[:, 1],average='weighted'), 3),
                                                                           round(precision_score(temp[:, 0], temp[:, 1],average='weighted'), 3)
                                                                           )
                )

        ii = ii+1
    try:
        test_acc = round(accuracy_score(temp[:, 0], temp[:, 1]), 3)
        test_f1 = round(f1_score(temp[:, 0], temp[:, 1],average='weighted'), 3)
        test_auc = round(roc_auc_score(temp[:, 0], temp[:, 2],average='weighted'), 3)
        test_recall = round(recall_score(temp[:, 0], temp[:, 1],average='weighted'), 3)
        test_precision = round(precision_score(temp[:, 0], temp[:, 1],average='weighted'), 3)
    except ValueError:
        test_acc = round(accuracy_score(temp[:, 0], temp[:, 1]), 3)
        test_f1 = round(f1_score(temp[:, 0], temp[:, 1], average='weighted'), 3)
        test_auc = 0
        test_recall = round(recall_score(temp[:, 0], temp[:, 1], average='weighted'), 3)
        test_precision = round(precision_score(temp[:, 0], temp[:, 1], average='weighted'), 3)
    fid.write(
        'total_epoch: %d\t test_acc:%f \t test_f1:%f\t test_auc:%f \t test_recall:%f\t test_precision:%f\t \n' \
        % (args.epoch, test_acc, test_f1, test_auc, test_recall, test_precision))
    fid.flush()

