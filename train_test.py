from sklearn.metrics import accuracy_score, f1_score
from models import init_model_dict_mymodel,init_optim_mymodel
from utils import one_hot_tensor,gen_adj_mat_tensor,gen_test_adj_mat_tensor, cal_adj_mat_parameter
import glob
import os
import random
from scipy import stats
import numpy as np
import torch
import shutil
from sklearn.cluster import KMeans
from scipy.io import loadmat,savemat
from sklearn.preprocessing import MaxAbsScaler
import xlwt

def write_excel_xls(path, sheet_name, value):
    index = len(value)
    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet(sheet_name)
    for i in range(0, index):
        for j in range(0, len(value[i])):
            sheet.write(i, j, value[i][j])
    workbook.save(path)


def label_smoothing(inputs, epsilon=0.1):
    K = 2
    return ((1 - epsilon) * inputs) + (epsilon / K)

def knnsearch(center, feature):
    c = feature
    for i in range(len(feature)):
        temp = feature[i]
        temp_1 = abs(center - temp)
        c[i] = np.argmin(temp_1)
    return c

def hist(idx):
    c = np.zeros((1, 10))
    idx = idx.tolist()
    for i in range(10):
        c[0, i] = idx.count(i)
    return c[0, :]


def get_feature(haralick, morphology, center_haralick, center_morphology):

    bowsFeas_h = np.zeros((1, np.shape(center_haralick)[0] * np.shape(center_haralick)[1]))
    ent_h = np.zeros((1, np.shape(center_haralick)[0]))
    for j in range(np.shape(center_haralick)[0]):
        idx = knnsearch(center_haralick[j, :], haralick[:, j])
        counts = hist(idx)
        counts = counts / sum(counts)
        bowsFeas_h[0, j * 10:(j + 1) * 10] = counts
        a = counts * np.log2(counts)
        a[np.isnan(a)] = 0
        ent_h[0, j] = -sum(a)
    feature_haralick = np.concatenate((bowsFeas_h[0, :], np.mean(haralick, axis=0), np.std(haralick, axis=0),
                                       stats.skew(haralick, axis=0), stats.kurtosis(haralick, axis=0), ent_h[0, :]),
                                      axis=0)


    bowsFeas_m = np.zeros((1, np.shape(center_morphology)[0] * np.shape(center_morphology)[1]))
    ent_m = np.zeros((1, np.shape(center_morphology)[0]))
    for j in range(np.shape(center_morphology)[0]):
        idx = knnsearch(center_morphology[j, :], morphology[:, j])
        counts = hist(idx)
        counts = counts / sum(counts)
        bowsFeas_m[0, j * 10:(j + 1) * 10] = counts
        a = counts * np.log2(counts)
        a[np.isnan(a)] = 0
        ent_m[0, j] = -sum(a)
    feature_morphology = np.concatenate((bowsFeas_m[0, :], np.mean(morphology, axis=0), np.std(morphology, axis=0),
                                         stats.skew(morphology, axis=0), stats.kurtosis(morphology, axis=0),
                                         ent_m[0, :]),
                                        axis=0)
    a = np.zeros((1, len(feature_haralick)))
    b = np.zeros((1, len(feature_morphology)))

    a[0, :] = feature_haralick
    b[0, :] = feature_morphology
    return a, b

def get_center(path, num_center=10, scale=0.4):

    path_haralick = os.path.join(path, 'haralick')
    path_morphology = os.path.join(path, 'morphology')
    file_names = os.listdir(path_haralick)
    num = int(scale * len(file_names))
    sequence = random.sample(range(0,len(file_names)),num)


    flag = 0
    for i in range(num):
        if flag == 0:
            p_h = os.path.join(path_haralick, file_names[sequence[i]])
            p_m = os.path.join(path_morphology, file_names[sequence[i]])
            haralick = loadmat(p_h)['haralick']
            morphology = loadmat(p_m)['morphology']
            flag = 1
        else:
            p_h = os.path.join(path_haralick, file_names[sequence[i]])
            p_m = os.path.join(path_morphology, file_names[sequence[i]])
            temp_h = loadmat(p_h)['haralick']
            temp_m = loadmat(p_m)['morphology']
            haralick = np.concatenate((haralick, temp_h), axis=0)
            morphology = np.concatenate((morphology, temp_m), axis=0)

    center_haralick = np.zeros((13, num_center))
    center_morphology = np.zeros((5, num_center))

    for i in range(13):
        clf = KMeans(n_clusters=num_center, max_iter=500,random_state=0)
        temp = haralick[:, i].reshape(-1, 1)
        clf.fit(temp)
        temp_1 = clf.cluster_centers_
        center_haralick[i, :] = temp_1[:, 0].T
    center_haralick = np.sort(center_haralick, axis=1)

    for i in range(5):
        clf = KMeans(n_clusters=num_center, max_iter=500,random_state=0)
        temp = morphology[:, i].reshape(-1, 1)
        clf.fit(temp)
        temp_1 = clf.cluster_centers_
        center_morphology[i, :] = temp_1[:, 0].T
    center_morphology = np.sort(center_morphology, axis=1)
    return center_haralick, center_morphology


def prepare_trte_data(src_data_folder, target_data_folder,center_folder,train,test):

    file_names = os.listdir(os.path.join(src_data_folder, 'haralick'))

    split_names = ['train','test']


    for split_name in split_names:
        split_path = os.path.join(target_data_folder, split_name)
        if os.path.exists(split_path):
            shutil.rmtree(split_path)


    for split_name in split_names:
        split_path = os.path.join(target_data_folder, split_name)
        if os.path.isdir(split_path):
            pass
        else:
            os.mkdir(split_path)

        if os.path.exists(os.path.join(split_path, 'haralick')):
            pass
        else:
            os.mkdir(os.path.join(split_path, 'haralick'))

        if os.path.exists(os.path.join(split_path, 'morphology')):
            pass
        else:
            os.mkdir(os.path.join(split_path, 'morphology'))


    file_names = np.array(file_names)
    train_list = file_names[train]
    test_list = file_names[test]
    file_names.tolist()



    for n in train_list:
        s_h = os.path.join(src_data_folder, 'haralick', n)
        s_m = os.path.join(src_data_folder, 'morphology', n)
        t_h = os.path.join(target_data_folder, 'train', 'haralick', n)
        t_m = os.path.join(target_data_folder, 'train', 'morphology', n)
        if os.path.exists(t_h):
            pass
        else:
            shutil.copy(s_h, t_h)

        if os.path.exists(t_m):
            pass
        else:
            shutil.copy(s_m, t_m)



    for n in test_list:
        s_h = os.path.join(src_data_folder, 'haralick', n)
        s_m = os.path.join(src_data_folder, 'morphology', n)
        t_h = os.path.join(target_data_folder, 'test', 'haralick', n)
        t_m = os.path.join(target_data_folder, 'test', 'morphology', n)
        if os.path.exists(t_h):
            pass
        else:
            shutil.copy(s_h, t_h)

        if os.path.exists(t_m):
            pass
        else:
            shutil.copy(s_m, t_m)



    path_train = os.path.join(target_data_folder, 'train')
    path_test = os.path.join(target_data_folder, 'test')
    if os.path.exists(os.path.join(center_folder,'center_haralick.mat')):
        center_haralick = loadmat(os.path.join(center_folder,'center_haralick.mat'))['center_haralick']
        center_morphology = loadmat(os.path.join(center_folder,'center_morphology.mat'))['center_morphology']
    else:
        center_haralick, center_morphology = get_center(src_data_folder,num_center=10, scale=0.4)
        savemat(os.path.join(center_folder,'center_haralick.mat'),{'center_haralick':center_haralick})
        savemat(os.path.join(center_folder, 'center_morphology.mat'), {'center_morphology': center_morphology})

    data_train_list = []
    data_test_list = []
    data_all_list = []
    idx_dict = {}
    labels = []

    p_tr_haralick = os.path.join(path_train, 'haralick')
    p_tr_morphology = os.path.join(path_train, 'morphology')


    p_te_haralick = os.path.join(path_test, 'haralick')
    p_te_morphology = os.path.join(path_test, 'morphology')

    tr_haralick = glob.glob(os.path.join(p_tr_haralick, '*.mat'))
    tr_morphology = glob.glob(os.path.join(p_tr_morphology, '*.mat'))
    te_haralick = glob.glob(os.path.join(p_te_haralick, '*.mat'))
    te_morphology = glob.glob(os.path.join(p_te_morphology, '*.mat'))

    label_tr = []




    flag = 0
    for i in range(len(tr_haralick)):
        temp_h = loadmat(tr_haralick[i])['haralick']
        temp_m = loadmat(tr_morphology[i])['morphology']
        f_h, f_m = get_feature(temp_h, temp_m, center_haralick, center_morphology)
        n = tr_haralick[i].split('\\')[-1][0]

        if n == 'N' or n == 'C':
            label_tr.append(0)
        elif n == 'T' or 'D':
            label_tr.append(1)

        if flag == 0:
            temp_haralick = f_h
            temp_morphology = f_m
            flag = 1
        else:
            temp_haralick = np.concatenate((temp_haralick, f_h), axis=0)
            temp_morphology = np.concatenate((temp_morphology, f_m), axis=0)

    data_train_list.append(temp_haralick)
    data_train_list.append(temp_morphology)


    ss_tr_h = MaxAbsScaler()
    ss_tr_m = MaxAbsScaler()
    ss_tr_h.fit(data_train_list[0])
    ss_tr_m.fit(data_train_list[1])

    data_train_list[0] = ss_tr_h.transform(data_train_list[0])
    data_train_list[1] = ss_tr_m.transform(data_train_list[1])



    label_te = []
    flag = 0
    for i in range(len(te_haralick)):
        temp_h = loadmat(te_haralick[i])['haralick']
        temp_m = loadmat(te_morphology[i])['morphology']
        f_h, f_m = get_feature(temp_h, temp_m, center_haralick, center_morphology)
        n = te_haralick[i].split('\\')[-1][0]
        if n == 'N' or n == 'C':
            label_te.append(0)
        elif n == 'T' or 'D':
            label_te.append(1)

        if flag == 0:
            temp_haralick = f_h
            temp_morphology = f_m
            flag = 1
        else:
            temp_haralick = np.concatenate((temp_haralick, f_h), axis=0)
            temp_morphology = np.concatenate((temp_morphology, f_m), axis=0)
    data_test_list.append(temp_haralick)
    data_test_list.append(temp_morphology)


    data_test_list[0] = ss_tr_h.transform(data_test_list[0])
    data_test_list[1] = ss_tr_m.transform(data_test_list[1])


    data_all_list.append(np.concatenate((data_train_list[0], data_test_list[0]), axis=0))
    data_all_list.append(np.concatenate((data_train_list[1], data_test_list[1]), axis=0))


    idx_dict["tr"] = list(range(len(label_tr)))
    idx_dict["te"] = list(range(len(label_tr), len(label_tr)+len(label_te)))
    labels = np.concatenate((label_tr, label_te))
    return data_train_list, data_all_list, idx_dict, labels


def gen_trvalte_adj_mat(data_tr_list, data_val_list,data_test_list,adj_parameter,eps=1e-8):
    adj_metric = "cosine"
    adj_train_list = []
    adj_val_list = []
    adj_test_list = []
    for i in range(len(data_tr_list)):
        adj_parameter_adaptive = cal_adj_mat_parameter(adj_parameter, data_tr_list[i],eps, adj_metric)
        adj_train_list.append(gen_adj_mat_tensor(data_tr_list[i], adj_parameter_adaptive,eps, adj_metric))
        adj_val_list.append(gen_test_adj_mat_tensor(data_tr_list[i],data_val_list[i],adj_parameter_adaptive,eps, adj_metric))
        adj_test_list.append(gen_test_adj_mat_tensor(data_tr_list[i],data_test_list[i], adj_parameter_adaptive,eps, adj_metric))

    return adj_train_list, adj_val_list,adj_test_list

def train_epoch_mymodel(data_list,adj_list,one_hot_label,model_dict, optim_dict, train_VCDN=False):
    loss_dict = {}
    criterion = torch.nn.MSELoss(reduction='none')

    for m in model_dict:
        model_dict[m].train()
    num_view = len(data_list)

    for i in range(num_view):
        optim_dict["C{:}".format(i + 1)].zero_grad()
        ci_loss = 0
        ci = model_dict["C{:}".format(i + 1)](
            model_dict["E{:}".format(i + 1)](data_list[i], adj_list[i]))
        label_s = label_smoothing(one_hot_label)
        ci_loss = torch.mean(criterion(ci, label_s))
        ci_loss.backward()
        optim_dict["C{:}".format(i + 1)].step()
        loss_dict["C{:}".format(i + 1)] = ci_loss.detach().cpu().numpy().item()



    optim_dict["F"].zero_grad()
    f = model_dict["F"](
                         [model_dict["E1"](data_list[0], adj_list[0]),model_dict["E2"](data_list[1], adj_list[1])]
                        )
    f_loss =  torch.mean(criterion(f,label_s))
    f_loss.backward()
    optim_dict["F"].step()
    loss_dict["F"] = f_loss.detach().cpu().numpy().item()

    return loss_dict

def val_epoch_mymodel(data_tr_list,data_val_list, adj_val_list, model_dict):

    data_all_list = data_tr_list.copy()
    length_val = len(data_val_list[0])
    data_all_list[0] = torch.cat((data_tr_list[0],data_val_list[0]),dim=0)
    data_all_list[1] = torch.cat((data_tr_list[1],data_val_list[1]),dim=0)
    for m in model_dict:
        model_dict[m].eval()
    c = model_dict["F"]([model_dict["E1"](data_all_list[0], adj_val_list[0]),
                        model_dict["E2"](data_all_list[1], adj_val_list[1])]
                        )


    num_trte = data_all_list[0].shape[0]
    index_val = num_trte - length_val
    c = c[index_val:num_trte, :]
    prob = c.data.cpu().numpy()
    return prob



def train_test_mymodel(data_tr_list, data_val_list, data_te_list, adj_tr_list, adj_val_list, adj_te_list,
                       labels_tr_tensor, labels_val_tensor, labels_te_tensor,
                       view_list, num_class, lr_e_pretrain, lr_e,
                       num_epoch, dim_he_list):

    num_view = len(view_list)
    result = {}
    val_result = {}
    val_result['ACC'] = 0
    val_result['AUC'] = 0
    val_result['F1'] = 0

    a = data_tr_list[0].shape[1]
    b = data_tr_list[1].shape[1]
    dim_list = [a, b]

    model_dict_mymodel = init_model_dict_mymodel(2, num_class, dim_list, dim_he_list)

    for m in model_dict_mymodel:
        if torch.cuda.is_available():
            model_dict_mymodel[m].cuda()
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, 2)
    optim_dict = init_optim_mymodel(num_view, model_dict_mymodel, lr_e)



    for epoch in range(num_epoch + 1):
        loss_dict =train_epoch_mymodel(data_tr_list, adj_tr_list,
                    onehot_labels_tr_tensor, model_dict_mymodel, optim_dict)



        val_prob = val_epoch_mymodel(data_tr_list, data_val_list, adj_val_list, model_dict_mymodel)
        label_val = labels_val_tensor.cpu().numpy()
        val_acc = accuracy_score(label_val, val_prob.argmax(1))
        val_f1 = f1_score(label_val, val_prob.argmax(1))
        if val_acc > val_result['ACC']:
            val_result['ACC'] = val_acc
            val_result['F1'] = val_f1
            val_result['AUC'] = 0
            label_test = labels_te_tensor.cpu().numpy()
            te_prob = val_epoch_mymodel(data_tr_list,data_te_list,adj_te_list, model_dict_mymodel)
            result["predict"] = te_prob.argmax(1)[0]
            result["truth"] = label_test[0]
            result["score"] = te_prob[0,1]


    return result