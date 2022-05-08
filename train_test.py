"""
Training and testing of the model using 5-fold
"""
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score,recall_score,precision_score
from utils import train_test_split
from sklearn.model_selection import KFold
from models import init_model_dict_mymodel,init_optim_mymodel
from utils import one_hot_tensor, cal_sample_weight, gen_adj_mat_tensor,gen_test_adj_mat_tensor, cal_adj_mat_parameter
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
    index = len(value)  # 获取需要写入数据的行数
    workbook = xlwt.Workbook()  # 新建一个工作簿
    sheet = workbook.add_sheet(sheet_name)  # 在工作簿中新建一个表格
    for i in range(0, index):
        for j in range(0, len(value[i])):
            sheet.write(i, j, value[i][j])  # 像表格中写入数据（对应的行和列）
    workbook.save(path)  # 保存工作簿
    print("结果更新！")

def label_smoothing(inputs, epsilon=0.1):
    K = 2 # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / K)

def knnsearch(center, feature):  # 返回每个元素距离最近的聚类中心的索引
    c = feature  # 存放最近的中心
    for i in range(len(feature)):
        temp = feature[i]
        temp_1 = abs(center - temp)
        c[i] = np.argmin(temp_1)
    return c

def hist(idx):  # 得到聚类的频数
    c = np.zeros((1, 10))  # 存放每一类的频数
    idx = idx.tolist()
    for i in range(10):
        c[0, i] = idx.count(i)
    return c[0, :]


def get_feature(haralick, morphology, center_haralick, center_morphology):
    '''
    :param haralick:  haralick特征矩阵
    :param morphology: morphology特征矩阵
    :param center_haralick:  haralick聚类中心
    :param center_morphology:  morphology聚类中心
    :return feature_haralick : haralick模态特征
    :return feature_morphology: morphology模态特征
    '''

    # 先处理haralick特征
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

    # 再处理morphology特征
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
    '''
    :param
        path:  包含所有样本的文件夹的路径 例如：'./tasks/LUAD_LUSC'
        num_center: 选取的聚类中心的个数
        scale:随机选取训练集的多少数据提取聚类中心
    '''
    #print("*********************************开始计算聚类中心*************************************")
    path_haralick = os.path.join(path, 'haralick')  # haralick文件夹地址
    path_morphology = os.path.join(path, 'morphology')  # morphology文件夹地址

    file_names = os.listdir(path_haralick)  # 所有的文件名
    num = int(scale * len(file_names))  # 用于提取中心的sample数量
    sequence = random.sample(range(0,len(file_names)),num)  # 被用来计算中心的样本序号

    # 读数据
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

    center_haralick = np.zeros((13, num_center))  # 存放haralick的聚类中心
    center_morphology = np.zeros((5, num_center))  # 存放morphology的聚类中心

    for i in range(13):
        clf = KMeans(n_clusters=num_center, max_iter=500,random_state=0)
        temp = haralick[:, i].reshape(-1, 1)
        clf.fit(temp)
        temp_1 = clf.cluster_centers_
        center_haralick[i, :] = temp_1[:, 0].T
    center_haralick = np.sort(center_haralick, axis=1)  # 对聚类中心矩阵每一行进行排序

    for i in range(5):
        clf = KMeans(n_clusters=num_center, max_iter=500,random_state=0)
        temp = morphology[:, i].reshape(-1, 1)
        clf.fit(temp)
        temp_1 = clf.cluster_centers_
        center_morphology[i, :] = temp_1[:, 0].T
    center_morphology = np.sort(center_morphology, axis=1)  # 对聚类中心矩阵每一行进行排序
    return center_haralick, center_morphology


def prepare_trte_data(src_data_folder, target_data_folder,center_folder,train,test):
    '''
    :param src_data_folder:   LUAD_LUSC文件夹绝对地址
    :param:target_data_folder:      目标文件夹地址
    :param view_list:         模态个数
    :param train:训练索引
    :param test:测试索引


    :return: data_train_list：张量，是个list,list里每个元素是一个tensor,存放这个模态的训练数据
    :return: data_all_list :张量，是个list，list里每一个元素是一个tensor,存放这个模态的训练+测试数据
    :return: idx_dict: 字典,里面有两个元素  'tr'和'te'，分别存放训练数据索引和测试数据索引
    :return: labels:所有数据的标签，0和1 int型变量
    '''
    #print("开始数据集划分")
    file_names = os.listdir(os.path.join(src_data_folder, 'haralick'))  # 列出所有的名字
    # 在目标目录下创建文件夹
    split_names = ['train','test']

    # 删除上次训练的 train  test文件夹
    for split_name in split_names:
        split_path = os.path.join(target_data_folder, split_name)
        if os.path.exists(split_path):
            shutil.rmtree(split_path)

    # 创建本次训练的文件夹
    for split_name in split_names:
        split_path = os.path.join(target_data_folder, split_name)
        if os.path.isdir(split_path):
            pass
        else:
            os.mkdir(split_path)
        # 然后在的目录下创建haralick和morphology文件夹
        if os.path.exists(os.path.join(split_path, 'haralick')):
            pass
        else:
            os.mkdir(os.path.join(split_path, 'haralick'))

        if os.path.exists(os.path.join(split_path, 'morphology')):
            pass
        else:
            os.mkdir(os.path.join(split_path, 'morphology'))

    # 按照比例划分数据集，并进行数据复制
    file_names = np.array(file_names)
    train_list = file_names[train]   # 训练集.mat文件列表
    test_list = file_names[test]     # 测试集.mat文件列表
    file_names.tolist()

    # print("*********************************开始划分数据集*************************************")
    # print("训练集{}".format(len(train_list)))
    # print("测试集{}".format(len(test_list)))

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
    #print("*********************************结束划分数据集*************************************")

    # 获得聚类中心
    path_train = os.path.join(target_data_folder, 'train')  # 训练集数据
    path_test = os.path.join(target_data_folder, 'test')  # 测试集数据
    if os.path.exists(os.path.join(center_folder,'center_haralick.mat')):
        center_haralick = loadmat(os.path.join(center_folder,'center_haralick.mat'))['center_haralick']
        center_morphology = loadmat(os.path.join(center_folder,'center_morphology.mat'))['center_morphology']
    else:
        center_haralick, center_morphology = get_center(src_data_folder,num_center=10, scale=0.4)  # 从全部样本中抽取 40%计算聚类中心
        savemat(os.path.join(center_folder,'center_haralick.mat'),{'center_haralick':center_haralick})
        savemat(os.path.join(center_folder, 'center_morphology.mat'), {'center_morphology': center_morphology})

    data_train_list = []
    data_test_list = []
    data_all_list = []
    idx_dict = {}
    labels = []  # N和C是0  T和D是1

    # 按模态处理数据
    p_tr_haralick = os.path.join(path_train, 'haralick')
    p_tr_morphology = os.path.join(path_train, 'morphology')


    p_te_haralick = os.path.join(path_test, 'haralick')
    p_te_morphology = os.path.join(path_test, 'morphology')

    tr_haralick = glob.glob(os.path.join(p_tr_haralick, '*.mat'))      # 所有训练haralick文件名
    tr_morphology = glob.glob(os.path.join(p_tr_morphology, '*.mat'))  # 所有训练morphology文件名
    te_haralick = glob.glob(os.path.join(p_te_haralick, '*.mat'))      # 所有测试haralick文件名
    te_morphology = glob.glob(os.path.join(p_te_morphology, '*.mat'))  # 所有测试morphology文件名

    label_tr = []



    # 生成data_train_list
    flag = 0
    for i in range(len(tr_haralick)):
        temp_h = loadmat(tr_haralick[i])['haralick']
        temp_m = loadmat(tr_morphology[i])['morphology']
        f_h, f_m = get_feature(temp_h, temp_m, center_haralick, center_morphology)
        n = tr_haralick[i].split('\\')[-1][0]  # 获得首字母

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

    # 对数据进行标准化处理
    ss_tr_h = MaxAbsScaler()
    ss_tr_m = MaxAbsScaler()
    ss_tr_h.fit(data_train_list[0])
    ss_tr_m.fit(data_train_list[1])

    data_train_list[0] = ss_tr_h.transform(data_train_list[0])
    data_train_list[1] = ss_tr_m.transform(data_train_list[1])



    label_te = []
    # 生成data_test_list
    flag = 0
    for i in range(len(te_haralick)):
        temp_h = loadmat(te_haralick[i])['haralick']
        temp_m = loadmat(te_morphology[i])['morphology']
        f_h, f_m = get_feature(temp_h, temp_m, center_haralick, center_morphology)
        n = te_haralick[i].split('\\')[-1][0]  # 获得首字母
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

    # 对测试数据进行标准化处理
    data_test_list[0] = ss_tr_h.transform(data_test_list[0])
    data_test_list[1] = ss_tr_m.transform(data_test_list[1])


    data_all_list.append(np.concatenate((data_train_list[0], data_test_list[0]), axis=0))
    data_all_list.append(np.concatenate((data_train_list[1], data_test_list[1]), axis=0))


    idx_dict["tr"] = list(range(len(label_tr)))
    idx_dict["te"] = list(range(len(label_tr), len(label_tr)+len(label_te)))
    labels = np.concatenate((label_tr, label_te))  # 所有的标签
    return data_train_list, data_all_list, idx_dict, labels


def gen_trvalte_adj_mat(data_tr_list, data_val_list,data_test_list,adj_parameter,eps=1e-8):  # 生成训练集和测试集的邻接矩阵
    adj_metric = "cosine"  # cosine distance
    adj_train_list = []
    adj_val_list = []
    adj_test_list = []
    for i in range(len(data_tr_list)):  # 对每个模态都建立邻接矩阵
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
            model_dict["E{:}".format(i + 1)](data_list[i], adj_list[i]))  # 每个模态都是 GCN 再接一个简单的全连接分类器
        label_s = label_smoothing(one_hot_label)
        ci_loss = torch.mean(criterion(ci, label_s))
        ci_loss.backward()
        optim_dict["C{:}".format(i + 1)].step()
        loss_dict["C{:}".format(i + 1)] = ci_loss.detach().cpu().numpy().item()


    # 使用 attention 机制融合
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
                       num_epoch, dim_he_list):  # train:训练样本索引    test:测试样本索引

    num_view = len(view_list)
    result = {}
    val_result = {}  # 验证集最优结果
    val_result['ACC'] = 0
    val_result['AUC'] = 0
    val_result['F1'] = 0

    a = data_tr_list[0].shape[1]
    b = data_tr_list[1].shape[1]
    dim_list = [a, b]  # 每一个模态的特征数

    model_dict_mymodel = init_model_dict_mymodel(2, num_class, dim_list, dim_he_list)

    for m in model_dict_mymodel:
        if torch.cuda.is_available():
            model_dict_mymodel[m].cuda()
    #print("\nPretrain GCNs...")
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, 2)
    optim_dict = init_optim_mymodel(num_view, model_dict_mymodel, lr_e)



    for epoch in range(num_epoch + 1):
        loss_dict =train_epoch_mymodel(data_tr_list, adj_tr_list,
                    onehot_labels_tr_tensor, model_dict_mymodel, optim_dict)

        # MORONET验证阶段

        val_prob = val_epoch_mymodel(data_tr_list, data_val_list, adj_val_list, model_dict_mymodel)
        label_val = labels_val_tensor.cpu().numpy()
        val_acc = accuracy_score(label_val, val_prob.argmax(1))
        val_f1 = f1_score(label_val, val_prob.argmax(1))
        #val_auc = roc_auc_score(label_val, val_prob[:, 1])
        if val_acc > val_result['ACC']:  # 记录在验证集上取得最优时测试集的结果
            val_result['ACC'] = val_acc
            val_result['F1'] = val_f1
            val_result['AUC'] = 0
            label_test = labels_te_tensor.cpu().numpy()
            te_prob = val_epoch_mymodel(data_tr_list,data_te_list,adj_te_list, model_dict_mymodel)
            result["predict"] = te_prob.argmax(1)[0]
            result["truth"] = label_test[0]
            result["score"] = te_prob[0,1]


    return result