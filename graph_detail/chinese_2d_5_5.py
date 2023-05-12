#!/usr/bin/env python
# coding: utf-8

# In[5]:
import numpy as np

from my_utils import *
import pandas as pd
from ReadAndWrite import RAW
raw = RAW()
# In[6]:


chinese_2d_5_5_csv_filelist = getFilePathList("./log_dir_5_5_2d_chinese", "csv")


# In[7]:


chinese_2d_5_5_csv_filelist = getFilePathList("./log_dir_5_5_2d_chinese", "csv")

#recall.csv  train_acc.csv train_loss.csv f1.csv
dev_acc_filelist = getAccuracyFileList(chinese_2d_5_5_csv_filelist, 'accuracy.csv')
dev_loss_filelist = getAccuracyFileList(chinese_2d_5_5_csv_filelist, 'dev_loss.csv')
precision_filelist = getAccuracyFileList(chinese_2d_5_5_csv_filelist, 'precision.csv')
recall_file_filelist = getAccuracyFileList(chinese_2d_5_5_csv_filelist, 'recall.csv')
train_acc_filelist = getAccuracyFileList(chinese_2d_5_5_csv_filelist, 'train_acc.csv')
train_loss_filelist = getAccuracyFileList(chinese_2d_5_5_csv_filelist, 'train_loss.csv')
f1_filelist = getAccuracyFileList(chinese_2d_5_5_csv_filelist, 'f1.csv')


# In[11]:




# In[34]:


# 第k1行到第k2行，第j1列到第j2列的子列表
def getSubList(data, k1, k2, j1,j2):
    sub_List = []
    for row in data[k1:k2]:
        sub_List.append(row[j1:j2])
    return sub_List


# In[36]:


data = readCsv(dev_acc_filelist[0])
print(len(data))
getSubList(data,1,51,1,3)


# In[10]:


pd.read_csv(dev_acc_filelist[0])


# In[12]:


#　参数解释，
def PltLineChart(data_dic, title, HoriLine, savepath, x_label, y_label):
    # 获取数据的键和值
    keys = list(data_dic.keys())
    values = list(data_dic.values())
    # 绘制折线图
    fig = plt.figure()
    for i in range(len(keys)):
        # 绘制折线图
        plt.plot(HoriLine, values[i], label=keys[i])
        # 添加数据点标识
#         for j in range(len(HoriLine)):
#             plt.annotate(str(values[i][j]), xy=(HoriLine[j], values[i][j]))
    # 添加标题和标签
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # 添加图例
    plt.legend()
    plt.savefig(savepath)
    # 显示图形
    # plt.show()


# In[ ]:
# 第k1行到第k2行，第j1列到第j2列的子列表
def getSubList(data, k1, k2, j1,j2):
    sub_List = []
    for row in data[k1:k2]:
        sub_List.append(row[j1:j2])
    return sub_List

# 返回行数和列数
def get_num_rows_cols(lst):
    num_rows = len(lst)
    num_cols = len(lst[0]) if num_rows else 0
    return num_rows, num_cols

def selectValues_mean(lst, k):
    new_lst = []
    for i in range(0, len(lst), k):
        if i != 0:
            new_lst.append(np.mean(lst[i:i+k]))
        else:
            new_lst.append(lst[i])
    return new_lst

def selectValues(lst, k):
    new_lst = []
    for i in range(0, len(lst), k):
        new_lst.append(lst[i])
    return new_lst

def getMinList(data_list):
    """
    Returns the one dimensional list from a two dimensional list that has the smallest length.

    Args:
    data_list (list): a two dimensional list whose one dimensional lists need to be checked

    Returns:
    list: the one dimensional list from the input list with the smallest length
    """
    min_list = data_list[0]
    for lst in data_list:
        if len(lst) < len(min_list):
            min_list = lst
    return min_list

indecators = {'train_acc':train_acc_filelist, 'train_loss':train_loss_filelist, 'dev_acc':dev_acc_filelist, 'dev_loss':dev_loss_filelist, 'recall':recall_file_filelist, 'precsion':precision_filelist, 'f1':f1_filelist}
models = ['TextCNN', 'Inception', 'Lenet5', 'LSTM', 'VGG16']

steps = []
for i in range(0, len(indecators)):
    line_value = {"TextCNN": [], "Inception": [], "Lenet5": [], "LSTM": [], "VGG16": []}
    for file in indecators[list(indecators.keys())[i]]:
        data = readCsv(file)
        num_rows, num_cols = get_num_rows_cols(data)
        if 'TextCNN' in file:
            steps.append(getSubList(data, 1, num_rows, 1, 2))
            line_value['TextCNN'].append(getSubList(data, 1, num_rows, 2, 3))
        elif 'inception' in file:
            steps.append(getSubList(data, 1, num_rows, 1, 2))
            line_value['Inception'].append(getSubList(data, 1, num_rows, 2, 3))
        elif 'Lenet5' in file:
            steps.append(getSubList(data, 1, num_rows, 1, 2))
            line_value['Lenet5'].append(getSubList(data, 1, num_rows, 2, 3))
        elif 'Lstm' in file:
            steps.append(getSubList(data, 1, num_rows, 1, 2))
            line_value['LSTM'].append(getSubList(data, 1, num_rows, 2, 3))
        elif 'vgg16' in file:
            steps.append(getSubList(data, 1, num_rows, 1, 2))
            line_value['VGG16'].append(getSubList(data, 1, num_rows, 2, 3))


    for j in range(0, len(steps)):
        # steps[j] = sum(steps[j], [])
        steps[j] = np.array(steps[j]).astype(np.int).squeeze().tolist()

    step = getMinList(steps)

    select_step = 20 if len(step) >= 1000 else 1


    # steps = steps[i] for i in len(steps) if len(steps[i]) == min(steps)
    for key in line_value.keys():
        line_value[key] = np.array(line_value[key]).astype(np.float).squeeze().tolist()[0:len(step)]
        line_value[key] = selectValues_mean(line_value[key], select_step)
    step = selectValues(step, select_step)
    try:
        raw.CreateFolder('./Img/chinese_2d_5_5/')
    except:
        pass
    PltLineChart(line_value, list(indecators.keys())[i], step, './Img/chinese_2d_5_5/'+list(indecators.keys())[i]+'.jpg', 'step', list(indecators.keys())[i])


        

