import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import confusion_matrix
from ReadAndWrite import RAW
raw = RAW()


def DrawFusionMatrix(classes, confusion_matrix, Title, savepath, fig_size=(3, 3), fraction=0.0453):
    plt.figure(figsize=(fig_size))
    proportion = []
    for i in confusion_matrix:
        for j in i:
            temp = j / (np.sum(i))
            proportion.append(temp)
    # print(np.sum(confusion_matrix[0]))
    # print(proportion)
    pshow = []
    for i in proportion:
        pt = "%.2f%%" % (i * 100)
        pshow.append(pt)
    proportion = np.array(proportion).reshape(confusion_matrix.shape[0],
                                              confusion_matrix.shape[1])  # reshape(列的长度，行的长度)
    pshow = np.array(pshow).reshape(confusion_matrix.shape[0], confusion_matrix.shape[1])
    # print(pshow)
    config = {
        "font.family": 'Times New Roman',  # 设置字体类型
    }
    rcParams.update(config)
    plt.imshow(proportion, interpolation='nearest', cmap=plt.cm.Blues)  # 按照像素显示出矩阵
    # (改变颜色：'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds','YlOrBr', 'YlOrRd',
    # 'OrRd', 'PuRd', 'RdPu', 'BuPu','GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn')
    plt.title(Title)
    plt.colorbar(fraction=fraction)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=11)
    plt.yticks(tick_marks, classes, fontsize=11)

    thresh = confusion_matrix.max() / 2.
    # iters = [[i,j] for i in range(len(classes)) for j in range((classes))]
    # ij配对，遍历矩阵迭代器
    iters = np.reshape([[[i, j] for j in range(confusion_matrix.shape[0])] for i in range(confusion_matrix.shape[0])],
                       (confusion_matrix.size, 2))
    for i, j in iters:
        if (i == j):
            plt.text(j, i - 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=11, color='white',
                     weight=5)  # 显示对应的数字
            plt.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=11, color='white')
        else:
            plt.text(j, i - 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=11)  # 显示对应的数字
            plt.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=11)

    plt.ylabel('True label', fontsize=16)
    plt.xlabel('Predict label', fontsize=16)
    plt.tight_layout()
    plt.savefig(savepath)
    plt.show()


def confusion_matrix_indicator(matrix):
    '''
    tn|fp
    fn|tp
    '''
    tn = matrix[0][0]
    fp = matrix[0][1]
    fn = matrix[1][0]
    tp = matrix[1][1]
    acc = (tp + tn) / (tp + fp + fn + tn)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * precision * recall / (precision + recall)
    return acc, recall, precision, f1

# confusion_matrix = confusion_matrix(true_labels, predicted_labels)
# DrawFusionMatrix(classes=['0', '1'], confusion_matrix=confusion_matrix, Title='Title',
#                      savepath='./2d.jpg')


data_list_2d = [[[376, 265], [273, 472]],
                [[40, 601], [23, 722]],
                [[546, 95], [211, 534]],
                [[372, 269], [146, 599]],
                [[465, 326], [320, 572]],
                [[102, 689], [87, 805]],
                [[504, 287], [242, 650]],
                [[471, 320], [334, 558]]
                ]
titles_2d = ['Inception_chinese', 'vgg16_chinese', 'LSTM_chinese', 'LeNet5_chinese', 'Inception_english',
             'vgg16_english', 'LSTM_english', 'LeNet5_english']

data_list_1d = [[[45, 614], [26, 726]],
                [[45, 614], [26, 726]],
                [[0, 659], [0, 752]],
                [[45, 614], [24, 728]],
                [[0, 614], [0, 745]],
                [[51, 740], [36, 856]],
                [[0, 791], [0, 892]],
                [[0, 791], [0, 892]],
                [[47, 744], [33, 859]],
                [[0, 791], [0, 892]]
                ]
titles_1d = ['Inception_chinese', 'vgg16_chinese', 'LSTM_chinese', 'LeNet5_chinese', 'TextCNN_chinese', 'Inception_english', 'vgg16_english', 'LSTM_english', 'LeNet5_english', 'TextCNN_english']

data_list_CRModel = [[[396, 245], [230, 515]],
                     [[513, 278], [166, 726]],
                     [[427, 214], [208, 537]],
                     [[498, 293], [144, 748]]
                     ]

data_list_CRModel_titles = ['CRModel_chinese', 'CRModel_2dense_chinese', 'CRModel_english', 'CRModel_2dense_english']

data_list_LSTM_CRModel = [[[603, 195], [141, 707]],
                          [[664, 184], [101, 747]],
                          [[686, 312], [207, 788]],
                          [[631, 367], [153, 842]],
                          [[690, 158], [106, 742]],
                          [[599, 399], [273, 722]]
                          ]
data_list_LSTM_CRModel_titles = ['CRModel_chinese', 'CRModel_2dense_chinese', 'CRModel_english', 'CRModel_2dense_english', 'LSTM_chinese', 'LSTM_english']

def write_indicator(data_2d, title_list, save_path):
    Indicator = []
    for i in range(0, len(data_2d)):
        OneLine = []
        confusion_matrix = np.array(data_2d[i])
        acc, recall, precision, f1 = confusion_matrix_indicator(confusion_matrix)
        acc = round(acc, 2)
        recall = round(recall, 2)
        precision = round(precision, 2)
        f1 = round(f1, 2)
        OneLine.append(acc)
        OneLine.append(recall)
        OneLine.append(precision)
        OneLine.append(f1)
        Indicator.append(OneLine)
        print(title_list[i] + ' acc: ' + str(acc) + ' recall: ' + str(recall) + ' precision: ' + str(
            precision) + ' f1: ' + str(f1))

    np.savetxt(save_path, np.array(Indicator), delimiter=',')


data = [data_list_2d, data_list_1d, data_list_CRModel, data_list_LSTM_CRModel]
titles = [titles_2d, titles_1d, data_list_CRModel_titles, data_list_LSTM_CRModel_titles]
save_paths = ['2d.csv', '1d.csv', 'CRModel.csv', 'LSTM_CRModel.csv']
for i in range(0, len(data)):
    write_indicator(data[i], titles[i], save_paths[i])


# for i in range(0, len(data_list_2d)):
#     OneLine = []
#     confusion_matrix = np.array(data_list_2d[i])
    # Title = titles_2d[i] + '_confusion_matrix'
#
#     DrawFusionMatrix(classes=['0', '1'], confusion_matrix=confusion_matrix, Title=Title,
#                      savepath='./new_confutin_matrix/' + Title + '_2d.jpg')



# # 写一个函数，输入真是标签列表和预测标签列表，返回混淆矩阵
# for i in range(0, len(data_list_1d)):
#     confusion_matrix = np.array(data_list_1d[i])
#     acc, recall, precision, f1 = confusion_matrix_indicator(confusion_matrix)
#     print(titles_1d[i] + ' acc: ' + str(acc) + ' recall: ' + str(recall) + ' precision: ' + str(precision) + ' f1: ' + str(f1))

#     Title = titles_1d[i] + '_confusion_matrix'
#
#     DrawFusionMatrix(classes=['0', '1'], confusion_matrix=confusion_matrix, Title=Title,
#                      savepath='./new_confutin_matrix/' + Title + '_1d.jpg')

# # 写一个函数，输入真是标签列表和预测标签列表，返回混淆矩阵
# for i in range(0, len(data_list_CRModel)):
#     confusion_matrix = np.array(data_list_CRModel[i])
#     acc, recall, precision, f1 = confusion_matrix_indicator(confusion_matrix)
#     print(titles_2d[i] + ' acc: ' + str(acc) + ' recall: ' + str(recall) + ' precision: ' + str(precision) + ' f1: ' + str(f1))

#     Title = data_list_CRModel_titles[i] + '_confusion_matrix'
#
#     DrawFusionMatrix(classes=['0', '1'], confusion_matrix=confusion_matrix, Title=Title,
#                      savepath='./new_confutin_matrix/' + Title + '_1d.jpg')

# for i in range(0, len(data_list_LSTM_CRModel)):
#     confusion_matrix = np.array(data_list_LSTM_CRModel[i])
#     acc, recall, precision, f1 = confusion_matrix_indicator(confusion_matrix)
#     print(titles_2d[i] + ' acc: ' + str(acc) + ' recall: ' + str(recall) + ' precision: ' + str(precision) + ' f1: ' + str(f1))
#
#     Title = data_list_LSTM_CRModel_titles[i] + '_confusion_matrix'
#
#     DrawFusionMatrix(classes=['0', '1'], confusion_matrix=confusion_matrix, Title=Title,
#                      savepath='./LSTM_CRModel/' + Title + '_1d.jpg')