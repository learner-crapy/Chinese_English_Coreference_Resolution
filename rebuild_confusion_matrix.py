import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import confusion_matrix


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


# Generate 100 random true labels (0 or 1)
true_labels = np.array([1, 0, 0, 1, 1, 0, 0])
predicted_labels = np.array([0, 0, 1, 1, 0, 1, 0])

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
# # confusion_matrix = confusion_matrix(true_labels, predicted_labels)
# for i in range(0, len(data_list_2d)):
#     confusion_matrix = np.array(data_list_2d[i])
#
#     Title = titles_2d[i] + '_confusion_matrix'
#
#     DrawFusionMatrix(classes=['0', '1'], confusion_matrix=confusion_matrix, Title=Title,
#                      savepath='./new_confutin_matrix/' + Title + '_2d.jpg')



# # 写一个函数，输入真是标签列表和预测标签列表，返回混淆矩阵
# for i in range(0, len(data_list_1d)):
#     confusion_matrix = np.array(data_list_1d[i])
#
#     Title = titles_1d[i] + '_confusion_matrix'
#
#     DrawFusionMatrix(classes=['0', '1'], confusion_matrix=confusion_matrix, Title=Title,
#                      savepath='./new_confutin_matrix/' + Title + '_1d.jpg')

# 写一个函数，输入真是标签列表和预测标签列表，返回混淆矩阵
for i in range(0, len(data_list_CRModel)):
    confusion_matrix = np.array(data_list_CRModel[i])

    Title = data_list_CRModel_titles[i] + '_confusion_matrix'

    DrawFusionMatrix(classes=['0', '1'], confusion_matrix=confusion_matrix, Title=Title,
                     savepath='./new_confutin_matrix/' + Title + '_1d.jpg')