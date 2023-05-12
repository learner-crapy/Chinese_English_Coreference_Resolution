import csv
import os

from matplotlib import pyplot as plt


def getFilePathList(path, filetype):
    pathList = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(filetype):
                pathList.append(os.path.join(root, file))
    return pathList  # 输出以filetype为后缀的列表


# 写一个函数，提取出****.csv的文件，并返回一个列表file_type for exp:'accuracy.csv'
def getAccuracyFileList(FileList, file_type):
    acc_file_list = []
    for filename in FileList:
        if file_type in filename:
            acc_file_list.append(filename)
    return acc_file_list


# 写一个函数，读取csv文件
def readCsv(filepath):
    data = []
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    return data


# 　参数解释，
def PltLineChart(data_dic, title, HoriLine, savepath, x_label, y_label):
    # 获取数据的键和值
    keys = list(data_dic.keys())
    values = list(data_dic.values())
    # 绘制折线图
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
    plt.show()

# 写一个函数，从一个列表中选取部分值作为新的列表，每隔k个值取一个值，返回新列表



# if __name__ == "__main__":
#     FileList = getFilePathList("./log_dir_5_5_2d_chinese", "csv")
#     print(FileList)
#     print(getAccuracyFileList(FileList))
