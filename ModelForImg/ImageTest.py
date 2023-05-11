import time
import matplotlib
matplotlib.use('Agg')
import sklearn
import torch
from matplotlib import rcParams
from tensorboardX import SummaryWriter
from torch.optim import optimizer
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import torchvision
import numpy as np
from torch.autograd import Variable
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
import config

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

args = config.Args().get_parser()

transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
data_train = datasets.MNIST(root="./data/",
                            transform=transform,
                            train=True,
                            download=True)

data_test = datasets.MNIST(root="./data/",
                           transform=transform,
                           train=False)

data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                batch_size=64,
                                                shuffle=True)

data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                               batch_size=64,
                                               shuffle=True)

from vgg16 import VGG16

model = VGG16(num_classes=10)


class Metrics:
    def __init__(self, trues, preds):
        self.trues = trues
        self.preds = preds

    def accuracy(self):
        return sum([1 for i in range(len(self.trues)) if self.trues[i] == self.preds[i]]) / len(self.trues)

    def precision(self):
        tp = sum([1 for i in range(len(self.trues)) if self.trues[i] == 1 and self.preds[i] == 1])
        fp = sum([1 for i in range(len(self.trues)) if self.trues[i] == 0 and self.preds[i] == 1])
        return tp / (tp + fp)

    def recall(self):
        tp = sum([1 for i in range(len(self.trues)) if self.trues[i] == 1 and self.preds[i] == 1])
        fn = sum([1 for i in range(len(self.trues)) if self.trues[i] == 1 and self.preds[i] == 0])
        return tp / (tp + fn)

    def f1(self):
        p = self.precision()
        r = self.recall()
        return 2 * p * r / (p + r)


def DrawFusionMatrix(classes, confusion_matrix, Title, savepath, fig_size=(7, 7), fraction=0.0453):
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
        "font.family": 'DejaVu Sans',  # 设置字体类型
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
    iters = np.reshape(
        [[[i, j] for j in range(confusion_matrix.shape[0])] for i in range(confusion_matrix.shape[0])],
        (confusion_matrix.size, 2))
    for i, j in iters:
        if (i == j):
            plt.text(j, i - 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=11,
                     color='white', weight=5)  # 显示对应的数字
            plt.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=11, color='white')
        else:
            plt.text(j, i - 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=11)  # 显示对应的数字
            plt.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=11)

    plt.ylabel('True label', fontsize=16)
    plt.xlabel('Predict label', fontsize=16)
    plt.tight_layout()
    plt.savefig(savepath)


if __name__ == '__main__':
    from ReadAndWrite import RAW

    # from method_explore import SOMEUTILS
    #
    # someutils = SOMEUTILS(args, 'en')

    raw = RAW()
    path_time = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
    log_dir_path = os.path.join('./log_dir/', path_time)

    if raw.IfFolderExists(log_dir_path):
        pass
    else:
        raw.CreateFolder(log_dir_path)

    SumWriter = SummaryWriter(logdir=log_dir_path)

    # ###################################################################### VGG16 ######################################################################
    from vgg16 import VGG16
    model = VGG16(num_classes=10)
    confusion_matrix_folder = 'vgg16-confusion-matrix'
    # ###################################################################### inception ######################################################################
    from inception import Net
    model = Net()
    confusion_matrix_folder = 'inception-confusion-matrix'
    # ###################################################################### LeNet5 ######################################################################
    from lenet5 import LeNet5
    model = LeNet5()
    confusion_matrix_folder = 'lenet5-confusion-matrix'
    # ###################################################################### lstm ######################################################################
    from lstm import LSTM
    model = LSTM(input_size=1024 * 3, hidden_size=1024, num_layers=8, num_classes=2, args=args)
    confusion_matrix_folder = 'lenet5-confusion-matrix'
    # ######################################################################  ######################################################################
    # ######################################################################  ######################################################################
    # ######################################################################  ######################################################################
    # ######################################################################  ######################################################################

    if raw.IfFolderExists(confusion_matrix_folder) == False:
        raw.CreateFolder(confusion_matrix_folder)

    cost = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    n_epochs = 10
    for epoch in range(n_epochs):
        running_loss = 0.0
        running_correct = 0
        print("Epoch {}/{}".format(epoch, n_epochs))
        print("-" * 10)
        for step, data in enumerate(data_loader_train):
            # try:
            X_train, y_train = data
            X_train, y_train = Variable(X_train), Variable(y_train)
            outputs = model(X_train)
            _, pred = torch.max(outputs.data, 1)
            optimizer.zero_grad()
            loss = cost(outputs, y_train)
            true_labels = y_train.tolist()
            predicted_labels = pred.tolist()

            confusion_mat = sklearn.metrics.confusion_matrix(np.array(true_labels), np.array(predicted_labels))

            DrawFusionMatrix(classes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ],
                             confusion_matrix=confusion_mat, Title='confusion_matrix',
                             savepath=confusion_matrix_folder + '/confusion_matrix.jpg')

            metrics = Metrics(true_labels, predicted_labels)
            try:
                train_acc, recall, precision, f1 = metrics.accuracy(), metrics.recall(), metrics.precision(), metrics.f1()
                print("epoch:{}/{}, setp:{}, loss:{}, train_acc:{}, recall:{}, precision:{}, f1:{}".format(epoch,
                                                                                                           n_epochs,
                                                                                                           step, loss,
                                                                                                           train_acc,
                                                                                                           recall,
                                                                                                           precision,
                                                                                                           f1))
            except:
                print("epoch:{}/{}, setp:{}, loss:{}".format(epoch, n_epochs, step, loss),
                      '-----------all samples were categrayed into one class----------------')
            loss.backward()
            optimizer.step()
            # running_loss += loss.data[0]
            running_loss += loss.item()
                # running_correct += torch.sum(pred == y_train.data)
            # except:
            #     print('error')
            #     continue

            if (step != 0) & (step % 100 == 0):
                with torch.no_grad():
                    for data in data_loader_test:
                        try:
                            X_test, y_test = data
                            X_test, y_test = Variable(X_test.view(64, 1, 28 * 28)), Variable(y_test)
                            outputs = model(X_test)
                            loss = cost(outputs, y_test)
                            _, pred = torch.max(outputs.data, 1)
                            true_labels = y_test.tolist()
                            predicted_labels = pred.tolist()
                            metrics = Metrics(true_labels, predicted_labels)
                            try:
                                train_acc, recall, precision, f1 = metrics.accuracy(), metrics.recall(), metrics.precision(), metrics.f1()
                                print(
                                    "test, loss:{}, train_acc:{}, recall:{}, precision:{}, f1:{}".format(loss,
                                                                                                         train_acc,
                                                                                                         recall,
                                                                                                         precision,
                                                                                                         f1))
                                SumWriter.add_scalar('dev_loss', loss.item(), step)
                                SumWriter.add_scalar('dev_acc', train_acc, step)
                                SumWriter.add_scalar('recall', recall, step)
                                SumWriter.add_scalar('precision', precision, step)
                                SumWriter.add_scalar('f1', f1, step)
                                # SumWriter.add_image('confusion_matrix', confusion_mat, step)
                                # SumWriter.add_histogram('confusion_matrix', confusion_mat, step)
                            except:
                                print("test, loss:{}".format(loss),
                                      '-----------all samples were categrayed into one class----------------')
                        except:
                            print('test, error')
                            continue
        # testing_correct = 0
        # for data in data_loader_test:
        #     X_test, y_test = data
        #     X_test, y_test = Variable(X_test), Variable(y_test)
        #     outputs = model(X_test)
        #     _, pred = torch.max(outputs.data, 1)
        #     testing_correct += torch.sum(pred == y_test.data)
        # print(
        #     "Loss is:{:.4f}, Train Accuracy is:{:.4f}%, Test Accuracy is:{:.4f}".format(running_loss / len(data_train),
        #                                                                                 100 * running_correct / len(
        #                                                                                     data_train),
        #                                                                                 100 * testing_correct / len(
        #                                                                                     data_test)))
    torch.save(model.state_dict(), "model_parameter.pkl")
