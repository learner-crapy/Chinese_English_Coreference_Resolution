# coding=utf-8
import os
import logging
import time

import numpy as np
import sklearn
import torch
import torch.nn as nn
from matplotlib import pyplot as plt, rcParams
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

import config
import dataset
import CRModel
from utils import utils
from utils import early_stop
from preprocess import CRBertFeature, get_data, CRProcessor, CRProcessor_BIG, get_data_big
from tensorboardX import SummaryWriter

args = config.Args().get_parser()
utils.set_seed(args.seed)
logger = logging.getLogger(__name__)
utils.set_logger(os.path.join(args.log_dir, 'main.log'))
from inception import InceptionModel


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


class BertForCR:
    def __init__(self, model, args):
        self.args = args
        self.model = model
        gpu_ids = args.gpu_ids.split(',')
        self.device = torch.device("cpu" if gpu_ids[0] == '-1' else "cuda:" + gpu_ids[0])
        self.criterion = nn.CrossEntropyLoss()
        self.earlyStopping = early_stop.EarlyStopping(
            monitor='f1',
            patience=1000000,
            verbose=True,
            mode='max',
        )

    def build_optimizer_and_scheduler(self, t_total):
        module = (
            self.model.module if hasattr(model, "module") else self.model
        )

        # 差分学习率
        no_decay = ["bias", "LayerNorm.weight"]
        model_param = list(module.named_parameters())

        bert_param_optimizer = []
        other_param_optimizer = []

        for name, para in model_param:
            space = name.split('.')
            # print(name)
            if space[0] == 'bert_module':
                bert_param_optimizer.append((name, para))
            else:
                other_param_optimizer.append((name, para))

        optimizer_grouped_parameters = [
            # bert other module
            {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay": self.args.weight_decay, 'lr': self.args.lr},
            {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0, 'lr': self.args.lr},

            # 其他模块，差分学习率
            {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay": self.args.weight_decay, 'lr': self.args.other_lr},
            {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0, 'lr': self.args.other_lr},
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.lr, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(self.args.warmup_proportion * t_total), num_training_steps=t_total
        )
        return optimizer, scheduler

    def train(self, train_loader, dev_loader=None):
        self.model.to(self.device)

        global_step = 0
        flag = False
        t_total = self.args.train_epochs * len(train_loader)
        eval_step = 100
        optimizer, scheduler = self.build_optimizer_and_scheduler(t_total)
        best_f1 = 0.0
        stop_count = 0
        stop_dev_loss = float('-inf')
        # 每次读取的数据量
        each_time_lines = self.args.each_time_lines
        # 为了节约内存，这个数据需要手动统计
        lines = self.args.sum_lines
        ################################
        train_data = []
        train_label = []
        dev_data = []
        dev_label = []
        ################################
        for epoch in range(1, 2):
            # try:
            for step, batch_data in enumerate(train_loader):
                self.model.train()
                for key in batch_data.keys():
                    batch_data[key] = batch_data[key].to(self.device)
                # input_data = batch_data['token_ids'], batch_data['attention_masks'], batch_data[
                #     'token_type_ids'], batch_data['span1_ids'], batch_data['span2_ids'],
                train_output = self.model(batch_data['token_ids'], batch_data['attention_masks'], batch_data[
                    'token_type_ids'], batch_data['span1_ids'], batch_data['span2_ids'],).cpu().detach().numpy()

                train_data.append(train_output)
                train_label.append(batch_data['label'].cpu().detach().numpy())
                # train_label.append(batch_data['label'])
                print("epoch:{}, step:{}/{}]".format(epoch,step, t_total))

        # for i in range(len(train_data)):
        #     train_data[i] = torch.stack([torch.tensor(x).clone() for x in train_data[i]], dim=0)

        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            for eval_step, dev_batch_data in enumerate(dev_loader):
                for key in dev_batch_data.keys():
                    dev_batch_data[key] = dev_batch_data[key].to(self.device)
                dev_output = self.model(dev_batch_data['token_ids'],
                                    dev_batch_data['attention_masks'],
                                    dev_batch_data['token_type_ids'],
                                    dev_batch_data['span1_ids'],
                                    dev_batch_data['span2_ids']).cpu().detach().numpy()
                print("eval_step:{}/{}]".format(eval_step, len(dev_loader)))
                dev_label.append(dev_batch_data['label'].cpu().detach().numpy())
                dev_data.append(dev_output)

        train_data = np.concatenate(train_data, axis=0).reshape(-1, 2048) # 1536
        train_label = np.concatenate(train_label, axis=0).reshape(-1, )
        dev_data = np.concatenate(dev_data, axis=0).reshape(-1, 2048) # 1536
        dev_label = np.concatenate(dev_label, axis=0).reshape(-1, )
        from machine import supervised_kmeans_clustering, svm_classification, decision_tree_classification, linear_regression_prediction
        supervised_kmeans_clustering(vectors=train_data, labels=train_label,
                                     dev_vectors=dev_data, dev_labels=dev_label, num_clusters=2,
                                     batch_size=train_data.shape[0], epochs=1)
        svm_classification(train_data, train_label, dev_data, dev_label,
                           num_classes=2, batch_size=train_data.shape[0], epochs=1)

        # In[52]:

        decision_tree_classification(train_data, train_label, dev_data, dev_label, num_classes=2, batch_size=train_data.shape[0], epochs=1)

        # In[53]:

        # linear_regression_prediction(train_data, train_label, dev_data, dev_label, num_classes=2, batch_size=train_data.shape[0], epochs=1)
            # except:
            #     print("error")
            #     continue

        SumWriter.close()
    def DrawFusionMatrix(self, classes, confusion_matrix, Title, savepath, fig_size=(7, 7), fraction=0.0453):

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
    def dev(self, dev_loader):
        self.model.eval()
        self.model.to(self.device)
        total_loss = 0.0
        trues = []
        preds = []
        with torch.no_grad():
            for eval_step, dev_batch_data in enumerate(dev_loader):
                for key in dev_batch_data.keys():
                    dev_batch_data[key] = dev_batch_data[key].to(self.device)
                output = self.model(dev_batch_data['token_ids'],
                                    dev_batch_data['attention_masks'],
                                    dev_batch_data['token_type_ids'],
                                    dev_batch_data['span1_ids'],
                                    dev_batch_data['span2_ids'])
                label = dev_batch_data['label']
                loss = self.criterion(output, label)
                total_loss += loss.item()
                labels = label.cpu().detach().numpy().tolist()
                logits = np.argmax(output.cpu().detach().numpy().tolist(), -1)
                preds.extend(logits)
                trues.extend(labels)

                true_labels = np.array(trues)
                predicted_labels = np.array(preds)
                confusion_matrix = sklearn.metrics.confusion_matrix(true_labels, predicted_labels)
                self.DrawFusionMatrix(classes=['0', '1'], confusion_matrix=confusion_matrix, Title='confusion_matrix',
                                      savepath='./confusion_matrix/confusion_matrix.jpg')

            metrics = Metrics(trues, preds)
            accuracy = metrics.accuracy()
            precision = metrics.precision()
            recall = metrics.recall()
            f1 = metrics.f1()
            return total_loss, accuracy, precision, recall, f1


if __name__ == '__main__':
    from ReadAndWrite import RAW

    raw = RAW()
    path_time = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
    log_dir_path = os.path.join('./log_dir/', path_time)

    if raw.IfFolderExists(log_dir_path):
        pass
    else:
        raw.CreateFolder(log_dir_path)

    SumWriter = SummaryWriter(logdir=log_dir_path)
    processor = CRProcessor()
    train_features = get_data(processor, 'train.json', 'train', args)
    train_dataset = dataset.CRDataset(train_features)
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.train_batch_size,
                              sampler=train_sampler,
                              num_workers=2)
    dev_features = get_data(processor, 'dev.json', 'dev', args)
    dev_dataset = dataset.CRDataset(dev_features)
    dev_loader = DataLoader(dataset=dev_dataset,
                            batch_size=args.eval_batch_size,
                            num_workers=2)

    # test_features = dev_features
    # test_dataset = dataset.CRDataset(test_features)
    # test_loader = DataLoader(dataset=test_dataset,
    #                          batch_size=args.eval_batch_size,
    #                          num_workers=2)
    from machine import Machine
    model = Machine(Shape=(args.train_batch_size, 1, 768), args=args)
    #
    # from lenet5 import Lenet5
    # model = Lenet5(Shape=(args.train_batch_size, 13, 768), args=args)

    # model = InceptionModel(input_shape=(args.train_batch_size, 13, 768), args=args)
    # model = CRModel.CorefernceResolutionModel(args)
    bertForCR = BertForCR(model, args)

    # ===================================
    bertForCR.train(train_loader, dev_loader)
