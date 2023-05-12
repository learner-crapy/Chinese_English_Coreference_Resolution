# coding=utf-8
import os
import logging
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from TextCNN import TextCNN
import config
import dataset
import CRModel
from utils import utils
from utils import early_stop
from preprocess import CRBertFeature, get_data, CRProcessor, CRProcessor_BIG, get_data_big


args = config.Args().get_parser()
utils.set_seed(args.seed)
logger = logging.getLogger(__name__)
utils.set_logger(os.path.join(args.log_dir, 'main.log'))
from tensorboardX import SummaryWriter
import time

# from inception import InceptionModel
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import sklearn


'''
python main --ensemble_model_list ['vgg16'] --en_cn cn --exp_name 8-8-chinese-300-deep-200-epoch --model_type 2d --lr 2e-8 --other_lr 2e-8 --bert_dir ./pre_model/chinese/chinese_roberta_wwm_large_ext_pytorch/ --data_dir ./data/chinese/ --train_batch_size 32 --eval_batch_size 32 --train_epochs 200 --max_seq_len 300 --dropout_prob 0.1
python main --ensemble_model_list ['vgg16'] --en_cn en --exp_name 8-8-english-300-deep-200-epoch --model_type 2d --lr 2e-8 --other_lr 2e-8 --bert_dir ./pre_model/english/bert-base-uncased/ --data_dir ./data/english/ --train_batch_size 32 --eval_batch_size 32 --train_epochs 200 --max_seq_len 300 --dropout_prob 0.1
'''
'''
--ensemble_model_list
LSTM TextCNN vgg16 Inception LeNet5 CRModel CRModel_2dense
--en_cn
cn
--exp_name
confusion_matrix_rebuild-2d
--model_type
2d
--lr
2e-5
--other_lr
2e-5
--bert_dir
./pre_model/chinese/chinese_roberta_wwm_large_ext_pytorch/
--data_dir
./data/chinese/
--train_batch_size
16
--eval_batch_size
16
--train_epochs
200
--max_seq_len
300
--dropout_prob
0.1
'''
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

    def tp(self):
        return sum([1 for i in range(len(self.trues)) if self.trues[i] == 1 and self.preds[i] == 1])

    def fn(self):
        return sum([1 for i in range(len(self.trues)) if self.trues[i] == 1 and self.preds[i] == 0])

    def fp(self):
        return sum([1 for i in range(len(self.trues)) if self.trues[i] == 0 and self.preds[i] == 1])

    def tn(self):
        return sum([1 for i in range(len(self.trues)) if self.trues[i] == 0 and self.preds[i] == 0])


class BertForCR:
    def __init__(self, model, args):
        self.args = args
        self.model = model
        gpu_ids = args.gpu_ids.split(',')
        self.device = torch.device("cpu" if gpu_ids[0] == '-1' else "cuda:" + gpu_ids[0])
        self.criterion = nn.CrossEntropyLoss()
        self.earlyStopping = early_stop.EarlyStopping(
            monitor='f1',
            patience=2000,
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
        eval_step = 2000
        optimizer, scheduler = self.build_optimizer_and_scheduler(t_total)
        best_f1 = 0.0
        stop_count = 0
        stop_dev_loss = float('-inf')
        # 每次读取的数据量
        # each_time_lines = self.args.each_time_lines
        # 为了节约内存，这个数据需要手动统计
        # lines = self.args.sum_lines
        num_confusion_matrix = 0
        for epoch in range(1, self.args.train_epochs + 1):
            # try:
            for step, batch_data in enumerate(train_loader):
                self.model.train()
                for key in batch_data.keys():
                    batch_data[key] = batch_data[key].to(self.device)

                output = self.model(
                    batch_data['token_ids'],
                    batch_data['attention_masks'],
                    batch_data['token_type_ids'],
                    batch_data['span1_ids'],
                    batch_data['span2_ids'],
                )
                targets_numpy = batch_data['label'].cpu().detach().numpy()

                metrics = Metrics(targets_numpy,
                                  np.argsort(output.cpu().detach().numpy())[:, -1])
                train_accuracy = metrics.accuracy()
                train_tp = metrics.tp()
                train_fp = metrics.fp()
                train_tn = metrics.tn()
                train_fn = metrics.fn()
                # print('-------------train_acc=', train_accuracy, '------------------')

                loss = self.criterion(output, batch_data['label'])
                SumWriter.add_scalar('train_loss', loss, global_step)
                SumWriter.add_scalar('train_acc', train_accuracy, global_step)

                self.model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                optimizer.step()
                scheduler.step()

                logger.info(
                    '[train {}] epoch:{}/{} step:{}/{} loss:{:.6f} train_acc:{} tp:{}/{} fp:{} fn:{} tn:{}/{}'.format(
                        experiment, epoch,
                        self.args.train_epochs,
                        global_step,
                        t_total,
                        loss.item(),
                        train_accuracy, train_tp, targets_numpy.tolist().count(1), train_fp, train_fn, train_tn,
                        targets_numpy.tolist().count(0)))
                global_step += 1
                # if global_step % eval_step == 0:
                #     num_confusion_matrix += 1
            try:
                num_confusion_matrix += 1
                dev_loss, accuracy, precision, recall, f1, tp, fp, fn, tn, trues, preds = self.dev(dev_loader,
                                                                                                   num_confusion_matrix)
            except:
                print(
                    "出现了全部分为一类的现象!---------------------------------------------------------------------------------------")
                continue
            SumWriter.add_scalar('dev_loss', dev_loss, global_step)
            SumWriter.add_scalar('accuracy', accuracy, global_step)
            SumWriter.add_scalar('precision', precision, global_step)
            SumWriter.add_scalar('recall', recall, global_step)
            SumWriter.add_scalar('f1', f1, global_step)
            logger.info(
                '[dev {}] loss:{:.6f} accuracy:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f} tp:{}/{} fp:{} fn:{} tn:{}/{}'.format(
                    experiment, dev_loss, accuracy, precision, recall, f1, tp, trues.count(1), fp, fn, tn,
                    trues.count(0)))
            # set the early stop mode to f1 max
            self.earlyStopping(f1, self.model)
            if self.earlyStopping.early_stop and epoch > 1:
                flag = True
                break
            if f1 > best_f1:
                best_f1 = f1

                pt_dir = os.path.join(self.args.output_dir, experiment)
                if raw.IfFolderExists(pt_dir):
                    pass
                else:
                    raw.CreateFolder(pt_dir)

                torch.save(self.model.state_dict(), pt_dir + '/best.pt')
            if flag:
                break
            # except:
            #     print(
            #         "some errors occured!---------------------------------------------------------------------------------------")
            #     # raw.AddWrite('./error.txt', str(k) + ' ' + str(m))
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
        # plt.show()

    def plot_confusion_matrix(self, classes, cm, savename, title='Confusion Matrix'):
        plt.figure(figsize=(12, 8), dpi=100)
        np.set_printoptions(precision=2)

        # 在混淆矩阵中每格的概率值
        ind_array = np.arange(len(classes))
        x, y = np.meshgrid(ind_array, ind_array)
        for x_val, y_val in zip(x.flatten(), y.flatten()):
            c = cm[y_val][x_val]
            if c > 0.001:
                plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')

        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
        plt.title(title)
        plt.colorbar()
        xlocations = np.array(range(len(classes)))
        plt.xticks(xlocations, classes, rotation=90)
        plt.yticks(xlocations, classes)
        plt.ylabel('Actual label')
        plt.xlabel('Predict label')

        # offset the tick
        tick_marks = np.array(range(len(classes))) + 0.5
        plt.gca().set_xticks(tick_marks, minor=True)
        plt.gca().set_yticks(tick_marks, minor=True)
        plt.gca().xaxis.set_ticks_position('none')
        plt.gca().yaxis.set_ticks_position('none')
        plt.grid(True, which='minor', linestyle='-')
        plt.gcf().subplots_adjust(bottom=0.15)

        # show confusion matrix
        plt.savefig(savename, format='png')

    def dev(self, dev_loader, num_confusion_matrix=0):
        self.model.eval()
        self.model.to(self.device)
        total_loss = 0.0
        trues = []
        preds = []
        confusion_matrix_i = 0
        with torch.no_grad():
            for eval_step, dev_batch_data in enumerate(dev_loader):
                # if eval_step >= len(dev_loader)-1:
                #     continue
                # print(eval_step, '/', len(dev_loader))
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
            savepath_fold = './confusion_matrix/' + experiment
            if raw.IfFolderExists(savepath_fold) == False:
                raw.CreateFolder(savepath_fold)
            # confusion_matrix_i += 1

            metrics = Metrics(trues, preds)
            accuracy = metrics.accuracy()
            precision = metrics.precision()
            recall = metrics.recall()
            f1 = metrics.f1()
            tp = metrics.tp()
            fp = metrics.fp()
            tn = metrics.tn()
            fn = metrics.fn()

            Title = "tn-{} fp-{} fn-{} tp-{}".format(tn, fp, fn, tp)

            self.DrawFusionMatrix(classes=['0', '1'], confusion_matrix=confusion_matrix, Title=Title, fig_size=(3, 3),
                                  savepath=savepath_fold + '/confusion_matrix' +str(num_confusion_matrix)+' '+ Title + '.jpg')

            # self.plot_confusion_matrix(['0', '1'], confusion_matrix, savepath_fold + '/confusion_matrix' + Title + '.jpg', Title)
            return total_loss, accuracy, precision, recall, f1, tp, fp, fn, tn, trues, preds

    def test(self, model, test_loader):
        model.eval()
        model.to(self.device)
        total_loss = 0.0
        trues = []
        preds = []
        with torch.no_grad():
            for eval_step, test_batch_data in enumerate(test_loader):
                for key in test_batch_data.keys():
                    test_batch_data[key] = test_batch_data[key].to(self.device)
                output = self.model(test_batch_data['token_ids'],
                                    test_batch_data['attention_masks'],
                                    test_batch_data['token_type_ids'],
                                    test_batch_data['span1_ids'],
                                    test_batch_data['span2_ids'])
                label = test_batch_data['label']
                loss = self.criterion(output, label)
                total_loss += loss.item()
                labels = label.cpu().detach().numpy().tolist()
                logits = np.argmax(output.cpu().detach().numpy().tolist(), -1)
                preds.extend(logits)
                trues.extend(labels)
            logger.info(classification_report(trues, preds))

    def predict(self, model, raw_text, span1, span2, args):
        model.to(self.device)
        model.eval()
        with torch.no_grad():
            tokenizer = BertTokenizer(
                os.path.join(args.bert_dir, 'vocab.txt'))
            tokens = [i for i in raw_text]
            span1_ids = [0] * len(tokens)
            span1_start = span1[1]
            span1_end = span1_start + len(span1[0])

            for i in range(span1_start, span1_end):
                span1_ids[i] = 1
            span2_ids = [0] * len(tokens)
            span2_start = span2[1]
            span2_end = span2_start + len(span2[0])
            for i in range(span2_start, span2_end):
                span2_ids[i] = 1

            if len(span1_ids) <= args.max_seq_len - 2:  # 这里减2是[CLS]和[SEP]
                pad_length = args.max_seq_len - 2 - len(span1_ids)
                span1_ids = span1_ids + [0] * pad_length  # CLS SEP PAD label都为O
                span2_ids = span2_ids + [0] * pad_length
                span1_ids = [0] + span1_ids + [0]  # 增加[CLS]和[SEP]
                span2_ids = [0] + span2_ids + [0]
            else:
                if span2_end > self.max_seq_len - 2:
                    raise Exception('发生了不该有的截断')
                span1_ids = span1_ids[:args.max_seq_len - 2]
                span2_ids = span2_ids[:args.max_seq_len - 2]
                span1_ids = [0] + span1_ids + [0]  # 增加[CLS]和[SEP]
                span2_ids = [0] + span2_ids + [0]

            encode_dict = tokenizer.encode_plus(text=tokens,
                                                max_length=args.max_seq_len,
                                                padding="max_length",
                                                truncation='only_first',
                                                return_token_type_ids=True,
                                                return_attention_mask=True,
                                                return_tensors='pt', )
            token_ids = encode_dict['input_ids'].to(self.device)
            attention_masks = encode_dict['attention_mask'].to(self.device)
            token_type_ids = encode_dict['token_type_ids'].to(self.device)
            span1_ids = torch.tensor([span1_ids]).to(self.device)
            span2_ids = torch.tensor([span2_ids]).to(self.device)
            output = self.model(token_ids,
                                attention_masks,
                                token_type_ids,
                                span1_ids,
                                span2_ids)
            logits = np.argmax(output.cpu().detach().numpy().tolist(), -1)
            logger.info('result:' + str(logits[0]))


if __name__ == '__main__':

    from ReadAndWrite import RAW

    raw = RAW()
    # path_time = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
    # log_dir_path = os.path.join('./log_dir/', path_time+'-lstm_cn-test')
    #
    # if raw.IfFolderExists(log_dir_path):
    #     pass
    # else:
    #     raw.CreateFolder(log_dir_path)
    #
    # SumWriter = SummaryWriter(logdir=log_dir_path)
    processor = CRProcessor()
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
    # ============================================================================================================================================================
    if 'Inception' in args.ensemble_model_list:
        from inception import Net

        ###269632 chinese  50512 english
        ###
        if args.en_cn == 'en':
            fx_in = 24572
            model = Net(args=args, fx_in=fx_in)
        else:
            fx_in = 201960 #98304
            model = Net(args=args, fx_in=fx_in)

        if args.model_type == '1d':
            from inception_1d import InceptionModel

            if args.en_cn == 'en':
                model = InceptionModel(input_shape=(args.train_batch_size, 13, 1024 * 3), args=args)  # english
            else:
                model = InceptionModel(input_shape=(args.train_batch_size, 25, 1024 * 3), args=args) # chinese


        ###269632 chinese  50512english
        # model = InceptionModel(input_shape=(args.train_batch_size, 25, 1024 * 3), args=args)

        experiment = '-inception-'+args.exp_name
        path_time = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
        log_dir_path = os.path.join('./log_dir/',  experiment)

        if raw.IfFolderExists(log_dir_path):
            pass
        else:
            raw.CreateFolder(log_dir_path)

        SumWriter = SummaryWriter(logdir=log_dir_path)
        ckpt_path = './checkpoints/2023-05-05 03-11-41-inception-5-5-new-note-300/best.pt'
        model.load_state_dict(torch.load(ckpt_path))
        bertForCR = BertForCR(model, args)

        total_loss, accuracy, precision, recall, f1, tp, fp, fn, tn, trues, preds = bertForCR.dev(dev_loader)


    # ============================================================================================================================================================
    if 'vgg16' in args.ensemble_model_list:
        from vgg16 import VGG16
        if args.model_type == '1d':
            from vgg16_1d import VGG16

            model = VGG16(Shape=(32, 1, 1024), args=args)
        else:
            # english
            # 24572
            # chinese
            # 98304
            if args.en_cn == 'en':
                fx_in = 24572
            else:
                fx_in = 98304

            model = VGG16(num_classes=2, fx_in=fx_in, args=args)

        experiment = '-vgg16-'+args.exp_name
        path_time = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
        log_dir_path = os.path.join('./log_dir/',  experiment)

        if raw.IfFolderExists(log_dir_path):
            pass
        else:
            raw.CreateFolder(log_dir_path)

        SumWriter = SummaryWriter(logdir=log_dir_path)

        ckpt_path = './checkpoints/2023-05-05 18-58-33-vgg16-5-5-new-note-300/best.pt'
        model.load_state_dict(torch.load(ckpt_path))

        bertForCR = BertForCR(model, args)
        total_loss, accuracy, precision, recall, f1, tp, fp, fn, tn, trues, preds = bertForCR.dev(dev_loader)

    # # ============================================================================================================================================================
    if 'CRModel_2dense' in args.ensemble_model_list:
        import CRModel_2dense
        model = CRModel_2dense.CorefernceResolutionModel(args)
        experiment = '-CRModel_2dense-'+args.exp_name
        path_time = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
        log_dir_path = os.path.join('./log_dir/',  experiment)

        if raw.IfFolderExists(log_dir_path):
            pass
        else:
            raw.CreateFolder(log_dir_path)

        SumWriter = SummaryWriter(logdir=log_dir_path)

        bertForCR = BertForCR(model, args)

        total_loss, accuracy, precision, recall, f1, tp, fp, fn, tn, trues, preds = bertForCR.dev(dev_loader)

    # ============================================================================================================================================================
    if 'TextCNN' in args.ensemble_model_list:
        in_channels, output_size, kernel_sizes, num_filters = 1, 2, [3, 4, 5], 100
        model = TextCNN(in_channels, output_size, kernel_sizes, num_filters, args)
        experiment = 'TextCNN-'+args.exp_name
        path_time = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
        log_dir_path = os.path.join('./log_dir/',  experiment)

        if raw.IfFolderExists(log_dir_path):
            pass
        else:
            raw.CreateFolder(log_dir_path)

        SumWriter = SummaryWriter(logdir=log_dir_path)

        ckpt_path = './checkpoints/2023-04-17 21-39-34-inception/best.pt'
        model.load_state_dict(torch.load(ckpt_path))
        bertForCR = BertForCR(model, args)

        total_loss, accuracy, precision, recall, f1, tp, fp, fn, tn, trues, preds = bertForCR.dev(dev_loader)


    # ============================================================================================================================================================
    if 'LeNet5' in args.ensemble_model_list:
        from lenet5 import LeNet5

         # english 9184, chinese 49024
        if args.en_cn == 'en':
            fx_in = 9184
            model = LeNet5(fx_in=fx_in, args=args)
        else:
            fx_in = 49024
            model = LeNet5(fx_in=fx_in, args=args)

        if args.model_type == '1d':
            from lenet5_1d import Lenet5
            if args.en_cn == 'en':
                model = Lenet5(Shape=(args.train_batch_size, 13, 1024), args=args)
            else:
                model = Lenet5(Shape=(args.train_batch_size, 25, 1024), args=args)
            # model = Lenet5(Shape=(args.train_batch_size, 25, 1024), args=args) # chinese
            # model = Lenet5(Shape=(args.train_batch_size, 13, 1024), args=args)  # english


        # model = Lenet5(Shape=(args.train_batch_size, 25, 1024), args=args)
        experiment = '-Lenet5-'+args.exp_name
        path_time = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
        log_dir_path = os.path.join('./log_dir/',  experiment)

        if raw.IfFolderExists(log_dir_path):
            pass
        else:
            raw.CreateFolder(log_dir_path)

        SumWriter = SummaryWriter(logdir=log_dir_path)

        bertForCR = BertForCR(model, args)
        ckpt_path = './checkpoints/2023-05-05 04-20-07-Lenet5-5-5-new-note-300/best.pt'
        model.load_state_dict(torch.load(ckpt_path))
        total_loss, accuracy, precision, recall, f1, tp, fp, fn, tn, trues, preds = bertForCR.dev(dev_loader)

    # ============================================================================================================================================================
    if 'LSTM' in args.ensemble_model_list:
        # from lstm import LstmModel

        from lstm import LSTM
        if args.en_cn == 'en':
            fx_in = 2304
            model = LSTM(input_size=fx_in, hidden_size=1024, layer_size=8, output_size=2,
                         args=args)  # chinese 3072, english 2304

        else:
            fx_in = 3072
            model = LSTM(input_size=fx_in, hidden_size=1024, layer_size=8, output_size=2,
                         args=args)  # chinese 3072, english 2304

        # model = LstmModel(input_size=1024 * 3, hidden_size=1024, num_layers=8, num_classes=2, args=args)

        if args.model_type == '1d':
            from lstm_1d import LstmModel
            if args.en_cn == 'en':
                model = LstmModel(input_size=2304, hidden_size=1024, num_layers=8, num_classes=2, args=args)
            else:
                model = LstmModel(input_size=3072, hidden_size=1024, num_layers=8, num_classes=2, args=args)

        experiment = '-LstmModel-'+args.exp_name
        path_time = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
        log_dir_path = os.path.join('./log_dir/',  experiment)

        if raw.IfFolderExists(log_dir_path):
            pass
        else:
            raw.CreateFolder(log_dir_path)

        SumWriter = SummaryWriter(logdir=log_dir_path)
        ckpt_path = './checkpoints/2023-05-05 06-19-02-LstmModel-5-5-new-note-300/best.pt'
        model.load_state_dict(torch.load(ckpt_path))
        bertForCR = BertForCR(model, args)

        total_loss, accuracy, precision, recall, f1, tp, fp, fn, tn, trues, preds = bertForCR.dev(dev_loader)

    # ============================================================================================================================================================
    if 'CRModel' in args.ensemble_model_list:
        model = CRModel.CorefernceResolutionModel(args)
        experiment = '-CRModel-'+args.exp_name
        path_time = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
        # log_dir_path = os.path.join('./log_dir/',  experiment)
        log_dir_path = os.path.join('./log_dir/',  experiment)

        if raw.IfFolderExists(log_dir_path):
            pass
        else:
            raw.CreateFolder(log_dir_path)

        SumWriter = SummaryWriter(logdir=log_dir_path)

        bertForCR = BertForCR(model, args)
        total_loss, accuracy, precision, recall, f1, tp, fp, fn, tn, trues, preds = bertForCR.dev(dev_loader)

