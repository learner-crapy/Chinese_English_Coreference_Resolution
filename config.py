import argparse

class Args:
  @staticmethod
  def parse():
    parser = argparse.ArgumentParser()
    return parser

  @staticmethod
  def initialize(parser):
    # args for path
    parser.add_argument('--output_dir', default='./checkpoints/',
                        help='the output dir for model checkpoints')

    parser.add_argument('--bert_dir', default='./pre_model/chinese/chinese_roberta_wwm_large_ext_pytorch/',
                        help='bert dir for uer')
    parser.add_argument('--data_dir', default='./data/chinese/',
                        help='data dir for uer')
    parser.add_argument('--log_dir', default='./logs/',
                        help='log dir for uer')

    # other args
    parser.add_argument('--num_tags', default=2, type=int,
                        help='number of tags')
    parser.add_argument('--seed', type=int, default=123, help='random seed')

    parser.add_argument('--gpu_ids', type=str, default='0',
                        help='gpu ids to use, -1 for cpu, "0,1" for multi gpu')

    parser.add_argument('--max_seq_len', default=200, type=int)

    parser.add_argument('--eval_batch_size', default=16, type=int)

    parser.add_argument('--swa_start', default=3, type=int,
                        help='the epoch when swa start')

    # train args
    parser.add_argument('--train_epochs', default=200, type=int,
                        help='Max training epoch')

    parser.add_argument('--dropout_prob', default=0.1, type=float,
                        help='drop out probability')

    parser.add_argument('--lr', default=2e-8, type=float,
                        help='learning rate for the bert module')

    parser.add_argument('--other_lr', default=2e-8, type=float,
                        help='learning rate for the module except bert')

    parser.add_argument('--max_grad_norm', default=1.0, type=float,
                        help='max grad clip')

    parser.add_argument('--warmup_proportion', default=0.1, type=float)

    parser.add_argument('--weight_decay', default=0.01, type=float)

    parser.add_argument('--adam_epsilon', default=1e-5, type=float)

    parser.add_argument('--train_batch_size', default=16, type=int)

    parser.add_argument('--eval_model', default=True, action='store_true',
                        help='whether to eval model after training')
    # select which language you want to apply
    parser.add_argument('--en_cn', default='cn', type=str, choices=['cn', 'en'], help='en or cn')

    # set a name of your experiment, the model will be saved in checkpoints/model_name+exp_name, and the log will be saved in log_dir/model_name+exp_name
    # also the the name of confusion matrix will be related to exp_name
    parser.add_argument('--exp_name', default='exp', type=str, help='exp name')

    # select 1d model or 2d model
    parser.add_argument('--model_type', default='2d', type=str, choices=['1d', '2d'], help='1d or 2d')

    # how many models you want to ensemble, set a list of model names
    parser.add_argument('--ensemble_model_list', default=['CRModel', 'CRModel_2dense', 'vgg16', 'Inception', 'LeNet5', 'LSTM', 'TextCNN', 'SVM', 'decisoin_tree', 'k-means'],
                        nargs='+', type=str,
                        # choices=['CRModel', 'CRModel_2dense', 'vgg16', 'Inception', 'LeNet5', 'LSTM', 'TextCNN', 'SVM', 'decisoin_tree', 'k-means'],
                        help='ensemble model names')



    return parser

  def get_parser(self):
    parser = self.parse()
    parser = self.initialize(parser)
    return parser.parse_args()
