import torch
from torch import nn
import torch.nn.functional as F
from bertBaseModel import BaseModel

class Inception(BaseModel):
    def __init__(self, in_channels,
                 args,
                 **kwargs):  # 输入通道数定为未知量，当实例化模型时可以调用
        super(Inception, self).__init__(args.bert_dir, dropout_prob=args.dropout_prob)
        self.branch11 = nn.Conv2d(in_channels, 16, kernel_size=1)  # 1*1的卷积分支 C1=16 图片形状（h,w）未发生变化

        self.branch55_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch55_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)  # 5*5的卷积分支 C2=24 图片形状不发生变化

        self.branch33_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch33_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch33_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)  # 3*3的卷积分支 C3=24 图片形状不发生变化

        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)  # 1*1的池化分支 C4=24 图片形状不发生变化

    def forward(self, x):
        branch11 = self.branch11(x)

        branch55 = self.branch55_1(x)
        branch55 = self.branch55_2(branch55)

        branch33 = self.branch33_1(x)
        branch33 = self.branch33_2(branch33)
        branch33 = self.branch33_3(branch33)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch11, branch33, branch55, branch_pool]

        return torch.cat(outputs, dim=1)  # dim=1,在(B，C，H，W)的第二个维度C上合并  其他维度的值必须相同


class Net(BaseModel):
    def __init__(self, fx_in, args, **kwargs):
        super(Net, self).__init__(args.bert_dir, dropout_prob=args.dropout_prob)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(88, 20, kernel_size=3)  # 将Inception沿着C维度拼接之后，通道总数未=为88

        self.incep1 = Inception(in_channels=10, args=args)
        self.incep2 = Inception(in_channels=20, args=args)

        self.mp = nn.MaxPool2d(kernel_size=2)
        self.fc = nn.Linear(fx_in, 2)  ###269632 chinese  50512english

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                span1_mask=None,
                span2_mask=None):
        bert_outputs = self.bert_module(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )


        logits = []
        for bert_outputs_i in list(bert_outputs.hidden_states):
            token_out = bert_outputs_i # [batch, max_seq_len, dim]
            # token_out = bert_outputs[0]  # [batch, max_seq_len, dim]
            seq_out = bert_outputs[1]  # [batch, dim]
            logit = []

            for t_out, s_out, s1_mask, s2_mask in zip(token_out, seq_out, span1_mask, span2_mask):
                s1_mask = s1_mask == 1
                s2_mask = s2_mask == 1
                span1_out = t_out[s1_mask]
                span2_out = t_out[s2_mask]
                # out = torch.cat([s_out.unsqueeze(0), span1_out, span2_out], dim=0).unsqueeze(0)
                # 这里可以使用最大池化或者平均池化，使用平均池化的时候要注意，
                # 要除以每一个句子本身的长度
                # out = torch.sum(out, 1)
                #
                out = torch.cat(
                    [s_out.unsqueeze(0),torch.mean(span1_out, 0).unsqueeze(0), torch.mean(span2_out, 0).unsqueeze(0)],
                    dim=1)
                out = torch.where(torch.isnan(out), torch.full_like(out, 0), out)
                logit.append(out)
            logits.append(logit)
        for i in range(len(logits)):
            logits[i] = torch.stack([torch.tensor(x).clone() for x in logits[i]], dim=0)
        x = torch.nn.functional.normalize(torch.stack([torch.tensor(x).clone() for x in logits], dim=0).squeeze().transpose(0, 1), p=2, dim=1)
        x = x.unsqueeze(1)
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = self.incep1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.incep2(x)
        x = x.view(in_size, -1)  ###
        x = self.fc(x)  ### 随机产生一个x 去掉这三行 输出x.size()
        probas = F.log_softmax(x, dim=1)  ###

        # return x, probas  ###
        return x