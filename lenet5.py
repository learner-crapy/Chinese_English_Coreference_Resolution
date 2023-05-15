import torch
from torch import nn
import torch.nn.functional as F
from bertBaseModel import BaseModel
class LeNet5(BaseModel):
    def __init__(self, fx_in,
                 args,
                 **kwargs):
        super(LeNet5, self).__init__(args.bert_dir, dropout_prob=args.dropout_prob)
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(fx_in, 120) # english 9184, chinese 49024
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 2)
        self.relu5 = nn.ReLU()

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

        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        probas = F.softmax(y, dim=1)
        return y
