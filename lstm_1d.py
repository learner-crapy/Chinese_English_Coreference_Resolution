import torch
import torch.nn as nn
from bertBaseModel import BaseModel

class LstmModel(BaseModel):
    def __init__(self, input_size, hidden_size, num_layers, num_classes,
                 args,
                 **kwargs):
        super(LstmModel, self).__init__(args.bert_dir, dropout_prob=args.dropout_prob)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(32, num_classes)
        self.sigmoid = nn.Sigmoid()

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
            seq_out = bert_outputs[1]  # [batch, dim]
            logit = []

            for t_out, s_out, s1_mask, s2_mask in zip(token_out, seq_out, span1_mask, span2_mask):
                s1_mask = s1_mask == 1
                s2_mask = s2_mask == 1
                span1_out = t_out[s1_mask]
                span2_out = t_out[s2_mask]
                out = torch.cat(
                    [s_out.unsqueeze(0), torch.mean(span1_out, 0).unsqueeze(0), torch.mean(span2_out, 0).unsqueeze(0)],
                    dim=1)
                # out = torch.cat([s_out.unsqueeze(0), span1_out, span2_out], dim=0).unsqueeze(0)
                # 这里可以使用最大池化或者平均池化，使用平均池化的时候要注意，
                # 要除以每一个句子本身的长度
                # out = torch.sum(out, 1)
                out = torch.where(torch.isnan(out), torch.full_like(out, 0), out)
                logit.append(out)
            logits.append(logit)
        for i in range(len(logits)):
            logits[i] = torch.stack([torch.tensor(x).clone() for x in logits[i]], dim=0)
        x = torch.nn.functional.normalize(torch.stack([torch.tensor(x).clone() for x in logits], dim=0).squeeze().transpose(0, 1), p=2, dim=1)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc1(out[:, -1, :])
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

