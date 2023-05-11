import torch
import torch.nn as nn
from bertBaseModel import BaseModel


class TextCNN(BaseModel):
    def __init__(self, in_channels, output_size, kernel_sizes, num_filters, args, **kwargs):
        # def __init__(self, in_channels, output_size, kernel_sizes=[3, 4, 5], num_filters=32):
        super(TextCNN, self).__init__(args.bert_dir, dropout_prob=args.dropout_prob)
        out_dims = self.bert_config.hidden_size

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, num_filters, k, ) for k in kernel_sizes
        ])
        self.fc = nn.Linear(num_filters * len(kernel_sizes), output_size)
        self.dropout = nn.Dropout(0.5)

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

        token_out = bert_outputs[0] # [batch, max_seq_len, dim]
        seq_out = bert_outputs[1]  # [batch, dim]
        logits = []
        for t_out, s_out, s1_mask, s2_mask in zip(token_out, seq_out, span1_mask, span2_mask):
            s1_mask = s1_mask == 1
            s2_mask = s2_mask == 1
            span1_out = t_out[s1_mask]
            span2_out = t_out[s2_mask]
            out = torch.cat(
                [s_out.unsqueeze(0), torch.mean(span1_out, 0).unsqueeze(0), torch.mean(span2_out, 0).unsqueeze(0)],
                dim=1)
            out = torch.where(torch.isnan(out), torch.full_like(out, 0), out)
            # out = torch.cat([s_out.unsqueeze(0), span1_out, span2_out], dim=0).unsqueeze(0)
            # # 这里可以使用最大池化或者平均池化，使用平均池化的时候要注意，
            # # 要除以每一个句子本身的长度
            # out = torch.sum(out, 1)
            logits.append(out)
        # logits = torch.cat(logits, dim=0)
        # logits = logits.unsqueeze(0).repeat(logits.shape[0], 1, 1)
        for i in range(len(logits)):
            logits[i] = torch.stack([torch.tensor(x).clone() for x in logits[i]], dim=0)
        logits = torch.nn.functional.normalize(torch.stack([torch.tensor(x).clone() for x in logits], dim=0), p=2, dim=1)
        logits = [nn.functional.relu(conv(logits)) for conv in self.convs]
        logits = [nn.functional.max_pool1d(i, i.size(2)).squeeze(2) for i in logits]
        logits = torch.cat(logits, 1)
        logits = self.dropout(logits)
        logits = self.fc(logits)
        return logits


    # def forward(self, x):
    #     x = [nn.functional.relu(conv(x)) for conv in self.convs]
    #     x = [nn.functional.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
    #     x = torch.cat(x, 1)
    #     x = self.dropout(x)
    #     logit = self.fc(x)
    #     return logit



if __name__ == "__main__":
    # Generate random input data
    # in_channels = 32
    # input_size = 144
    # x = torch.randn(in_channels, 32, input_size)
    #
    # # Initialize the model
    # output_size = 2
    # model = TextCNN(in_channels, output_size)
    # print(model)
    #
    # # Pass the input data through the model
    # logits = model(x)
    # # Print the output shape
    # print(logits.shape)  # should be (in_channels, output_size)
    class Args:
        bert_dir = './chinese_roberta_wwm_ext_pytorch/'
        dropout_prob = 0.3
    args = Args()
    in_channels, output_size, kernel_sizes, num_filters = 64, 2, [3, 4, 5], 100
    model = TextCNN(in_channels, output_size, kernel_sizes, num_filters, args)
    print(model)
    from transformers import BertTokenizerFast
    tokenizer = BertTokenizerFast.from_pretrained(args.bert_dir)
    sentence = '塑料椅子这边坐着很多候烧者，沙发那边只有五个候烧者，他们舒适地架着二郎腿，都是一副功成名就的>模样，塑料椅子这边的个个都是正襟危坐。'
    inputs = tokenizer.encode_plus(sentence,
                                   return_attention_mask=True,
                                   return_token_type_ids=True,
                                   return_tensors='pt')
    print(inputs)
    outputs = model(**inputs)