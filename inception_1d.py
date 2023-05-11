import torch
import torch.nn as nn
from bertBaseModel import BaseModel
# inception
class InceptionBlock(BaseModel):
    def __init__(self, in_channels,
                 args,
                 **kwargs):
        super(InceptionBlock, self).__init__(args.bert_dir, dropout_prob=args.dropout_prob)
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=1, padding='same')
        self.bn1 = nn.BatchNorm1d(64)
        self.conv3_1 = nn.Conv1d(in_channels, 64, kernel_size=1, padding='same')
        self.bn3_1 = nn.BatchNorm1d(64)
        self.conv3_2 = nn.Conv1d(64, 64, kernel_size=3, padding='same')
        self.bn3_2 = nn.BatchNorm1d(64)
        self.conv5_1 = nn.Conv1d(in_channels, 64, kernel_size=1, padding='same')
        self.bn5_1 = nn.BatchNorm1d(64)
        self.conv5_2 = nn.Conv1d(64, 64, kernel_size=5, padding='same')
        self.bn5_2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_max = nn.Conv1d(in_channels, 64, kernel_size=1, padding='same')
        self.bn_max = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv1 = self.bn1(conv1)
        conv1 = self.relu(conv1)

        conv3_1 = self.conv3_1(x)
        conv3_1 = self.bn3_1(conv3_1)
        conv3_1 = self.relu(conv3_1)
        conv3_2 = self.conv3_2(conv3_1)
        conv3_2 = self.bn3_2(conv3_2)
        conv3_2 = self.relu(conv3_2)

        conv5_1 = self.conv5_1(x)
        conv5_1 = self.bn5_1(conv5_1)
        conv5_1 = self.relu(conv5_1)
        conv5_2 = self.conv5_2(conv5_1)
        conv5_2 = self.bn5_2(conv5_2)
        conv5_2 = self.relu(conv5_2)

        pool = self.pool(x)
        conv_max = self.conv_max(pool)
        conv_max = self.bn_max(conv_max)
        conv_max = self.relu(conv_max)

        out = torch.cat([conv1, conv3_2, conv5_2, conv_max], dim=2)
        return out


class InceptionModel(BaseModel):
    def __init__(self, input_shape, args, **kwargs):
        super(InceptionModel, self).__init__(args.bert_dir, dropout_prob=args.dropout_prob)
        self.conv1 = nn.Conv1d(input_shape[1], 64, kernel_size=7, padding='same')
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, padding='same')
        self.bn2 = nn.BatchNorm1d(64)
        self.inception1 = InceptionBlock(64, args, **kwargs)
        self.inception2 = InceptionBlock(64, args, **kwargs)
        self.pool2 = nn.MaxPool1d(kernel_size=7, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, 2)
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
                # out = torch.cat([s_out.unsqueeze(0), span1_out, span2_out], dim=1).unsqueeze(0)
                # 这里可以使用最大池化或者平均池化，使用平均池化的时候要注意，
                # 要除以每一个句子本身的长度
                # out = torch.sum(out, 1)
                out = torch.where(torch.isnan(out), torch.full_like(out, 0), out)
                logit.append(out)
            logits.append(logit)
        for i in range(len(logits)):
            logits[i] = torch.stack([torch.tensor(x).clone() for x in logits[i]], dim=0)
        x = torch.nn.functional.normalize(torch.stack([torch.tensor(x).clone() for x in logits], dim=0).squeeze().transpose(0, 1), p=2, dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.pool2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


if __name__ == '__main__':
    # To generate test data for this model, we need to create a tensor of the same shape as the input shape of the model.
    # The input shape of the model is (batch_size, num_channels, sequence_length).
    # We can create a random tensor of this shape using torch.randn() function.

    test_input = torch.randn(32, 32, 144)  # replace batch_size, num_channels, sequence_length with appropriate values.model = InceptionModel(test_input.shape)
    model = InceptionModel(input_shape=test_input.shape)

    # Pass test_input through the model
    output = model(test_input)

    # Print the output
    print(output)