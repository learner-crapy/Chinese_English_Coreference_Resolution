import torch
from torch import nn
from bertBaseModel import BaseModel


class LSTM(BaseModel):
    def __init__(self, input_size, hidden_size, layer_size, output_size,
                 args,
                 bidirectional=True,
                 **kwargs, ):
        super(LSTM, self).__init__(args.bert_dir, dropout_prob=args.dropout_prob)

        self.input_size, self.hidden_size, self.layer_size, self.output_size = input_size, hidden_size, layer_size, output_size
        self.bidirectional = bidirectional

        # Step1: the LSTM model
        self.lstm = nn.LSTM(input_size, hidden_size, layer_size, batch_first=True, bidirectional=bidirectional)

        # Step2: the FNN
        if bidirectional:  # we'll have 2 more layers
            self.layer = nn.Linear(hidden_size * 2, output_size)
        else:
            self.layer = nn.Linear(hidden_size, output_size)

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                span1_mask=None,
                span2_mask=None,
                prints=False):
        bert_outputs = self.bert_module(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        logits = []
        for bert_outputs_i in list(bert_outputs.hidden_states):
            token_out = bert_outputs_i  # [batch, max_seq_len, dim]
            token_out = bert_outputs[0]  # [batch, max_seq_len, dim]
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
                    [s_out.unsqueeze(0), torch.mean(span1_out, 0).unsqueeze(0), torch.mean(span2_out, 0).unsqueeze(0)],
                    dim=1)
                out = torch.where(torch.isnan(out), torch.full_like(out, 0), out)
                logit.append(out)
            logits.append(logit)
        for i in range(len(logits)):
            logits[i] = torch.stack([torch.tensor(x).clone() for x in logits[i]], dim=0)
        x = torch.nn.functional.normalize(
            torch.stack([torch.tensor(x).clone() for x in logits], dim=0).squeeze().transpose(0, 1), p=2, dim=1)
        # x = x.unsqueeze(1)
        images = x.squeeze(1)
        if prints: print('images shape:', images.shape)
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Set initial states
        if self.bidirectional:
            # Hidden state:
            hidden_state = torch.zeros(self.layer_size * 2, images.size(0), self.hidden_size).to(DEVICE)
            # Cell state:
            cell_state = torch.zeros(self.layer_size * 2, images.size(0), self.hidden_size).to(DEVICE)
        else:
            # Hidden state:
            hidden_state = torch.zeros(self.layer_size, images.size(0), self.hidden_size).to(DEVICE)
            # Cell state:
            cell_state = torch.zeros(self.layer_size, images.size(0), self.hidden_size).to(DEVICE)
        if prints: print('hidden_state t0 shape:', hidden_state.shape, '\n' +
                         'cell_state t0 shape:', cell_state.shape)

        # LSTM:

        output, (last_hidden_state, last_cell_state) = self.lstm(images, (hidden_state, cell_state))
        if prints: print('LSTM: output shape:', output.shape, '\n' +
                         'LSTM: last_hidden_state shape:', last_hidden_state.shape, '\n' +
                         'LSTM: last_cell_state shape:', last_cell_state.shape)
        # Reshape
        output = output[:, -1, :]
        if prints: print('output reshape:', output.shape)

        # FNN:
        output = self.layer(output)
        probas = nn.functional.softmax(output, dim=1)
        if prints: print('FNN: Final output shape:', output.shape)

        # return output, probas
        return output
