import torch
import torch.nn as nn
from skipcell import SkipLSTMCell
import numpy as np


class SelfAttn(nn.Module):
    def __init__(self, feature_size, hidden_size, num_desc):
        super(SelfAttn, self).__init__()
        self.linear_1 = nn.Linear(feature_size, hidden_size, bias=False)
        self.linear_2 = nn.Linear(hidden_size, num_desc, bias=False)
        self.num_desc = num_desc
        self.bn = nn.BatchNorm1d(feature_size)

    def forward(self, model_input):
        reshaped_input = model_input
        s1 = torch.tanh(self.linear_1(reshaped_input))
        A = torch.softmax(self.linear_2(s1), dim=1)
        M = self.bn(torch.bmm(model_input.permute(0, 2, 1), A)).permute(0, 2, 1)
        AAT = torch.bmm(A.permute(0, 2, 1), A)
        I = torch.eye(self.num_desc)
        if torch.cuda.is_available():
            I = I.cuda()
        P = torch.norm(AAT - I, 2)
        penal = P * P / model_input.shape[0]
        return M, penal


class ConvLSTM(nn.Module):
    def __init__(self, hidden_size, kernel,
                 stride, nb_filter, input_size):
        super(ConvLSTM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, nb_filter, kernel, stride),
            nn.ReLU(),
            nn.BatchNorm1d(nb_filter)
        )
        self.lstm = SkipLSTMCell(input_size=nb_filter, hidden_size=hidden_size, batch_first=True)
        self.hidden_size = hidden_size

    def forward(self, input):
        input = self.conv(input.permute(0, 2, 1))
        input = input.permute(0, 2, 1)
        output = self.lstm(input)
        output, hx, updated_state = output[0], output[1], output[2]
        return output[:, -1, :]

class Scoring(nn.Module):
    def __init__(self, feature_size):
        super(Scoring, self).__init__()

        conv_input = 128
        self.conv = nn.Sequential(
            nn.Conv1d(feature_size, conv_input, 1),
            nn.ReLU(),
            nn.BatchNorm1d(conv_input)
        )
        hidden_size = 256
        self.scale1 = ConvLSTM(hidden_size, 2, 1, 256, conv_input)
        self.scale2 = ConvLSTM(hidden_size, 4, 2, 256, conv_input)
        self.scale3 = ConvLSTM(hidden_size, 8, 4, 256, conv_input)
        self.attn = SelfAttn(conv_input, 64, 50)
        self.lstm = nn.LSTM(input_size=conv_input, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.linear_skip1 = nn.Linear(hidden_size, 64)
        self.linear_skip2 = nn.Linear(hidden_size, 64)
        self.linear_skip3 = nn.Linear(hidden_size, 64)
        self.linear_attn = nn.Linear(hidden_size, 64)
        self.linear_merge = nn.Linear(64 * 4, 64)
        self.cls = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(0.7)

    def forward(self, model_input):
        model_input = model_input.permute(0, 2, 1)
        model_input = self.conv.forward(model_input)
        model_input = model_input.permute(0, 2, 1)
        attn, penal = self.attn.forward(model_input)
        attn, _ = self.lstm(attn)
        attn = attn[:, -1, :]
        M_output = torch.cat(
            [
                self.relu(self.linear_skip1(self.scale1(model_input))),
                self.relu(self.linear_skip2(self.scale2(model_input))),
                self.relu(self.linear_skip3(self.scale3(model_input)))
            ],
            1
        )
        output = torch.cat([M_output, self.relu(self.linear_attn(attn))], 1)
        output = self.relu(self.linear_merge(output))
        return self.dropout(self.cls(output)), penal

    def loss(self, regression, actuals):
        regr_loss_fn = nn.MSELoss()
        return regr_loss_fn(regression, actuals)
