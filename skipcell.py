import torch
import numpy as np
from torch.nn.init import xavier_uniform_
from torch.nn.modules.rnn import RNNCellBase
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def skip_lstm_cell(input, hidden, w_ih, w_hh, w_uh,
                   b_ih=None, b_hh=None, b_uh=None,
                   activation=torch.tan):
    c_prev, h_prev, update_prob_prev, cum_update_prob_prev = hidden

    gates = F.linear(input, w_ih, b_ih) + F.linear(h_prev, w_hh, b_hh)

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = activation(cellgate)
    outgate = torch.sigmoid(outgate)

    new_c_tilde = (forgetgate * c_prev) + (ingate * cellgate)
    new_h_tilde = outgate * activation(new_c_tilde)

    new_update_prob_tilde = torch.sigmoid(F.linear(new_c_tilde, w_uh, b_uh))
    cum_update_prob = cum_update_prob_prev + torch.min(update_prob_prev, 1. - cum_update_prob_prev)
    update_gate = cum_update_prob.round()

    new_c = update_gate * new_c_tilde + (1. - update_gate) * c_prev
    new_h = update_gate * new_h_tilde + (1. - update_gate) * h_prev
    new_update_prob = update_gate * new_update_prob_tilde + (1. - update_gate) * update_prob_prev
    new_cum_update_prob = update_gate * 0. + (1. - update_gate) * cum_update_prob

    new_state = (new_c, new_h, new_update_prob, new_cum_update_prob)
    new_output = (new_h, update_gate)

    return new_output, new_state


class SkipLSTMCell(RNNCellBase):
    def __init__(self, input_size, hidden_size, bias=True,
                 batch_first=False, activation=torch.tanh):
        super(SkipLSTMCell, self).__init__(input_size, hidden_size, bias, num_chunks=4)
        self.cell = skip_lstm_cell
        self.activation = activation
        self.batch_first = batch_first
        self.weight_uh = Parameter(xavier_uniform_(torch.Tensor(1, hidden_size)))
        if bias:
            self.bias_uh = Parameter(torch.ones(1))
        else:
            self.register_parameter('bias_uh', None)

    def forward(self, input, hx=None):
        if len(input.shape) == 3:
            if self.batch_first:
                input = input.transpose(0, 1)
            sequence_length, batch_size, input_size = input.shape
        else:
            sequence_length = 1
            batch_size, input_size = input.shape

        if hx is None:
            hx = self.init_hidden(batch_size)
            if input.is_cuda:
                hx = tuple([x.cuda() for x in hx])

        # if len(input.shape) == 3:
        #     self.check_forward_input(input[0])
        #     self.check_forward_hidden(input[0], hx[0], '[0]')
        #     self.check_forward_hidden(input[0], hx[1], '[1]')
        # else:
        #     self.check_forward_input(input)
        #     self.check_forward_hidden(input, hx[0], '[0]')
        #     self.check_forward_hidden(input, hx[1], '[1]')

        lst_output = []
        lst_update_gate = []
        for t in np.arange(sequence_length):
            output, hx = self.cell(
                input[t], hx, self.weight_ih,
                self.weight_hh, self.weight_uh,
                self.bias_ih, self.bias_hh, self.bias_uh,
                activation=self.activation
            )
            new_h, update_gate = output
            lst_output.append(new_h)
            lst_update_gate.append(update_gate)
        output = torch.stack(lst_output)
        update_gate = torch.stack(lst_update_gate)
        if self.batch_first:
            output = output.transpose(0, 1)
            update_gate = update_gate.transpose(0, 1)
        return output, hx, update_gate

    def init_hidden(self, batch_size):
        return (torch.randn(batch_size, self.hidden_size),
                torch.randn(batch_size, self.hidden_size),
                torch.ones(batch_size, 1),
                torch.zeros(batch_size, 1))
