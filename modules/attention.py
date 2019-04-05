import torch
import torch.nn as nn

from torch.autograd import Variable


class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.dim = dim
        self.linear_in = nn.Linear(dim, dim)
        self.linear_in.weight.data.uniform_(-0.1, 0.1)
        self.linear_out = nn.Linear(dim*2, dim)
        self.linear_out.weight.data.uniform_(-0.1, 0.1)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()


    def forward(self, input, memory_bank):
        is_cuda = input.is_cuda
        [batch_size, t_len, _] = input.size()
        [batch_size, s_len, _] = memory_bank.size()
        
        h_t = self.linear_in(input.view(batch_size*t_len, self.dim)).view(batch_size, t_len, self.dim)
        h_s = memory_bank.transpose(1, 2)
        score = torch.bmm(h_t, h_s)

        align_vectors = self.softmax(score.view(batch_size*t_len, s_len))
        align_vectors = align_vectors.view(batch_size, t_len, s_len)

        context = torch.bmm(align_vectors, memory_bank)

        concat_c = torch.cat([context, input], 2).view(batch_size*t_len, self.dim*2)
        attn_h = self.linear_out(concat_c).view(batch_size, t_len, self.dim)

        attn_h = self.tanh(attn_h)

        return attn_h, align_vectors
