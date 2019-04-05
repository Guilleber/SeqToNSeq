import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class Variational(nn.Module):
    def __init__(self, dim):
        super(Variational, self).__init__()
        self.dim = dim
        self.input_to_mu = nn.Linear(dim, dim)
        #self.input_to_mu2 = nn.Linear(dim, dim)
        #self.input_to_mu3 = nn.Linear(dim, dim)
        self.input_to_logvar = nn.Linear(dim, dim)
        #self.input_to_logvar2 = nn.Linear(dim, dim)
        #self.input_to_logvar3 = nn.Linear(dim, dim)


    def forward(self, input, var=True):
        is_cuda = input.is_cuda
        [batch_size, _] = input.size()

        #pdr = 0.2
        #mu = self.input_to_mu3(F.relu(F.dropout(self.input_to_mu2(F.relu(F.dropout(self.input_to_mu1(input), p=pdr, training=self.training))), p=pdr, training=self.training)))
        #logvar = self.input_to_logvar3(F.relu(F.dropout(self.input_to_logvar2(F.relu(F.dropout(self.input_to_logvar1(input), p=pdr, training=self.training))), p=pdr, training=self.training)))
        mu = self.input_to_mu(input)
        logvar = self.input_to_logvar(input)
        std = torch.exp(0.5*logvar)

        #print(mu[0,0].data.cpu().numpy())
        #print(std[0,0].data.cpu().numpy())

        if var:
            z = Variable(torch.randn([batch_size, self.dim]))
            if is_cuda:
                z = z.cuda()

            z = std*z + mu
        else:
            z = mu

        kld = (-0.5 * torch.sum(logvar - torch.pow(mu, 2) - torch.exp(logvar) + 1, 1)).mean()

        return z, kld
