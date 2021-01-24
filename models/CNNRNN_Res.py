import torch
import torch.nn as nn
from torch.nn import Parameter

class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        self.ratio = args.ratio #The ratio between CNNRNN and residual
        self.use_cuda = args.cuda #use gpu or not
        self.P = args.window #window size
        self.m = data.m #number of features (10 for hhs regions)
        self.hidR = args.hidRNN #number of RNN hidden units

        #nn.GRU(input_size – The number of expected features in the input x, hidden_size – The number of features in the hidden state h)
        self.GRU1 = nn.GRU(self.m, self.hidR)

        self.residual_window = args.residual_window #The window size of the residual component

        self.mask_mat = Parameter(torch.Tensor(self.m, self.m)) #Parameter(Tensor(10x10))

        #data.adj = torch.Tensor(np.loadtxt(args.sim_mat, delimiter=','))
        self.adj = data.adj

        #https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html
        #dropout = dropout applied to layers (0 = no dropout)
        self.dropout = nn.Dropout(p=args.dropout)

        #https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        #number of RNN hidden units, number of features (10 for hhs regions)
        #Applies a linear transformation to the incoming data: y = xA^T + b
        self.linear1 = nn.Linear(self.hidR, self.m)

        if (self.residual_window > 0):
            self.residual_window = min(self.residual_window, self.P)
            self.residual = nn.Linear(self.residual_window, 1);
        self.output = None

        #the output function of neural net
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid
        if (args.output_fun == 'tanh'):
            self.output = F.tanh

    def forward(self, x):
        # x: batch x window (self.P) x #signal (m)
        # first transform
        masked_adj = self.adj * self.mask_mat
        x = x.matmul(masked_adj)
        # RNN
        # r: window (self.P) x batch x #signal (m)
        r = x.permute(1, 0, 2).contiguous()
        _, r = self.GRU1(r)
        r = self.dropout(torch.squeeze(r, 0))

        res = self.linear1(r)

        #residual
        if (self.residual_window > 0):
            z = x[:, -self.residual_window:, :];
            z = z.permute(0,2,1).contiguous().view(-1, self.residual_window);
            z = self.residual(z);
            z = z.view(-1,self.m);
            res = res * self.ratio + z;

        if self.output is not None:
            res = self.output(res).float()

        return res
