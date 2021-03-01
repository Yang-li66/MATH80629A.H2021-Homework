import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

# because Conv1d has two-sided padding, so in the case of TCN, we should cut the right side of the padding
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()      # Cut the right side of the padding

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # Create a weight norm layer
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # Apply the right padding cut
        self.chomp1 = Chomp1d(padding)
        # Apply ReLU as activation
        self.relu1 = nn.ReLU()
        # Apply dropout
        self.dropout1 = nn.Dropout(dropout)
        # 
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

import torch.nn.functional as F

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        o = self.linear(y1[:, :, -1])
        return F.log_softmax(o, dim=1)

from torch.autograd import Variable
import torch.optim as optim
import sys
import numpy as np
import easydict

args = easydict.EasyDict({
        "batch_size": 1,
        "cuda": False,
        "dropout": 0.02,
        "clip": -1,
        "epochs": 50,
        "ksize": 2,
        "levels": 5,
        "log_interval": 100,
        "lr": 1e-2,
        "optim": 'Adam',
        "nhid": 100,
        "seed": 1111,
        "permute": False
})

torch.manual_seed(args.seed)

batch_size = args.batch_size
n_classes = 2
input_channels = 4
seq_length = 50
epochs = args.epochs
steps = 0

print(args)
train_loader = DataLoader(TslaDataset(np_train, tgt_train),batch_size)
test_loader = DataLoader(TslaDataset(np_val, tgt_val),batch_size)

permute = torch.Tensor(np.random.permutation(784).astype(np.float64)).long()
channel_sizes = [args.nhid] * args.levels
kernel_size = args.ksize
model = TCN(input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=args.dropout)

lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)

lst_training_loss = []
lst_val_loss = []
lst_accuracy = []

def train(ep):
    global steps
    train_loss = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, input_channels, seq_length)
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        train_loss += loss
        steps += seq_length
        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSteps: {}'.format(
                ep, batch_idx * batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), train_loss.item()/args.log_interval, steps))
            if batch_idx * batch_size == 500:
                lst_training_loss.append(train_loss.item()/args.log_interval)
            
            train_loss = 0


def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(-1, input_channels, seq_length)
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        lst_val_loss.append(test_loss)
        lst_accuracy.append((100. * correct / len(test_loader.dataset)).item())
        return test_loss

for epoch in range(1, epochs+1):
        train(epoch)
        test()
        if epoch % 10 == 0:
            lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
