import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F

class LSTM_NET(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_layers, dropout=0.5):
        super(LSTM_NET, self).__init__()
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.classifier = nn.Sequential(nn.Dropout(dropout),
                                        nn.Linear(hidden_dim, 2))
    def forward(self, inputs):
        inputs = torch.transpose(inputs, 1, 2)
        x, _ = self.lstm(inputs, None)
        # take the last sequencial features
        x = x[:, -1, :]
        x = self.classifier(x)
        
        return F.log_softmax(x, dim=1)

lstm_model = LSTM_NET(embed_dim=4, hidden_dim=64, num_layers=1)

lst_training_loss_lstm = []
lst_val_loss_lstm = []
lst_accuracy_lstm = []

def lstm_train(eq):
    lstm_model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    total_loss = .0
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = lstm_model(inputs)
        #outputs = outputs.squeeze()
        loss = F.nll_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        #steps += seq_length
        if i > 0 and i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * batch_size, len(train_loader.dataset),
                100. * i / len(train_loader), total_loss/args.log_interval, steps))
            if i * batch_size == 500:
                lst_training_loss_lstm.append(total_loss/args.log_interval)
            total_loss = 0

def lstm_test():  
    lstm_model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data = data.view(-1, input_channels, seq_length)
            if args.permute:
                data = data[:, :, permute]
            data, target = Variable(data, volatile=True), Variable(target)
            output = lstm_model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        lst_val_loss_lstm.append(test_loss)
        lst_accuracy_lstm.append((100. * correct / len(test_loader.dataset)).item())
        return test_loss

for epoch in range(1, epochs+1):
    lstm_train(epoch)
    lstm_test()
    if epoch % 10 == 0:
        lr /= 10
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
