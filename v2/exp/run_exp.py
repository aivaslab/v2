import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim

from v2.util import conf
from sklearn.metrics import confusion_matrix

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = 'cpu'


class Learner:
    def __init__(self, net, train_loader, test_loader):
        self.net = net
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [200, 500])
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.best_acc = 0
        self.ckpt_dir = os.path.join(conf.CKPT_DIR, 'tmp.pth')

    def train(self, epoch):
        """ Training

        Args:
            epoch:
            criterion:
            trainloader:
            net:
            optimizer:

        Returns:

        """
        print('\nEpoch: %d' % epoch)
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        self.scheduler.step(epoch)
        print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        return train_loss / (batch_idx + 1), correct / total

    def test(self, epoch):
        # , criterion, testloader, net, hyper_model
        """ Testing

        Args:
            epoch:
            criterion:
            testloader:
            net:
            hyper_model:

        Returns:

        """
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)

                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        # Save checkpoint.
        acc = 100. * correct / total
        if acc > self.best_acc:
            print('Saving to %s...' % self.ckpt_dir)
            state = {
                'net': self.net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            self.best_acc = acc
            torch.save(state, self.ckpt_dir)
        return test_loss / (batch_idx + 1), correct / total

    def resume(self):
        print('==> Resuming from checkpoint..')
        ckpt = torch.load(self.ckpt_dir)
        self.net.load_state_dict(ckpt['net'])
        self.best_acc = ckpt['acc']
        return ckpt

    def final_test(self):
        print('== Final Test ==')
        ckpt = self.resume()
        epoch = ckpt['epoch']
        self.test(epoch)


def save_confusion_matrix(net, dataloader, labels, hyper_rst):
    net.eval()
    y_true = []  # ground truth
    y_pred = []  # prediction
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = net(inputs)
            _, predicted = outputs.max(1)

            y_true.append(targets.cpu().numpy())
            y_pred.append(predicted.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    print(labels)
    # labels = [l[:6] for l in labels]
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    print(cm)
    pd_cm = pd.DataFrame(cm, index=labels, columns=labels)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    # print(pd_cm)

    if hyper_rst['save']:
        with open(hyper_rst['cm_dir'], 'w+') as f:
            pd_cm.to_csv(f)


def save_results(cam, arc, pre, acc_te, ckpt_dir, rst_dir):
    f = open(rst_dir, 'a+')
    line = '%s\t%s\t%s\t%f\t%s\n' % (cam, arc, pre, acc_te, ckpt_dir)
    f.write(line)
    f.close()
