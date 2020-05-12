import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

import logging
import argparse
import os
import pandas as pd
import datetime

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

parser = argparse.ArgumentParser(description='speech_recognition')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--batch_size', default=512, type=int, help='batch size')
parser.add_argument('--context_size', default=12, type=int, help='context size')
parser.add_argument('--input_size', default=1000, type=int, help='input size')
parser.add_argument('--output_size', default=138, type=int, help='output size')
parser.add_argument('--num_epochs', default=18, type=int, help='epoch number')
parser.add_argument('--decay_steps', default='7, 12', type=str,
                    help='The step where learning rate decay by 0.1')
parser.add_argument('--save_step', default=5, type=int, help='step for saving model')
parser.add_argument('--eval_step', default=1, type=int, help='step for validation')
parser.add_argument('--train_data_path', default='../data/train.npy', type=str)
parser.add_argument('--train_label_path', default='../data/train_labels.npy', type=str)
parser.add_argument('--val_data_path', default='../data/dev.npy', type=str)
parser.add_argument('--val_label_path', default='../data/dev_labels.npy', type=str)
parser.add_argument('--test_data_path', default='../data/test.npy', type=str)
parser.add_argument('--checkpoint_dir', default='../checkpoints/', help='checkpoint folder root')
parser.add_argument('--result_file_name', default='hw1p2_test_result.csv', type=str, help='testing result save path')

args = parser.parse_args()

args.expr_dir = os.path.join(args.checkpoint_dir, current_time)
os.makedirs(args.expr_dir)

# Create the log
log_path = os.path.join(args.expr_dir, 'speech_recognition_{}.log'.format(current_time))
logging.basicConfig(filename=log_path, level=logging.INFO)

# Modify the result save path
args.result_file_name = os.path.join(args.expr_dir, args.result_file_name)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


def save_log(message):
    print(message)
    logging.info(message)


class load_dataset(Dataset):
    def __init__(self, data_path, label_path=None):
    	# Both data and label has the same time length for one utterrance
    	# Data shape: (utterance, seq_len, 40), Label shape: (utterance, seq_len)
        self.data = np.load(data_path, encoding='bytes', allow_pickle=True)
        if label_path:
            self.label = np.load(label_path, allow_pickle=True)
        else:
            self.label = None

        self.idx_map = []
        for i, xs in enumerate(self.data):
            for j in range(xs.shape[0]):
                self.idx_map.append((i, j))

    def __getitem__(self, index):
        i, j = self.idx_map[index]
        # Select the context_size before and after the current frame
        x = self.data[i].take(range(j - args.context_size, j + args.context_size + 1), mode='clip', axis=0).flatten()
        # Normalize
        # x = (x - x.mean()) / x.std()
        # Select the phoneme state label for the current frame
        y = np.int32(self.label[i][j]).reshape(1) if self.label is not None else np.int32(-1).reshape(1)
        return torch.from_numpy(x).float(), torch.LongTensor(y)

    def __len__(self):
        return len(self.idx_map)


###
# * Layers -> [input_size, 2048, 2048, 1024, 1024, output_size]
# * ReLU activations
# * Context size k = 12 frames on both sides
# * Adam optimizer, with the default learning rate 1e-3
# * Zero padding of k frames on both sides of each utterance
###
class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_size, 2048),
                                 nn.ReLU(inplace=True),
                                 nn.BatchNorm1d(2048),
                                 nn.Linear(2048, 2048),
                                 nn.ReLU(inplace=True),
                                 nn.BatchNorm1d(2048),
                                 nn.Linear(2048, 2048),
                                 nn.ReLU(inplace=True),
                                 nn.BatchNorm1d(2048),
                                 nn.Linear(2048, 1024),
                                 nn.ReLU(inplace=True),
                                 nn.BatchNorm1d(1024),
                                 nn.Linear(1024, 1024),
                                 nn.ReLU(inplace=True),
                                 nn.BatchNorm1d(1024),
                                 nn.Linear(1024, 1024),
                                 nn.ReLU(inplace=True),
                                 nn.BatchNorm1d(1024),
                                 nn.Linear(1024, 1024),
                                 nn.ReLU(inplace=True),
                                 nn.BatchNorm1d(1024),
                                 nn.Linear(1024, 512),
                                 nn.ReLU(inplace=True),
                                 nn.BatchNorm1d(512),
                                 nn.Linear(512, 512),
                                 nn.ReLU(inplace=True),
                                 nn.BatchNorm1d(512),
                                 nn.Linear(512, 512),
                                 nn.ReLU(inplace=True),
                                 nn.BatchNorm1d(512),
                                 nn.Linear(512, output_size)
                                 )

    def forward(self, x):
        return self.net(x)


def train(net, loader, optimizer, criterion, epoch):
    net.train()

    running_batch = 0
    running_loss = 0.0
    running_corrects = 0

    # Iterate over images.
    for i, (data, label) in enumerate(loader):
        data = data.to(device)
        label = label.to(device)
        output = net(data)
        _, label_pred = torch.max(output, 1)
        loss = criterion(output, label.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_batch += label.size(0)
        running_loss += loss.item()
        running_corrects += torch.sum(label_pred == label.view(-1)).item()

        if (i + 1) % 20 == 0:  # print every 5 mini-batches
            message = '[%d, %5d] loss: %.3f accuracy: %.3f' % (
            epoch, i + 1, running_loss / running_batch, running_corrects / running_batch)
            save_log(message)


def validate(net, loader, criterion, epoch):
    net.eval()

    running_batch = 0
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        message = '*' * 40
        save_log(message)
        for i, (data, label) in enumerate(loader):
            data = data.to(device)
            label = label.to(device)
            output = net(data)

            # label_pred = torch.nn.functional.softmax(output, dim=1)
            _, label_pred = torch.max(output, 1)

            loss = criterion(output, label.view(-1))
            running_batch += label.size(0)
            running_loss += loss.item()
            running_corrects += torch.sum(label_pred == label.view(-1)).item()

        running_loss /= running_batch
        acc = running_corrects / running_batch
        message = 'Epoch: %d, testing Loss %.3f, testing accuracy: %.3f' % (epoch, running_loss, acc)
        save_log(message)
        message = '*' * 40
        save_log(message)

    return acc

def test(net, loader):
    net.eval()
    label = []
    running_batch = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(loader):
            data = data.to(device)
            output = net(data)
            _, label_pred = torch.max(output, 1)
            label.extend(label_pred.cpu().numpy())
            running_batch += data.size(0)
    return running_batch, label


def save_networks(net, which_epoch):
    save_filename = '%s_net.pth' % (which_epoch)
    save_path = os.path.join(args.expr_dir, save_filename)
    if torch.cuda.is_available():
        try:
            torch.save(net.module.cpu().state_dict(), save_path)
        except:
            torch.save(net.cpu().state_dict(), save_path)
    else:
        torch.save(net.cpu().state_dict(), save_path)


def weights_init(m, type='kaiming'):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1 or classname.find('Conv2d') != -1:
        if type == 'xavier':
            nn.init.xavier_normal_(m.weight)
        elif type == 'kaiming':
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif type == 'orthogonal':
            nn.init.orthogonal_(m.weight)
        elif type == 'gaussian':
            m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

if __name__ == '__main__':

    net = MLP(input_size=args.input_size, output_size=args.output_size)
    net.apply(weights_init)
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    # Set learning rate decay steps
    str_steps = args.decay_steps.split(',')
    args.decay_steps = []
    for str_step in str_steps:
        str_step = int(str_step)
        args.decay_steps.append(str_step)
    scheduler = MultiStepLR(optimizer, milestones=args.decay_steps, gamma=0.1)

    save_log('Logging data')
    train_data = load_dataset(args.train_data_path, args.train_label_path)
    train_loader = DataLoader(dataset=train_data, num_workers=4, batch_size=args.batch_size, pin_memory=True,
                              shuffle=True)
    val_data = load_dataset(args.val_data_path, args.val_label_path)
    val_loader = DataLoader(dataset=val_data, num_workers=4, batch_size=args.batch_size, pin_memory=True,
                            shuffle=False)
    save_log('Data is loaded')
    # ------------------------
    # Start Training and Validating
    # ------------------------
    cur_acc = 0
    for epoch in range(1, args.num_epochs + 1):
        net.to(device)

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        message = '{}: {}/{} , {}: {:.4f}'.format('epoch', epoch, args.num_epochs, 'lr', lr)
        save_log(message)
        save_log('-' * 10)

        train(net, train_loader, optimizer, criterion, epoch)
        if epoch % args.eval_step == 0:
            val_acc = validate(net, val_loader, criterion, epoch)

            if val_acc > cur_acc:
                save_networks(net, epoch)
                cur_acc = val_acc

        # if epoch % args.save_step == 0:
        #     save_networks(epoch)
    save_networks(net, epoch)

    # ------------------------
    # Start Testing
    # ------------------------
    save_log('Loading test data')
    test_data = load_dataset(args.test_data_path)
    test_loader = DataLoader(dataset=test_data, num_workers=4, batch_size=args.batch_size, pin_memory=True, shuffle=False)
    save_log('Test data is loaded')
    net.to(device)
    test_num, test_label = test(net, test_loader)
    d = {'id': list(range(test_num)), 'label': test_label}
    df = pd.DataFrame(data=d)
    df.to_csv(args.file_name, header=True, index=False)
    save_log('Testing is done, result is saved to {}'.format(args.result_file_name))

