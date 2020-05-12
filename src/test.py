import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
import argparse
import os
import datetime

parser = argparse.ArgumentParser(description='speech_recognition')

parser.add_argument('--test_data_path', default='../data/test.npy', type=str)
parser.add_argument('--batch_size', default=512, type=int, help='batch size')
parser.add_argument('--input_size', default=1000, type=int, help='input size')
parser.add_argument('--context_size', default=12, type=int, help='context size')
parser.add_argument('--feature_dim', default=138, type=int, help='feature dimension')
parser.add_argument('--file_name', default='hw1p2_test_result.csv', type=str, help='testing result save path')
parser.add_argument('--checkpoint_dir', default='../checkpoints/', help='checkpoint folder root')
parser.add_argument('--model_path', default='../20_net.pth', help='the path of the testing model')

args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
args.expr_dir = os.path.join(args.checkpoint_dir, current_time)
os.makedirs(args.expr_dir)
args.file_name = os.path.join(args.expr_dir, args.file_name)


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


class load_dataset(Dataset):
    def __init__(self, data_path, label_path=None):
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
        x = self.data[i].take(range(j - args.context_size, j + args.context_size + 1), mode='clip', axis=0).flatten()
        y = np.int32(self.label[i][j]).reshape(1) if self.label is not None else np.int32(-1).reshape(1)
        return torch.from_numpy(x).float(), torch.LongTensor(y)

    def __len__(self):
        return len(self.idx_map)


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


if __name__ == '__main__':
    model = torch.load(args.model_path)
    net = MLP(input_size=args.input_size, output_size=args.feature_dim)
    net.load_state_dict(model)
    print('Loading test data')
    test_data = load_dataset(args.test_data_path)
    test_loader = DataLoader(dataset=test_data, num_workers=4, batch_size=args.batch_size, pin_memory=True, shuffle=False)
    print('Test data is loaded')
    test_num, test_label = test(net, test_loader)
    d = {'id': list(range(test_num)), 'label': test_label}
    df = pd.DataFrame(data=d)
    df.to_csv(args.file_name, header=True, index=False)
    print('Testing is done, result is saved to {}'.format(args.file_name))