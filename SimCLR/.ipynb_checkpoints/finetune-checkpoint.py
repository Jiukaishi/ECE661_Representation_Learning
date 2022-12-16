import argparse

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from thop import profile, clever_format
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm
import numpy as np
import utils
from model import Model


class Net(nn.Module):
    def __init__(self, num_class, pretrained_path):
        super(Net, self).__init__()

        # encoder
        self.f = Model().f
        # classifier
        self.fc = nn.Linear(2048, num_class, bias=True)
        self.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out

# def train(net, data_loader, train_optimizer):
#     net.train()
#     total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
#     for pos_1, pos_2, target in train_bar:
#         pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
#         feature_1, out_1 = net(pos_1)
#         feature_2, out_2 = net(pos_2)
#         # [2*B, D]
#         out = torch.cat([out_1, out_2], dim=0)
#         # [2*B, 2*B]
#         sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
#         mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
#         # [2*B, 2*B-1]
#         sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
#         #去掉和自己的similarity
#         # compute loss
#         pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
#         # [2*B]
#         pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
#         loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
#         train_optimizer.zero_grad()
#         loss.backward()
#         train_optimizer.step()

#         total_num += batch_size
#         total_loss += loss.item() * batch_size
#         train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

#     return total_loss / total_num

# train or test for one epoch
def train_val(net, data_loader, train_optimizer):
    is_train = train_optimizer is not None
    net.train() if is_train else net.eval()

    total_loss, total_correct_1, total_correct_5, total_num, data_bar = 0.0, 0.0, 0.0, 0, tqdm(data_loader)
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, target in data_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            out = net(data)
            loss = loss_criterion(out, target)

            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            prediction = torch.argsort(out, dim=-1, descending=True)
            total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.2f}% ACC@5: {:.2f}%'
                                     .format('Train' if is_train else 'Test', epoch, epochs, total_loss / total_num,
                                             total_correct_1 / total_num * 100, total_correct_5 / total_num * 100))

    return total_loss / total_num, total_correct_1 / total_num * 100, total_correct_5 / total_num * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Semi-supervise Finetune')
    parser.add_argument('--percent', type=int, default=10, help='percentage of the dataset to train')
    parser.add_argument('--model_path', type=str, default='results/128_0.5_200_512_500_model.pth',
                        help='The pretrained model path')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', type=int, default=100, help='Number of sweeps over the dataset to train')
    

    args = parser.parse_args()
    
    
    
    model_path, batch_size, epochs = args.model_path, args.batch_size, args.epochs
    train_data = CIFAR10(root='data', train=True, transform=utils.train_transform, download=True)
    label2idx =[[],[],[],[],[],[],[],[],[],[]]
    for i in range(len(train_data)):
        label2idx[train_data[i][1]].append(i)
    label2idx = np.array(label2idx)
    one_percent_idx = np.ones((10, int(5000*args.percent/100)))
    for i in range(10):
        one_percent_idx[i] = np.random.choice(label2idx[i], int(5000*args.percent/100), replace=False)
    one_percent_idx = one_percent_idx.reshape(-1)
    np.random.shuffle(one_percent_idx)
    one_percent_idx=one_percent_idx.astype(int)
    print(one_percent_idx.dtype)
    part_sampler = torch.utils.data.SubsetRandomSampler(one_percent_idx)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=16, pin_memory=True, sampler =part_sampler)
    test_data = CIFAR10(root='data', train=False, transform=utils.test_transform, download=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    

    model = Net(num_class=len(train_data.classes), pretrained_path=model_path).cuda()
#     for param in model.f.parameters():
#         param.requires_grad = False

#     flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
#     flops, params = clever_format([flops, params])
#     print('# Model Params: {} FLOPs: {}'.format(params, flops))
#     # optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.05, momentum=0.9, nesterov=True)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    loss_criterion = nn.CrossEntropyLoss()
    results = {'train_loss': [], 'train_acc@1': [], 'train_acc@5': [],
               'test_loss': [], 'test_acc@1': [], 'test_acc@5': []}

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc_1, train_acc_5 = train_val(model, train_loader, optimizer)
        results['train_loss'].append(train_loss)
        results['train_acc@1'].append(train_acc_1)
        results['train_acc@5'].append(train_acc_5)
        test_loss, test_acc_1, test_acc_5 = train_val(model, test_loader, None)
        results['test_loss'].append(test_loss)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('results/linear_statistics.csv', index_label='epoch')
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            torch.save(model.state_dict(), 'results/linear_model.pth')
