import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm

import utils
from model import Model




# train or test for one epoch
def train_val(net, data_loader, train_optimizer):
    is_train = train_optimizer is not None
    net.train() if is_train else net.eval()

    total_loss, total_correct_1,  total_num, data_bar = 0.0, 0.0, 0, tqdm(data_loader)
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, target in data_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
           
            out = net(data)

            # print(out)
            # print(target.shape)
            loss = loss_criterion(out, target)

            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            prediction = torch.argsort(out, dim=-1, descending=True)
            total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
        
            data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.2f}%'
                                     .format('Train' if is_train else 'Test', epoch, epochs, total_loss / total_num, total_correct_1 / total_num * 100))

    return total_loss / total_num, total_correct_1 / total_num * 100,


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--feature_dim', default=4, type=int, help='Feature dim for latent vector')
    
    parser.add_argument('--batch_size', default=512, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=1000, type=int, help='Number of sweeps over the dataset to train')

    # args parse
    args = parser.parse_args()
    feature_dim = args.feature_dim
    batch_size, epochs = args.batch_size, args.epochs
    # data prepare
    train_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.train_transform, download=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True,
                              drop_last=True)
    memory_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.test_transform, download=True)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_data = utils.CIFAR10Pair(root='data', train=False, transform=utils.test_transform, download=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
   
   
    loss_criterion = nn.CrossEntropyLoss()
    results = {'train_loss': [], 'train_acc@1': []}
    test_res= {'test_loss': [], 'test_acc@1': []}
    # model setup and optimizer config
    model = Model(feature_dim).cuda()
    # flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
    # flops, params = clever_format([flops, params])
    # print('# Model Params: {} FLOPs: {}'.format(params, flops))
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    c = len(memory_data.classes)

    if not os.path.exists('results'):
        os.mkdir('results')
   
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc_1 = train_val(model, train_loader, optimizer)
        results['train_loss'].append(train_loss)
        results['train_acc@1'].append(train_acc_1)
        if epoch%10==0:
          test_loss, test_acc_1 = train_val(model, test_loader, None)
          test_res['test_loss'].append(test_loss)
          test_res['test_acc@1'].append(test_acc_1)

          # save statistics
          data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
          data_frame.to_csv('results/statistics_train.csv', index_label='epoch')
          data_frame = pd.DataFrame(data=test_res, index=range(1, len(test_res['test_acc@1']) + 1))
          data_frame.to_csv('results/statistics_test.csv', index_label='epoch')
          if test_acc_1 > best_acc:
              best_acc = test_acc_1
              torch.save(model.state_dict(), 'results/model.pth')
