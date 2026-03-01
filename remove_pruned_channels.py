import os
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms


from spikingjelly.clock_driven import functional
from snnvgg import *
_seed_ = 2020
random.seed(2020)

torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import utils
import numpy as np

np.random.seed(_seed_)



device='cuda:0'
# Prune settings

parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=164,
                    help='depth of the resnet')
parser.add_argument('--percent', type=float, default=0.1,
                    help='scale sparse rate (default: 0.5)')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--save', default='./', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


def test(model):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

   
    tr = datasets.CIFAR10('./', train=True, download=True,
                      transform=transforms.Compose([
                          transforms.RandomHorizontalFlip(),
                          transforms.RandomCrop(32, padding=4),
                          transforms.ToTensor(),
                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                      ]))

    te = datasets.CIFAR10('./', train=False,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                      ]))
    print(f'dataset_train:{tr.__len__()}, dataset_test:{te.__len__()}')
    
    train_loader = torch.utils.data.DataLoader(
        tr,
        batch_size=64, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        te,
        batch_size=64, shuffle=True, drop_last=True)

    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    correct = 0
    with torch.no_grad():

        for image, target in metric_logger.log_every(test_loader, print_freq=100, header='Test:'):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            #loss = criterion(output, target)
            #pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            #correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            functional.reset_net(model)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]

            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    acc1, acc5 = metric_logger.acc1.global_avg, metric_logger.acc5.global_avg
    print(acc1)

    return acc1

if not os.path.exists(args.save):
    os.makedirs(args.save)

model = snnvgg16_bn().to(device)
modules = list(model.modules())
'''for layer_id in range(len(modules)):
    #print(layer_id)
    print(modules[39].kernel_size)'''
total = sum([param.nelement() for param in model.parameters()])
print('  + Number of params: %.2fM' % (total / 1e6))
model.to(device)

checkpoint = torch.load('./vgg16cifar10best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
#checkpoint = torch.load('./vggylbest.pt')
#model.load_state_dict(checkpoint)
test(model)
cfg_mask = []


import csv


with open('maskfinal.csv', 'r') as file: #The maskfinnal.csv is the Mask record corresponding to the best saved model.
  
    csv_reader = csv.reader(file)
    
    
    cfg_mask = []
    
    
    for row in csv_reader:
      
        row=list(map(lambda x: int(float(x)) if x != '' else x, row))
        #print(row)
       
        row=torch.tensor(row)
        print(torch.sum(row == 1).item())
        
        cfg_mask.append(row)
# simple test model after Pre-processing prune (simple set BN scales to zeros)


acc = test(model)

print("Cfg:")


cfg=[64-0,64-0, 'M',  128-0, 128-0, 'M',  256-0, 256-0, 256-4,  'M', 512-50, 512-111, 512-62,  'M', 512-93,512-103, 512-0,'M',]    #This list is the model structure record corresponding to the best saved model.
newmodel = snnvgg16_bn(cfg=cfg)

newmodel.to(device)

total = sum([param.nelement() for param in newmodel.parameters()])
print('  + Number of params: %.2fM' % (total / 1e6))
savepath = os.path.join(args.save, "prune.txt")
'''with open(savepath, "w") as fp:
    fp.write("Configuration: \n"+str(cfg)+"\n")
    fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
    fp.write("Test accuracy: \n"+str(acc))'''

old_modules = list(model.modules())
new_modules = list(newmodel.modules())
layer_id_in_cfg = 0
start_mask = torch.ones(3)
end_mask = cfg_mask[layer_id_in_cfg]
conv_count = 0




bn_count = 0


j=0
s=0
for [m0, m1] in zip(model.modules(), newmodel.modules()):
    

    if isinstance(m0, nn.BatchNorm2d):
       
            
        
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        if idx1.size == 1:
            idx1 = np.resize(idx1,(1,))
        m1.weight.data = m0.weight.data[idx1.tolist()].clone()
        m1.bias.data = m0.bias.data[idx1.tolist()].clone()
        m1.running_mean = m0.running_mean[idx1.tolist()].clone()
        m1.running_var = m0.running_var[idx1.tolist()].clone()
        layer_id_in_cfg += 1
        start_mask = end_mask.clone()
        if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
            end_mask = cfg_mask[layer_id_in_cfg]
        
        
    elif isinstance(m0, nn.BatchNorm1d):
        
        bb = torch.ones(512)
        idx1 = np.squeeze(np.argwhere(np.asarray(bb.cpu().numpy())))
        if idx1.size == 1:
            idx1 = np.resize(idx1,(1,))
        m1.weight.data = m0.weight.data[idx1.tolist()].clone()
        m1.bias.data = m0.bias.data[idx1.tolist()].clone()
        m1.running_mean = m0.running_mean[idx1.tolist()].clone()
        m1.running_var = m0.running_var[idx1.tolist()].clone()
            
    elif isinstance(m0, nn.Conv2d):
        #print('aaa')
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        if idx1.size == 1:
            idx1 = np.resize(idx1, (1,))
        w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
        #print(w1.shape)
        #print(w1[idx1.tolist(), :, :, :].shape)
        #print(idx1)
        w1 = w1[idx1.tolist(), :, :, :].clone()
        m1.weight.data = w1.clone()
    elif isinstance(m0, nn.Linear):
        if j==0:
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            print(idx0.size)
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            m1.weight.data = m0.weight.data[:, idx0].clone()
            if m1.bias is not None:
                print('bias')
    
                m1.bias.data = m0.bias.data.clone()
            j=j+1
        else:
            aa = torch.ones(512)
            #aa=cfg1[0].clone()
            idx0 = np.squeeze(np.argwhere(np.asarray(aa.cpu().numpy())))
            #print(idx0)
            #idx0=float(idx0)
            #idx0=np.asarray(idx0.cpu().numpy())
            #print(idx0.size)
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            m1.weight.data = m0.weight.data[:, idx0].clone()
            if m1.bias is not None:
                print('bias')
    
                m1.bias.data = m0.bias.data.clone()
            j=j+1
            
        

torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(args.save, 'pruned.pth.tar'))
'''for layer_id in range(len(old_modules)):
    print(layer_id)
    print(new_modules[layer_id])'''
#print(newmodel)
model = newmodel
test(model)
