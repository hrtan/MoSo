

'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models
import os
import argparse
import model
from tqdm import tqdm  
import random
import numpy as np
from utils import progress_bar
from utils import *
from models import *
from torch.utils.data import DataLoader
from torch.autograd import grad
import copy
torch.manual_seed(3407)

class ResNetT(nn.Module):
    def __init__(self, cfg="resnet50"):
        super().__init__()
        
        if cfg == "resnet18":
            self.base = torchvision.models.resnet18(num_classes=200)
        elif cfg == "resnet34":
            self.base = torchvision.models.resnet34(num_classes=200)
        elif cfg == "resnet50":
            self.base = torchvision.models.resnet50(num_classes=200)
        else:
            raise NotImplementedError()
        
        self.base.avgpool =  nn.AdaptiveAvgPool2d((1,1))
        #self.base.fc.apply(weight_init_kaiming)
    def forward(self, x):
        return self.base(x)

def MoSo_scoring(net, dataloader, criterion, lr = None):
    model = copy.deepcopy(net)
    model.eval()
    overall_grad = 0
    M = 0
    params = [ p for p in model.parameters() if p.requires_grad ]
    for i, (inputs, labels) in tqdm(enumerate(dataloader)):
        inputs = inputs.cuda()
        labels = labels.cuda()
        logits = model(inputs)
        loss = criterion(logits, labels)
        g = list(grad(loss, params, create_graph=False))
        g = torch.nn.utils.parameters_to_vector(g)
        g = g.detach()
        overall_grad = overall_grad * i/(i+1) + g / (i+1)
        N = i+1
    overall_grad = overall_grad
    
    score_list = []
    for i, (inputs, labels) in tqdm(enumerate(dataloader)):
        inputs = inputs.cuda()
        labels = labels.cuda()
        logits = model(inputs)
        loss = criterion(logits, labels)
        g = list(grad(loss, params, create_graph=False))
        g = torch.nn.utils.parameters_to_vector(g)
        g = g.detach()
        score = ((overall_grad - 1/N * g) * g).sum()
        score = score.detach().cpu()#.numpy()
        score_list.append(score)
    #print(score_list)
    score_list = torch.tensor(score_list).detach()
    return score_list


def MoSo_scoring_exact(net, dataloader, criterion, lr = None):
    model = copy.deepcopy(net)
    model.eval()
    overall_grad = 0
    M = 0
    params = [ p for p in model.parameters() if p.requires_grad ]
    for i, (inputs, labels) in tqdm(enumerate(dataloader)):
        inputs = inputs.cuda()
        labels = labels.cuda()
        logits = model(inputs)
        loss = criterion(logits, labels)
        g = list(grad(loss, params, create_graph=False))
        g = torch.nn.utils.parameters_to_vector(g)
        g = g.detach()
        overall_grad = overall_grad * i/(i+1) + g / (i+1)
        N = i+1
    overall_grad = overall_grad
    
    score_list = []
    for i, (inputs, labels) in tqdm(enumerate(dataloader)):
        inputs = inputs.cuda()
        labels = labels.cuda()
        logits = model(inputs)
        loss = criterion(logits, labels)
        g = list(grad(loss, params, create_graph=False))
        g = torch.nn.utils.parameters_to_vector(g)
        g = g.detach()
        #score = ((overall_grad - 1/N * g) * g).sum()
        score = (2*N-3)/(N-1)**2 * (overall_grad * overall_grad).sum() - 1/(N-1)**2 * (g * g).sum() + (2*N - 4)/(N-1)**2 * ((overall_grad - 1/N * g) * g).sum()
        score = score.detach().cpu()#.numpy()
        score_list.append(score)
    #print(score_list)
    score_list = torch.tensor(score_list).detach()
    return score_list

def extract_feature(net, dataloader):
    feature_list = []
    label_list = []
    net.eval()
    for i, (inputs, labels) in tqdm(enumerate(dataloader)):
        # compute output
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs,_ = net(inputs,out_feature=True)
        feature_list.append(outputs.squeeze().cpu().data)
        label_list.append(labels.item())
    return [feature_list,label_list]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset Pruning')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset: cifar10/cifar100/tiny')
    parser.add_argument('--model', default='resnet50', type=str, help='model')
    parser.add_argument('--ckptfreq', default=1, type=int, help='saving interval')
    parser.add_argument('--path', default='./MoSo_CIFAR100', type=str, help='the path of this exp')
    parser.add_argument('--bs', default=256, type=int, help='batchsize')
    parser.add_argument('--num_trails', default=8, type=int, help='number of trials')
    parser.add_argument('--maxepoch', default=50, type=int, help='max epoch')
    parser.add_argument('--samples', default=10, type=int, help='number of selected ckpts')
    parser.add_argument('--noise_ratio', default=0.0, type=float, help='noise_ratio')
    parser.add_argument('--trainaug', default=0, type=int, help='0: None, 1: AutoAug (Cifar10), 2: RandAug, 3: AugMix')
    #ckptfreq , cls_indim, num_classes
    args = parser.parse_args()
    #ckptfreq , cls_indim, num_classes
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    maxepoch = args.maxepoch

    # Data
    print('==> Preparing data..' + args.dataset)
    if args.dataset == 'tiny':
        mean = [0.4802, 0.4481, 0.3975]
        std = [0.2302, 0.2265, 0.2262]
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(55),
            transforms.Resize(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        transform_train_RandAug = transforms.Compose([
            transforms.RandomResizedCrop(55),
            transforms.Resize(64),
            transforms.RandAugment(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        transform_train_AugMix = transforms.Compose([
            transforms.RandomResizedCrop(55),
            transforms.Resize(64),
            #transforms.AugMix(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(int(64/0.875)),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        if True:
            if args.trainaug == 0:
                transform_train = transform_train
            elif args.trainaug == 3:
                transform_train = transform_train_AugMix
            else:
                transform_train = transform_train_RandAug
    else:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        transform_train_AutoAug = transforms.Compose([
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        transform_train_RandAug = transforms.Compose([
            transforms.RandAugment(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        transform_train_AugMix = transforms.Compose([
            #transforms.AugMix(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        if True:
            if args.trainaug == 0:
                transform_train = transform_train
            elif args.trainaug == 1:
                transform_train = transform_train_AutoAug
            elif args.trainaug == 2:
                transform_train = transform_train_RandAug
            elif args.trainaug == 3:
                transform_train = transform_train_AugMix
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    # Dataset
    print('==> Building model..')
    if args.dataset == 'cifar10': #classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        cls_outdim = 10
        trainset = torchvision.datasets.CIFAR10(root='/mnt/workspace/cifar10/exp1/data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='/mnt/workspace/cifar10/exp1/data', train=False, download=True, transform=transform_test)
        wholedataset = torchvision.datasets.CIFAR10(root='/mnt/workspace/cifar10/exp1/data', train=True, download=True, transform=transform_test)
    elif args.dataset == 'cifar100':
        cls_outdim = 100
        trainset = torchvision.datasets.CIFAR100(root='/earth-nas/datasets/cifar-100-python', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='/earth-nas/datasets/cifar-100-python', train=False, download=True, transform=transform_test)
        wholedataset = torchvision.datasets.CIFAR100(root='/earth-nas/datasets/cifar-100-python', train=True, download=True, transform=transform_test)
    elif args.dataset == 'tiny':
        cls_outdim = 200
        train_set_path = os.path.join('/mnt/workspace/workgroup/tanhaoru.thr/dataset/tiny-imagenet-200', 'train')
        test_set_path = os.path.join('/mnt/workspace/workgroup/tanhaoru.thr/dataset/tiny-imagenet-200', 'val')
        trainset = ImageFolder(root=train_set_path, transform=transform_train)
        testset = ImageFolder(root=test_set_path, transform=transform_test)
        wholedataset = ImageFolder(root=train_set_path, transform=transform_train)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=1)
    wholeloader = torch.utils.data.DataLoader(
        wholedataset, batch_size=1, shuffle=False, num_workers=1)
    
    # Model
    # net = VGG('VGG19')
    cls_indim = -10
    if args.model == 'resnet18':
        net = ResNet18(cls_indim, cls_outdim)
        if args.dataset == 'tiny':
            net = ResNetT('resnet18') #torchvision.models.resnet18(num_classes=200)
    elif args.model == 'resnet50':
        net = ResNet50(cls_indim, cls_outdim)
        if args.dataset == 'tiny':
            net = ResNetT('resnet50') #torchvision.models.resnet50(num_classes=200)
    elif args.model == 'senet50':
        net = SENet18(cls_indim, cls_outdim)
    elif args.model == 'mobilenetv2':
        net = MobileNetV2(cls_indim, cls_outdim)
    else:
        print('no valid network specified')
    
    #if args.dataset == 'tiny':
    #    net.avgpool = nn.AdaptiveAvgPool2d(1)
    #    num_ftrs = net.fc.in_features
    #    net.fc = nn.Linear(num_ftrs, 200)

    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = ShuffleNetV2(1)
    # net = EfficientNetB0()
    # net = RegNetX_200MF()
    # net = SimpleDLA()
    # net = model.ResNet18()

    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    dataset = args.dataset
    trial_number = args.num_trails
    support_set_size = len(trainset)//trial_number
    trial_mask_support = []
    for i in range(trial_number):
        if i != trial_number-1:
            start = i * support_set_size
            end = (i + 1) * support_set_size
            temp_mask = [int(j>=start and j<end) for j in range(len(trainset))]
            trial_mask_support.append(temp_mask)
        else:
            start = i * support_set_size
            end = len(trainset)
            temp_mask = [int(j>=start and j<end) for j in range(len(trainset))]
            trial_mask_support.append(temp_mask)
    
    # load the mask
    if args.noise_ratio != 0.0:
        # find the noise-label path
        noise_mask_path = os.path.join(args.path, 'noise_mask')
        assert os.path.exists(noise_mask_path), "the noise-label path not exists"
        noise_label_path = os.path.join(noise_mask_path, 'label.pth')
        temp = torch.load(noise_label_path)
        trainset.targets = temp

    # Scoring
    if True:
        saving_root = os.path.join(args.path, 'score')
        if not os.path.isdir(saving_root):
            os.makedirs(saving_root)
        ckpt_saving_root = os.path.join(args.path, 'checkpoint')
        # get all saved ckpt file names
        ckpt_file_list = os.listdir(ckpt_saving_root)
        ckpt_file_list = [file_name  for file_name in ckpt_file_list if file_name.find('.pth')!=-1 and file_name.find('best')==-1 and file_name.find('trial')!=-1]
        
        trial_numbers = list(range(trial_number))
        
        scores_all_trials = 0
        for trial_index in trial_numbers:
            # define the trial name
            trial_name = 'trial_' + str(trial_index) + '_'
            # get the mask
            support_mask = trial_mask_support[trial_index]
            #AOSP
            query_mask = [1-i for i in support_mask]
            #MOSO-P
            query_mask = support_mask#[1-i for i in support_mask]
            # conrtruct the support set
            support_set = copy.deepcopy(trainset)
            if args.dataset == 'tiny':
                support_set.samples = [support_set.samples[ind] for ind in range(len(support_mask)) if support_mask[ind] == 1]
            else:
                support_set.data = [support_set.data[ind, :] for ind in range(len(support_mask)) if support_mask[ind] == 1]
                support_set.data = torch.tensor(support_set.data).numpy()
            support_set.targets = [support_set.targets[ind] for ind in range(len(support_mask)) if support_mask[ind]==1]
            support_loader = torch.utils.data.DataLoader(support_set, batch_size=1, shuffle=False, num_workers=1)
            # get all ckpts of the current trial
            temp_file_list = []
            for file_name in ckpt_file_list:
                if file_name.find(trial_name)!=-1:
                    temp_file_list.append(file_name)
            # sample
            temp_file_list = random.sample(temp_file_list, args.samples)
            trial_scores = 0
            for ckpt_name in temp_file_list:
                print('processing ' + ckpt_name + ' .......')
                ckpt_path = os.path.join(ckpt_saving_root, ckpt_name)
                ckpt = torch.load(ckpt_path, map_location='cpu')
                current_lr = ckpt['lr']
                current_epoch = ckpt['epoch']
                current_acc = ckpt['acc']
                current_net = ckpt['net']
                # load the ckpt's parameters
                model = copy.deepcopy(net)
                model.load_state_dict(current_net)
                # moso scoring
                #scores = MoSo_scoring(model, support_loader, criterion)
                #scores = MoSo_scoring(classifier, sup_grad, current_lr, [features,labels], device, criterion)
                scores = MoSo_scoring_exact(model, support_loader, criterion)
                scores = scores * current_lr[-1]
                scores = scores.numpy()
                trial_scores = trial_scores + scores
            #scores_all_trials.append(trial_scores)
            demasked_scores = []
            flag = -1
            for query_index in range(len(query_mask)):
                if query_mask[query_index]==1:
                    flag = flag + 1
                    demasked_scores.append(trial_scores[flag])
                else:
                    demasked_scores.append(0)
            demasked_scores = torch.tensor(demasked_scores).cpu().numpy()
            scores_all_trials = scores_all_trials + demasked_scores
        # saving
        scores_all_trials = torch.tensor(scores_all_trials)
        score_save_path = os.path.join(saving_root, 'moso_score.pth')
        torch.save(scores_all_trials, score_save_path)