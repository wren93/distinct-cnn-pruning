import os
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
import torch.nn.functional as F

from lib.vgg11 import VGG11
from lib.dataset import VehicleX
from lib.utils import *


def prune(opt):
    """ collect pattern activation vectors """
    # set torch seed
    torch.manual_seed(opt.seed)
    # set up training set
    gt_path = opt.gt_path

    if opt.dataset_phase == 'train':
        trainset_root = opt.trainset_root
    elif opt.dataset_phase == 'test':
        trainset_root = opt.testset_root
    elif opt.dataset_phase == 'validation':
        trainset_root = opt.valset_root
    else:
        raise Exception("Invalid dataset phase: {}".format(opt.dataset_phase))

    trainset = VehicleX(trainset_root, gt_path)

    if opt.num_samples != -1:
        sampler = RandomSampler(trainset, replacement=True, num_samples=opt.num_samples)
    else:
        sampler = None
    trainset_loader = DataLoader(
        trainset,
        batch_size=opt.batch_size,
        shuffle=False,
        sampler=sampler,
        pin_memory=True,
        num_workers=0)

    # load pretrained model
    img_size = opt.img_size
    num_hidden = opt.num_hidden
    num_output = opt.num_output

    # define base network
    net = VGG11(
        num_hidden=num_hidden,
        num_output=num_output,
        img_size=img_size
    )
    # load pretrained model
    net = load_model(net, opt.load_pretrained)

    device = torch.device(opt.device)
    net.to(device=device)

    print("collecting activation vectors...")
    pattern_vector = []    
    features = net.features

    with torch.no_grad():
        for iter_id, batch in enumerate(trainset_loader):
            for item in batch:
                batch[item] = batch[item].to(device)
            x = batch['input']
            y = batch['label'].squeeze()
            
            i = 0
            for module in features:
                x = module(x)
                if isinstance(module, nn.Conv2d):
                    hidden = x.clone()
                    # hidden = F.relu(hidden)
                    hidden = hidden.permute(1, 0, 2, 3)
                    if iter_id == 0:
                        pattern_vector.append(hidden)
                    else:
                        pattern_vector[i] = torch.cat([pattern_vector[i], hidden], dim=1)
                    
                    i += 1

    filter_index_list = []
    print("distinctiveness-based pruning...")
    for k in range(len(pattern_vector)):
        num_filter = pattern_vector[k].shape[0]
        pattern = pattern_vector[k].view(num_filter, -1)
        # pattern /= torch.max(pattern)
        # pattern -= 0.5
        pattern = F.normalize(pattern, p=2, dim=1)

        """ compute cosine similarity for every pair of hidden unit """
        print("computing pairwise angles...")
        cosine_distance = torch.matmul(pattern, pattern.t())

        """ pruning based on distinctiveness """

        filter_to_prune = []
        sim = opt.similar_thres
        comp = opt.complementary_thres
        for i in range(num_filter):
            cosine_distance[i, i] = 0.
            if i in filter_to_prune:
                continue
            for j in range(num_filter):
                if j == i or j in filter_to_prune:
                    continue
                if cosine_distance[i, j] >= np.cos(2 * np.pi * (sim / 360.)):
                    filter_to_prune.append(j)
                elif cosine_distance[i, j] <= np.cos(2 * np.pi * (comp / 360.)):
                    if i not in filter_to_prune:
                        filter_to_prune.append(i)
                    filter_to_prune.append(j)

        filter_index = torch.ones(num_filter, dtype=torch.bool)
        filter_index[filter_to_prune] = False
        filter_index_list.append(filter_index)

        print("pruned {} neurons from Conv layer {}".format(len(filter_to_prune), k + 1))
        k += 1
    
    i = 0
    pruned_net_weights = []
    for module in features:
        if not isinstance(module, nn.Conv2d):
            continue
        weight = module.weight.data.clone()
        bias = module.bias.data.clone()
        if i > 0:
            weight = weight[:, filter_index_list[i - 1], :, :]
        weight = weight[filter_index_list[i], :, :, :]
        bias = bias[filter_index_list[i]]
        pruned_net_weights.append((weight, bias))
        i += 1
    
    final_fm_size = int(img_size[0] / 32.)
    fc_index = torch.zeros([512, final_fm_size, final_fm_size], dtype=torch.bool)
    fc_index[filter_index_list[-1], :, :] = True
    fc_index = fc_index.flatten()

    fc_weight = net.classifier[0].weight.data
    pruned_fc_weight = fc_weight[:, fc_index]

    # load pretrained model
    img_size = opt.img_size
    num_hidden = opt.num_hidden
    num_output = opt.num_output

    # define base network
    pruned_net = VGG11(
        num_hidden=num_hidden,
        num_output=num_output,
        img_size=img_size
    )
    pruned_net.load_state_dict(net.state_dict())

    i = 0
    for module in pruned_net.features:
        if isinstance(module, nn.Conv2d):
            module.weight.data = pruned_net_weights[i][0].clone()
            module.bias.data = pruned_net_weights[i][1].clone()
            i += 1
    pruned_net.classifier[0].weight.data = pruned_fc_weight.clone()


    save_model(pruned_net, os.path.join(opt.save_path, opt.output_name))

    return pruned_net


def prune_l1norm(opt, ratio=0.4):
    # set torch seed
    torch.manual_seed(opt.seed)

    # load pretrained model
    img_size = opt.img_size
    num_hidden = opt.num_hidden
    num_output = opt.num_output

    # define base network
    net = VGG11(
        num_hidden=num_hidden,
        num_output=num_output,
        img_size=img_size
    )
    # load pretrained model
    net = load_model(net, opt.load_pretrained)

    device = torch.device(opt.device)
    net.to(device=device)

    features = net.features

    filter_index_list = []
    for module in features:
        if not isinstance(module, nn.Conv2d):
            continue
        weight = module.weight.data.clone()
        bias = module.bias.data.clone()

        norm = torch.linalg.norm(weight, dim=(1, 2, 3))
        idx = torch.argsort(norm)
        remainer = idx[(int(ratio * weight.shape[0]) + 1):]
        filter_index = torch.zeros(weight.shape[0], dtype=torch.bool)
        filter_index[remainer] = True
        filter_index_list.append(filter_index)
    
    i = 0
    pruned_net_weights = []
    for module in features:
        if not isinstance(module, nn.Conv2d):
            continue
        weight = module.weight.data.clone()
        bias = module.bias.data.clone()
        if i > 0:
            weight = weight[:, filter_index_list[i - 1], :, :]
        weight = weight[filter_index_list[i], :, :, :]
        bias = bias[filter_index_list[i]]
        pruned_net_weights.append((weight, bias))
        i += 1
    
    final_fm_size = int(img_size[0] / 32.)
    fc_index = torch.zeros([512, final_fm_size, final_fm_size], dtype=torch.bool)
    fc_index[filter_index_list[-1], :, :] = True
    fc_index = fc_index.flatten()

    fc_weight = net.classifier[0].weight.data
    pruned_fc_weight = fc_weight[:, fc_index]

    # load pretrained model
    img_size = opt.img_size
    num_hidden = opt.num_hidden
    num_output = opt.num_output

    # define base network
    pruned_net = VGG11(
        num_hidden=num_hidden,
        num_output=num_output,
        img_size=img_size
    )
    pruned_net.load_state_dict(net.state_dict())

    i = 0
    for module in pruned_net.features:
        if isinstance(module, nn.Conv2d):
            module.weight.data = pruned_net_weights[i][0].clone()
            module.bias.data = pruned_net_weights[i][1].clone()
            i += 1
    pruned_net.classifier[0].weight.data = pruned_fc_weight.clone()


    save_model(pruned_net, os.path.join(opt.save_path, opt.output_name))

    return pruned_net

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # torch settings
    parser.add_argument('--seed', type=int, default=223)
    parser.add_argument('--device', default='cuda',
                        help="'cuda' for GPU training, 'cpu' for CPU training")

    # pretrained model settings
    parser.add_argument('--load_pretrained', default='./pretrained/vgg11.pth')

    # dataset paths
    parser.add_argument('--trainset_root', default='./vehicle-x_v2/Classification Task/train')
    parser.add_argument('--testset_root', default='./vehicle-x_v2/Classification Task/test')
    parser.add_argument('--valset_root', default='./vehicle-x_v2/Classification Task/val')
    parser.add_argument('--gt_path', default='./vehicle-x_v2/finegrained_label.xml')

    # network settings
    parser.add_argument('--img_size', default=(256, 256))
    parser.add_argument('--num_hidden', type=int, default=4096)
    parser.add_argument('--num_output', type=int, default=11)

    # pruning settings
    parser.add_argument('--similar_thres', type=int, default=30)
    parser.add_argument('--complementary_thres', type=int, default=150)
    parser.add_argument('--num_samples', type=int, default=8)
    parser.add_argument('--dataset_phase', default='train',
                        help="dataset used for activation vector generation. Available options: 'train', 'test', 'validation, random'")
    parser.add_argument('--save_path', default='./pretrained/l1')
    parser.add_argument('--output_name', default='vgg11_pruned_l1_dot90.pth')
    parser.add_argument('--batch_size', type=int, default=1)

    opt = parser.parse_args()

    # start pruning
    # pruned_net = prune(opt)
    pruned_net = prune_l1norm(opt, ratio=0.90)
