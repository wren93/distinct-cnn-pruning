import time
import argparse
import torch

import torch.nn.functional as F
from torch.utils.data import DataLoader, dataloader

from lib.vgg11 import VGG11, VGG11_test
from lib.dataset import VehicleX
from lib.utils import *


def test(opt):
    # set torch seed
    torch.manual_seed(opt.seed)
    # set up test set
    if opt.dataset_phase == 'train':
        testset_root = opt.trainset_root
    elif opt.dataset_phase == 'test':
        testset_root = opt.testset_root
    elif opt.dataset_phase == 'validation':
        testset_root = opt.valset_root
    else:
        raise Exception("Invalid dataset phase: {}".format(opt.dataset_phase))
    gt_path = opt.gt_path
    testset = VehicleX(testset_root, gt_path)
    testset_loader = DataLoader(
        testset,
        batch_size=opt.batch_size_test,
        shuffle=True,
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
    if opt.pruned:
        net = load_pruned_model(net, opt.load_pretrained)
    else:
        net = load_model(net, opt.load_pretrained)

    device = torch.device(opt.device)
    net.to(device=device)

    # compute accuracy
    correct = 0
    with torch.no_grad():
        for iter_id, batch in enumerate(testset_loader):
            for item in batch:
                batch[item] = batch[item].to(device)
            _, output = net(batch)
            output = F.softmax(output, dim=1)
            y_pred = output.argmax(-1)
            y = batch['label'].squeeze()
            correct += torch.sum(y == y_pred).item()

    accuracy = correct / ((iter_id + 1) * opt.batch_size_test)

    return accuracy

def count_param(opt):
    from ptflops import get_model_complexity_info
    # set torch seed
    torch.manual_seed(opt.seed)
    # load pretrained model
    img_size = opt.img_size
    num_hidden = opt.num_hidden
    num_output = opt.num_output

    # define base network
    net = VGG11_test(
        num_hidden=num_hidden,
        num_output=num_output,
        img_size=img_size
    )
    # load pretrained model
    if opt.pruned:
        net = load_pruned_model(net, opt.load_pretrained)
    else:
        net = load_model(net, opt.load_pretrained)

    device = torch.device(opt.device)
    net.to(device=device)


    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(net, (3, 256, 256), as_strings=True,
                                                 print_per_layer_stat=False, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

def test_speed(opt):
    # set torch seed
    torch.manual_seed(opt.seed)
    # set up test set
    testset_root = opt.testset_root
    gt_path = opt.gt_path
    testset = VehicleX(testset_root, gt_path)
    testset_loader = DataLoader(
        testset,
        batch_size=1,
        shuffle=True,
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
    if opt.pruned:
        net = load_pruned_model(net, opt.load_pretrained)
    else:
        net = load_model(net, opt.load_pretrained)

    device = torch.device(opt.device)
    net.to(device=device)

    with torch.no_grad():
        t_sum = 0.
        for _, batch in enumerate(testset_loader):
            for item in batch:
                batch[item] = batch[item].to(device)
            t1 = time.time()
            _, _ = net(batch)
            t2 = time.time()
            t_sum += (t2 - t1)
        t_sum /= len(testset_loader)
        return 1 / t_sum

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # torch settings
    parser.add_argument('--seed', type=int, default=223)
    parser.add_argument('--device', default='cuda',
                        help="'cuda' for GPU training, 'cpu' for CPU training")
    # dataset paths
    parser.add_argument('--trainset_root', default='./vehicle-x_v2/Classification Task/train')
    parser.add_argument('--testset_root', default='./vehicle-x_v2/Classification Task/test')
    parser.add_argument('--valset_root', default='./vehicle-x_v2/Classification Task/val')
    parser.add_argument('--gt_path', default='./vehicle-x_v2/finegrained_label.xml')

    # pretrained model settings
    # if loading pruned models, set to 'store_false' (opt.pruned = True)
    parser.add_argument('--pruned', action='store_false')
    parser.add_argument('--load_pretrained', default='./pretrained/vgg11.pth')

    # network settings
    parser.add_argument('--img_size', default=(256, 256))
    parser.add_argument('--num_hidden', type=int, default=4096)
    parser.add_argument('--num_output', type=int, default=11)

    # test settings
    parser.add_argument('--batch_size_test', type=int, default=32)
    parser.add_argument('--dataset_phase', default='test',
                        help="dataset used for testing. Available options: 'train', 'test', 'validation'")

    opt = parser.parse_args()

    print("{} accuracy: {}".format(opt.dataset_phase, test(opt)))
    count_param(opt)
    # print("FPS: {}".format(test_speed(opt)))
