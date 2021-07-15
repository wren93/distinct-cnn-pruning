import os
import argparse
import copy

import torch
from torch.utils.data import DataLoader

from lib.vgg11 import VGG11
from lib.dataset import VehicleX
from lib.trainer import Trainer
from lib.utils import *


def train(opt):
    # set torch seed
    torch.manual_seed(opt.seed)
    # set up training set
    gt_path = opt.gt_path

    trainset_root = opt.trainset_root
    trainset = VehicleX(trainset_root, gt_path)
    trainset_loader = DataLoader(
        trainset,
        batch_size=opt.batch_size_train,
        shuffle=True,
        pin_memory=True,
        num_workers=0)

    # set up validation set
    valset_root = opt.valset_root
    valset = VehicleX(valset_root, gt_path)
    valset_loader = DataLoader(
        valset,
        batch_size=opt.batch_size_val,
        shuffle=True,
        pin_memory=True,
        num_workers=0)

    # training configs
    img_size = opt.img_size
    num_hidden = opt.num_hidden
    num_output = opt.num_output

    num_epoch = opt.epoch
    lr = opt.lr

    # define base network
    net = VGG11(
        num_hidden=num_hidden,
        num_output=num_output,
        img_size=img_size
    )

    if opt.finetune:
        load_pruned_model(net, opt.load_pretrained)
    # define optimizer
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    trainer = Trainer(net, trainset_loader, optimizer)
    trainer.set_device(opt.device)

    print("start training...")
    # training
    for epoch in range(num_epoch):
        final_loss = trainer.train_epoch()
        if (epoch + 1) % 5 == 0:
            # every 5 epoch, test the network on validation set
            with torch.no_grad():
                val_loss = val(copy.deepcopy(net), valset_loader, opt.device)
            print('Epoch {}: training loss - {}, validation loss - {}'.format(epoch + 1, final_loss, val_loss))
        else:
            print('Epoch {}: training loss - {}'.format(epoch + 1, final_loss))

    # save model
    save_model(net, os.path.join(opt.save_path, opt.output_name))


def val(net, dataloader, device):
    device = torch.device(device)
    running_loss = 0
    for iter_id, batch in enumerate(dataloader):
        for item in batch:
            batch[item] = batch[item].to(device)

        # network forward & computing loss
        loss, _ = net(batch)
        running_loss += loss.item()
    return running_loss / (iter_id + 1)


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

    # network settings
    parser.add_argument('--img_size', default=(256, 256))
    parser.add_argument('--num_hidden', type=int, default=4096)
    parser.add_argument('--num_output', type=int, default=11)

    # training settings
    # if finetuning pruned models, set to 'store_false' (opt.finetune = True)
    # then uncomment the 'load_pretrained' line
    parser.add_argument('--finetune', action='store_false')
    parser.add_argument('--load_pretrained', default='./pretrained/l1/vgg11_pruned_l1_dot80.pth')
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size_train', type=int, default=32)
    parser.add_argument('--batch_size_val', type=int, default=32)
    parser.add_argument('--save_path', default='./pretrained/l1')
    parser.add_argument('--output_name', default='vgg11_finetuned_l1_dot80.pth')

    opt = parser.parse_args()

    # start training
    train(opt)
