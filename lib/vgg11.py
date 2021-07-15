import torch.nn as nn
import math


class VGG11(nn.Module):
    def __init__(self, num_hidden=4096, num_output=11, img_size=(256, 256)):
        super(VGG11, self).__init__()

        self.img_size = img_size
        self.num_output = num_output
        self.num_hidden = num_hidden

        self.features = nn.Sequential(
            # vgg block 1
            nn.Conv2d(in_channels=3, out_channels=64, padding=1, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # vgg block 2
            nn.Conv2d(in_channels=64, out_channels=128, padding=1, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # vgg block 3
            nn.Conv2d(in_channels=128, out_channels=256, padding=1, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, padding=1, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # vgg block 4
            nn.Conv2d(in_channels=256, out_channels=512, padding=1, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # vgg block 5
            nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        output_size = (int(img_size[0] / 32.) ** 2) * 512
        self.classifier = nn.Sequential(
            nn.Linear(in_features=output_size, out_features=self.num_hidden),
            nn.Linear(in_features=self.num_hidden, out_features=self.num_hidden),
            nn.Linear(in_features=self.num_hidden, out_features=self.num_output)
        )
        # define loss function
        self.loss = nn.CrossEntropyLoss()
        # initialize weights
        self._init_weights()

    def forward(self, batch):
        x = batch['input']
        y = batch['label'].squeeze(1)
        features = self.features(x)
        output = self.classifier(features.view(x.size(0), -1))
        loss = self.loss(output, y)

        return loss, output
    
    def _init_weights(self):
        # code snippet for initializing pytorch network weights
        # taken from: https://github.com/tonylins/pytorch-mobilenet-v2/blob/master/MobileNetV2.py
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


# vgg class for counting parameters in ptflops
class VGG11_test(VGG11):
    def __init__(self, num_hidden=4096, num_output=11, img_size=(256, 256)):
        super(VGG11_test, self).__init__(num_hidden, num_output, img_size)
    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features.view(x.size(0), -1))

        return output
