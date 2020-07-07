import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.models as models
import torch.backends.cudnn as cudnn
import os
from models.module import BasicRFB, Backbone, Backbone_, Backbone2, Backbone3
 
class CBAModule(nn.Module):
    def __init__(self, in_channels, out_channels=24, kernel_size=3, stride=1, padding=0, bias=False):
        super(CBAModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
 
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
 
 
class ContextModule(nn.Module):
    def __init__(self, in_channels):
        super(ContextModule, self).__init__()
        block_wide = in_channels // 4
        self.inconv = CBAModule(in_channels, block_wide, 3, 1, padding=1)
        self.upconv = CBAModule(block_wide, block_wide, 3, 1, padding=1)
        self.downconv = CBAModule(block_wide, block_wide, 3, 1, padding=1)
        self.downconv2 = CBAModule(block_wide, block_wide, 3, 1, padding=1)
 
    def forward(self, x):
        x= self.inconv(x)
        up = self.upconv(x)
        down = self.downconv(x)
        down = self.downconv2(down)
        return torch.cat([up, down], dim=1)
 
 
# SSH Detect Module
class DetectModule(nn.Module):
    def __init__(self, in_channels):
        super(DetectModule, self).__init__()
        self.upconv = CBAModule(in_channels, in_channels // 2, 3, 1, padding=1)
        self.context = ContextModule(in_channels)
 
    def forward(self, x):
        up = self.upconv(x)
        down = self.context(x)
        return torch.cat([up, down], dim=1)
 
 
 
 
class Light_VGG(nn.Module):
    def __init__(self, cfg = None, phase = 'train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(Light_VGG, self).__init__()
        self.phase = phase
        self.num_classes = 2
        self.base = Backbone_()
        out_channels = 128
        self.ssh1 = DetectModule(64)
        self.ssh2 = DetectModule(128)
        self.ssh3 = DetectModule(256)
        self.ssh4 = DetectModule(256)
 
 
        self.loc, self.conf, self.landm = self.multibox(self.num_classes);
 
 
    def multibox(self, num_classes):
        anchor_num = [4, 4, 4, 3] # number of boxes per feature map location
        loc_layers = []
        conf_layers = []
        landm_layers = []
 
 
        loc_layers  += [nn.Conv2d(64, anchor_num[0] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(64, anchor_num[0] * num_classes, kernel_size=3, padding=1)]
        landm_layers += [nn.Conv2d(64, anchor_num[0] * 10, kernel_size=3, padding=1)]
 
        loc_layers  += [nn.Conv2d(128, anchor_num[1] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(128, anchor_num[1] * num_classes, kernel_size=3, padding=1)]
        landm_layers += [nn.Conv2d(128, anchor_num[1] * 10, kernel_size=3, padding=1)]
 
 
        loc_layers  += [nn.Conv2d(256, anchor_num[2] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(256, anchor_num[2] * num_classes, kernel_size=3, padding=1)]
        landm_layers += [nn.Conv2d(256, anchor_num[2] * 10, kernel_size=3, padding=1)]
 
 
        loc_layers  += [nn.Conv2d(256, anchor_num[3] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(256, anchor_num[3] * num_classes, kernel_size=3, padding=1)]
        landm_layers += [nn.Conv2d(256, anchor_num[3]  * 10, kernel_size=3, padding=1)]
        '''
 
 
        ########### big
        loc_layers  += [nn.Conv2d(256, anchor_num[0] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(256, anchor_num[0] * num_classes, kernel_size=3, padding=1)]
        landm_layers += [nn.Conv2d(256, anchor_num[0] * 10, kernel_size=3, padding=1)]
 
        loc_layers  += [nn.Conv2d(256, anchor_num[1] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(256, anchor_num[1] * num_classes, kernel_size=3, padding=1)]
        landm_layers += [nn.Conv2d(256, anchor_num[1] * 10, kernel_size=3, padding=1)]
 
 
        loc_layers  += [nn.Conv2d(256, anchor_num[2] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(256, anchor_num[2] * num_classes, kernel_size=3, padding=1)]
        landm_layers += [nn.Conv2d(256, anchor_num[2] * 10, kernel_size=3, padding=1)]
 
 
        loc_layers  += [nn.Conv2d(256, anchor_num[3] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(256, anchor_num[3] * num_classes, kernel_size=3, padding=1)]
        landm_layers += [nn.Conv2d(256, anchor_num[3]  * 10, kernel_size=3, padding=1)]
        '''
 
        return nn.Sequential(*loc_layers), nn.Sequential(*conf_layers), nn.Sequential(*landm_layers)
 
 
 
    def forward(self,inputs):
        detections = list()
        loc = list()
        conf = list()
        landm = list()
 
        f1, f2, f3, f4 = self.base(inputs)
        f1 = self.ssh1(f1)
        f2 = self.ssh2(f2)
        f3 = self.ssh3(f3)
        f4 = self.ssh4(f4)
        detections = [f1, f2, f3, f4]
        for (x, l, c, lam) in zip(detections, self.loc, self.conf, self.landm):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
            landm.append(lam(x).permute(0, 2, 3, 1).contiguous())
 
        bbox_regressions = torch.cat([o.view(o.size(0), -1, 4) for o in loc], 1)
        classifications = torch.cat([o.view(o.size(0), -1, 2) for o in conf], 1)
        ldm_regressions = torch.cat([o.view(o.size(0), -1, 10) for o in landm], 1)
 
 
        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        return output
 
 
if __name__ == '__main__':
    # 0.99 MB
    x = torch.randn(2, 3, 300, 300)
    net = Light_VGG('test')
    from torchsummary import summary
    summary(net, (3, 300, 300))
