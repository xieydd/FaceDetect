import time
import torch
import torch.nn as nn
import torchvision.models._utils as _utils
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
from models.efficientnet import EfficientNet

def conv_bn(inp, oup, stride = 1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )

def conv_bn1X1(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

class CBAModule(nn.Module):
    def __init__(self, in_channels, out_channels=24, kernel_size=3,stride=1,padding=0,bias=False):
        super(CBAModule,self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,kernel_size,stride,padding=padding,bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class ContextModule(nn.Module):
    def __init__(self, in_channels):
        super(ContextModule, self).__init__()
        block_wide = in_channels // 4
        self.inconv = CBAModule(in_channels,block_wide,3,1,padding=1)
        self.upconv = CBAModule(block_wide,block_wide,3,1,padding=1)
        self.downconv = CBAModule(block_wide,block_wide,3,1,padding=1)
        self.downconv2 = CBAModule(block_wide,block_wide,3,1,padding=1)
    
    def forward(self,x):
        x = self.inconv(x)
        up = self.upconv(x)
        down = self.downconv(x)
        down = self.downconv2(down)
        return torch.cat([up, down],dim=1)

class DetectMoule(nn.Module):
    def __init__(self,in_channels):
        super(DetectMoule, self).__init__()
        self.upconv = CBAModule(in_channels, in_channels//2,3,1,padding=1)
        self.context = ContextModule(in_channels)
    def forward(self,x):
        up = self.upconv(x)
        down = self.context(x)
        return torch.cat([up, down],dim=1)


class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if (out_channel <= 64):
            leaky = 0.1
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel//2, stride=1)

        self.conv5X5_1 = conv_bn(in_channel, out_channel//4, stride=1)
        self.conv5X5_2 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

        self.conv7X7_2 = conv_bn(out_channel//4, out_channel//4, stride=1)
        self.conv7x7_3 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out

class FPN(nn.Module):
    def __init__(self,in_channels_list,out_channels):
        super(FPN,self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1
        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride = 1)
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride = 1)
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride = 1)
        self.output4 = conv_bn1X1(in_channels_list[3], out_channels, stride = 1)

        self.merge1 = conv_bn(out_channels, out_channels)
        self.merge2 = conv_bn(out_channels, out_channels)
        self.merge3 = conv_bn(out_channels, out_channels)

    def forward(self, input):
        # names = list(input.keys())
        #input = list(input.values())

        output1 = self.output1(input[0])
        output2 = self.output2(input[1])
        output3 = self.output3(input[2])
        output4 = self.output4(input[3])

        # up4 = F.interpolate(output4, scale_factor = 2, mode="nearest")
        up4 = F.interpolate(output4, size=[output3.size(2), output3.size(3)], mode="nearest")
        output3 = output3 + up4
        output3 = self.merge3(output3)

        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        out = [output1, output2, output3, output4]
        return out



class EfficientDet(nn.Module):
    def __init__(self, cfg = None, phase = 'train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(EfficientDet, self).__init__()
        self.phase = phase
        self.num_classes = 2
        self.base = EfficientNet.from_pretrained("efficientnet-b3")
        self.fpn_flag = True
        if self.fpn_flag:
            in_channels_list  = [48, 96,136,232]
            out_channels = 128
            self.ssh1 = DetectMoule(out_channels)
            self.ssh2 = DetectMoule(out_channels)
            self.ssh3 = DetectMoule(out_channels)
            self.ssh4 = DetectMoule(out_channels)
            '''
            self.ssh1 = SSH(out_channels,out_channels)
            self.ssh2 = SSH(out_channels,out_channels)
            self.ssh3 = SSH(out_channels,out_channels)
            self.ssh4 = SSH(out_channels,out_channels)
            '''
            self.fpn = FPN(in_channels_list,out_channels)
        self.loc, self.conf, self.landm = self.multibox(self.num_classes)
    def multibox(self, num_classes):
        anchor_num = [4,4,4,3]
        loc_layers = []
        conf_layers = []
        landm_layers = []
        if self.fpn_flag:
            loc_layers += [nn.Conv2d(128, anchor_num[0] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(128, anchor_num[0] * num_classes, kernel_size=3, padding=1)]
            landm_layers += [nn.Conv2d(128, anchor_num[0] * 10, kernel_size=3, padding=1)]

            loc_layers += [nn.Conv2d(128, anchor_num[1] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(128, anchor_num[1] * num_classes, kernel_size=3, padding=1)]
            landm_layers += [nn.Conv2d(128, anchor_num[1] * 10, kernel_size=3, padding=1)]

            loc_layers += [nn.Conv2d(128, anchor_num[2] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(128, anchor_num[2] * num_classes, kernel_size=3, padding=1)]
            landm_layers += [nn.Conv2d(128, anchor_num[2] * 10, kernel_size=3, padding=1)]

            loc_layers += [nn.Conv2d(128, anchor_num[3] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(128, anchor_num[3] * num_classes, kernel_size=3, padding=1)]
            landm_layers += [nn.Conv2d(128, anchor_num[3] * 10, kernel_size=3, padding=1)]
        else:
            loc_layers += [nn.Conv2d(48, anchor_num[0] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(48, anchor_num[0] * num_classes, kernel_size=3, padding=1)]
            landm_layers += [nn.Conv2d(48, anchor_num[0] * 10, kernel_size=3, padding=1)]

            loc_layers += [nn.Conv2d(96, anchor_num[1] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(96, anchor_num[1] * num_classes, kernel_size=3, padding=1)]
            landm_layers += [nn.Conv2d(96, anchor_num[1] * 10, kernel_size=3, padding=1)]

            loc_layers += [nn.Conv2d(136, anchor_num[2] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(136, anchor_num[2] * num_classes, kernel_size=3, padding=1)]
            landm_layers += [nn.Conv2d(136, anchor_num[2] * 10, kernel_size=3, padding=1)]

            loc_layers += [nn.Conv2d(232, anchor_num[3] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(232, anchor_num[3] * num_classes, kernel_size=3, padding=1)]
            landm_layers += [nn.Conv2d(232, anchor_num[3] * 10, kernel_size=3, padding=1)]
            # 56 112 160 272
        return nn.Sequential(*loc_layers), nn.Sequential(*conf_layers), nn.Sequential(*landm_layers)

    def forward(self, inputs):
        detections = list()
        loc = list()
        conf = list()
        landm = list()
        _,_,f1,f2,f3,f4,_ = self.base(inputs)
        if self.fpn_flag:
            fpn_out = self.fpn([f1,f2,f3,f4])
            f1 = self.ssh1(fpn_out[0])
            f2 = self.ssh1(fpn_out[1])
            f3 = self.ssh1(fpn_out[2])
            f4 = self.ssh1(fpn_out[3])
        detections = [f1,f2,f3,f4]
        for (x,l,c,lam) in zip(detections,self.loc,self.conf,self.landm):
            loc.append(l(x).permute(0,2,3,1).contiguous())
            conf.append(c(x).permute(0,2,3,1).contiguous())
            landm.append(lam(x).permute(0,2,3,1).contiguous())
        
        bbox_regressions = torch.cat([o.view(o.size(0), -1,4) for o in loc],1)
        classifications =  torch.cat([o.view(o.size(0), -1,2) for o in conf],1)
        ldm_regressions =  torch.cat([o.view(o.size(0), -1,10) for o in landm],1)
        #print(ldm_regressions.size())

        if self.phase == 'train':
            output = (bbox_regressions, classifications,ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications,dim=-1).ldm_regressions)

        return output

if __name__ == "__main__":
    x = torch.randn(2,3,320,320)
    net = EfficientDet('test')
    y = net(x)
    # import pdb
    # pdb.set_trace()
