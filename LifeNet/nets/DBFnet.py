import torch
import torch.nn as nn

from nets.resnet import resnet50
from nets.vgg import VGG16
from nets.fbf import FBFModule

class FBF_Layer(nn.Module):
    def __init__(self, in_c, out_c, num_iter, dilations):
        super(FBF_Layer, self).__init__()
        self.num_iter = num_iter
        self.fbf_m = FBFModule(num_iter=num_iter, dilations=dilations)
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_c),
            nn.ReLU(), )

    def forward(self, x):
        # x = self.conv0(x)
        x = self.fbf_m(x)
        x = self.conv(x)

        return x

def get_numiter_dilations(flag):
    if flag == 'train':

        # for potsdam
        num_iter = 1
        dilations = [[1, 3, 5, 7], [1, 3, 5], [1, 3], [1]]

    else:
        # for potsdam
        num_iter = 3
        dilations = [[1], [1], [1], [1]]

        # for vaihingen
        # num_iter = 5
        # dilations = [[1], [1], [1], [1]]
    return num_iter, dilations

class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.relu   = nn.ReLU(inplace = True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs

class BFnet(nn.Module):
    def __init__(self, num_classes = 21, pretrained = False, backbone = 'vgg'):
        super(BFnet, self).__init__()
        if backbone == 'vgg':
            self.vgg    = VGG16(pretrained = pretrained)
            in_filters  = [192, 384, 768, 1024]
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained = pretrained)
            in_filters  = [192, 512, 1024, 3072]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]

        num_iter, dilations = get_numiter_dilations('train')
       
        key_channels = 128
        self.FBF_layer1 = FBF_Layer(64, 64, num_iter, dilations[0])
        self.FBF_layer2 = FBF_Layer(128, 128, num_iter, dilations[1])
        self.FBF_layer3 = FBF_Layer(256, 256, num_iter, dilations[2])
        self.FBF_layer4 = FBF_Layer(512, 512, num_iter, dilations[3])
        self.FBF_layer = FBF_Layer(key_channels, key_channels, 1, dilations=[1])

        

        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        self.last_conv = nn.Sequential(
            nn.Conv2d(960, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            )

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor = 2), 
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone

        self.seg_decoder = nn.Sequential(
            nn.Conv2d(key_channels, key_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(key_channels),
            nn.ReLU(),
            nn.Conv2d(key_channels, num_classes, kernel_size=3, stride=1, padding=1),
        )


    # def resize_out(self, output):
    #     if output.size()[-1] != self.img_size:
    #         output = F.interpolate(output, size=(self.img_size, self.img_size), mode='bilinear')
    #     return output

    def upsample_cat(self, p1, p2, p3, p4):
        p2 = nn.functional.interpolate(p2, size=p1.size()[2:], mode='bilinear', align_corners=True)
        p3 = nn.functional.interpolate(p3, size=p1.size()[2:], mode='bilinear', align_corners=True)
        p4 = nn.functional.interpolate(p4, size=p1.size()[2:], mode='bilinear', align_corners=True)
        return torch.cat([p1, p2, p3, p4], dim=1)

    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        
        feat1 = self.FBF_layer1(feat1)
        
        feat2 = self.FBF_layer2(feat2)
        feat3 = self.FBF_layer3(feat3)
        feat4 = self.FBF_layer4(feat4)
        #feat5=
        up4 = feat4

        #up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)
        

        cat = self.upsample_cat(up1, up2, up3, up4)
        
        feat = self.last_conv(cat)
        feat = self.FBF_layer(feat)

        if self.up_conv != None:
            up1 = self.up_conv(up1)

        out = self.seg_decoder(feat)
        #out = self.resize_out(out)

        #final = self.final(up1)
        
        return out

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True
