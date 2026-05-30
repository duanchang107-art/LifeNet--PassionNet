import torch
import torch.nn as nn

from nets.resnet import resnet50
from nets.vgg import VGG16
import torch.nn.functional as F
import numpy as np
class Encoder(nn.Module):
    def __init__(self, input_channels, output_channels=256):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        B, C, H, W = x.shape
        x = x.view(B * H * W, C)  # 变形操作
        return x

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

class Unet(nn.Module):
    def __init__(self, num_classes = 21, pretrained = False, backbone = 'vgg'):
        super(Unet, self).__init__()
        if backbone == 'vgg':
            self.vgg    = VGG16(pretrained = pretrained)
            in_filters  = [192, 384, 768, 1024]
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained = pretrained)
            in_filters  = [192, 512, 1024, 3072]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]

        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(1024, out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])
        #self.layer_lorm = LoRM(in_dims=512, out_dims=512)

        self.cls = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, num_classes - 1, 1),
            nn.AdaptiveMaxPool2d(1),
            nn.Sigmoid())

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
        self.final2 = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone
        # self.deep_supervision = deep_supervision  # 添加这一行


        # if deep_supervision:
        self.supervision4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.supervision3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.supervision2 = nn.Conv2d(128, num_classes, kernel_size=1)


        self.class_encoder_w = Encoder(512, 256)  # 弱监督分类特征编码器
        self.seg_encoder_w = Encoder(64, 256)     # 弱监督分割特征编码器

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        cls_branch = self.cls(feat5).squeeze()

        # sda = self.SDA_block(feat5, 4, SDAblock_nb=5)
       
        # #center = self.conv_block(sda,512)
        # rmp = self.rmp_block(sda)


       

       
       
       

        up4 = self.up_concat4(feat4,feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        #att_loss = self.layer_lorm(feat5, cam)

        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1)
        final2 = self.final2(up1)
        
        # if self.deep_supervision:
        sup4 = self.supervision4(up4)  # Supervision at decoder4
        sup3 = self.supervision3(up3)  # Supervision at decoder3
        sup2 = self.supervision2(up2)  # Supervision at decoder2
        #     # print(sup4.shape,sup3.shape,sup2.shape,final.shape)
        #     return final,final2,sup4, sup3, sup2

      

         # 将分割特征缩放到分类特征的分辨率
        w_seg = F.interpolate(up1, size=(feat5.shape[2], feat5.shape[3]), mode='bilinear', align_corners=False)
        # f_seg = F.interpolate(f_seg, size=(f_cls.shape[2], f_cls.shape[3]), mode='bilinear', align_corners=False)


        w_cls = feat5  # [B, 1024, H/16, W/16]
        # w_seg = up1  # [B, 64, H/2, W/2]

        q = self.class_encoder_w(w_cls)
        v = self.seg_encoder_w(w_seg)
        
        return final,final2,sup4,sup3,sup2,q,v

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
