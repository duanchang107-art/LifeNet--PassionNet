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
# def initialize_weights(modules, init_mode):
#     for m in modules:
#         if isinstance(m, nn.Conv2d):
#             if init_mode == 'he':
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out',
#                                         nonlinearity='relu')
#             elif init_mode == 'xavier':
#                 nn.init.xavier_uniform_(m.weight.data)
#             elif init_mode == 'contant':
#                 #print(m)
#                 nn.init.constant_(m.weight, 0)
#             else:
#                 raise ValueError('Invalid init_mode {}'.format(init_mode))
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.BatchNorm2d):
#             nn.init.constant_(m.weight, 1)
#             nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.Linear):
#             nn.init.normal_(m.weight, 0, 0.01)
#             nn.init.constant_(m.bias, 0)

# class LoRM(nn.Module):
#     def __init__(self, in_dims=2048, out_dims=2048):
#         super(LoRM, self).__init__()
#         self.in_dims = in_dims
#         self.out_dims = out_dims if out_dims != None else in_dims
#         self.query_conv = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1)
#         self.key_conv = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1)
#         self.distance_func = self.mse_loss
#         self.gamma = nn.Parameter(torch.ones(1))
#         initialize_weights(self.modules(), init_mode='xavier')

#     def forward(self, x, cam, mode='cam'):  # cam: b,h,w 0,1,2... #log/Jun07_13-34-40_ubuntu 有用！
#         cam[cam == 255] = 0
#         new_x = torch.zeros_like(x)  # b,c,h,w
#         x_shape = x.size()[2:4]  # x: b,c,h,w
#         B, C, H, W = x.size()
#         cam = cam.unsqueeze(1).float()  # b,h,w -> b,1,h,w
#         cam = F.interpolate(cam, size=x_shape, mode='bilinear', align_corners=True)  # scale to features map's shape
#         # all_cls = torch.unique(cam)
#         # for clsid in all_cls:
#         #     if clsid == 255:
#         #         continue
#         curr_mask = torch.zeros_like(cam)
#         curr_mask[cam > 0] = 1  # b,1,h,w
#         curr_mask = curr_mask.view(B, -1, H * W).permute(0, 2, 1)  # b,1,h,w -> b,1,hw -> b,hw,1
#         curr_query = self.query_conv(x)
#         # curr_query = query*curr_mask # b,2048,h,w
#         curr_query = curr_query.view(B, -1, H * W).permute(0, 2, 1)  # b,c,h,w -> b,c,hw -> b,hw,c
#         curr_key = self.key_conv(x)  # b,2048,h,w
#         curr_key = curr_key.view(B, -1, H * W)  # b,c,h,w -> b,c,hw
#         energy = torch.bmm(curr_query, curr_key)  # b,hw(query),hw(key)
#         curr_query_norm = curr_query.pow(2).sum(-1, keepdim=True).sqrt()  # b,hw,c -> b,hw,1
#         curr_key_norm = curr_key.pow(2).sum(1, keepdim=True).sqrt()  # b,c,hw -> b,1,hw
#         curr_norm = torch.bmm(curr_query_norm, curr_key_norm)  # b,hw(qurey),hw(key)
#         atten = energy / curr_norm
#         atten = torch.softmax(atten, dim=-1)  # b,hw(query),hw(key,softmaxed)
#         atten = curr_mask * atten  # b,hw(query,masked),hw(key,softmaxed)
#         x_tmp = x.view(B, -1, H * W).permute(0, 2, 1)  # b,c,h,w -> b,c,hw -> b,hw,c
#         new_x = new_x + torch.bmm(atten, x_tmp).permute(0, 2, 1).view(B, C, H,
#                                                                       W)  # b,hw(query,masked),c(weighted channel) -> b,c,hw -> b,c,h,w
#         new_x = self.gamma * new_x  # ones
#         x_tmp = x_tmp.permute(0, 2, 1).view(B, -1, H, W)  # b,hw,c -> b,c,hw -> b,c,h,w
#         loss = self.distance_func(x, new_x)
#         return loss

#     def mse_loss(self, x1, x2):
#         result = (x1 - x2) ** 2
#         loss = torch.sum(result) / torch.count_nonzero(result)
#         return loss

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
        #   sup4 = self.supervision4(up4)  # Supervision at decoder4
        #   sup3 = self.supervision3(up3)  # Supervision at decoder3
        #   sup2 = self.supervision2(up2)  # Supervision at decoder2
        #     # print(sup4.shape,sup3.shape,sup2.shape,final.shape)
        # return final,final2,sup4, sup3, sup2

        
        
        return final,final2

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
