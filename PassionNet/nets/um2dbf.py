import torch
import torch.nn as nn

from nets.resnet import resnet50
from nets.vgg import VGG16
import torch.nn.functional as F

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







class CNN1(nn.Module):
    def __init__(self,channel,map_size,pad):
        super(CNN1,self).__init__()
        self.weight = nn.Parameter(torch.ones(channel,channel,map_size,map_size),requires_grad=False).cuda()
        self.bias = nn.Parameter(torch.zeros(channel),requires_grad=False).cuda()
        self.pad = pad
        self.norm = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU()

    def forward(self,x):
        out = F.conv2d(x,self.weight,self.bias,stride=1,padding=self.pad)
        out = self.norm(out)
        out = self.relu(out)
        return out

class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs


class Unet(nn.Module):
    def __init__(self, num_classes=21, pretrained=False, backbone='vgg'):
        super(Unet, self).__init__()
        if backbone == 'vgg':
            self.vgg = VGG16(pretrained=pretrained)
            in_filters = [192, 384, 768, 1024]
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained=pretrained)
            in_filters = [192, 512, 1024, 3072]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]

        out_filters = [64, 128, 256, 512]

        num_iter, dilations = get_numiter_dilations('train')
       
        key_channels = 128
        self.FBF_layer1 = FBF_Layer(64, 64, num_iter, dilations[0])
        self.FBF_layer2 = FBF_Layer(64, 64, num_iter, dilations[1])
        self.FBF_layer3 = FBF_Layer(64, 64, num_iter, dilations[2])
        #self.FBF_layer4 = FBF_Layer(64, 64, num_iter, dilations[3])
        self.FBF_layer4 = FBF_Layer(512, 512, num_iter, dilations[3])
        self.FBF_layer = FBF_Layer(key_channels, key_channels, 1, dilations=[1])






        self.cls = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, num_classes - 1, 1),
            nn.AdaptiveMaxPool2d(1),
            nn.Sigmoid())

        self.conv_3 = CNN1(64, 3, 1)
        self.conv_5 = CNN1(64, 5, 2)

        self.x5_dem_1 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.x4_dem_1 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.x3_dem_1 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.x2_dem_1 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))


        
        self.x5_x4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))
        self.x4_x3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))
        self.x3_x2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))
        self.x2_x1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))

        self.x5_x4_x3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.x4_x3_x2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.x3_x2_x1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))

        self.x5_x4_x3_x2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                         nn.ReLU(inplace=True))
        self.x4_x3_x2_x1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                         nn.ReLU(inplace=True))
        self.x5_dem_4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.x5_x4_x3_x2_x1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                            nn.ReLU(inplace=True))

        self.level3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.level2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.level1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.x5_dem_5 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.output4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True))
        self.output3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True))
        self.output2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True))
        self.output1 = nn.Sequential(nn.Conv2d(64,num_classes, kernel_size=3, padding=1))

        self.last_conv = nn.Sequential(
            nn.Conv2d(196, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            )

        self.seg_decoder = nn.Sequential(
            nn.Conv2d(key_channels, key_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(key_channels),
            nn.ReLU(),
            nn.Conv2d(key_channels, num_classes, kernel_size=3, stride=1, padding=1),
        )

        self.seg_decoder2 = nn.Sequential(
            nn.Conv2d(key_channels, key_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(key_channels),
            nn.ReLU(),
            nn.Conv2d(key_channels, num_classes, kernel_size=3, stride=1, padding=1),
        )


        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone
    def upsample_cat(self, p1, p2, p3, p4):
        p2 = nn.functional.interpolate(p2, size=p1.size()[2:], mode='bilinear', align_corners=True)
        p3 = nn.functional.interpolate(p3, size=p1.size()[2:], mode='bilinear', align_corners=True)
        p4 = nn.functional.interpolate(p4, size=p1.size()[2:], mode='bilinear', align_corners=True)
        return torch.cat([p1, p2, p3, p4], dim=1)

    def forward(self, inputs):
        if self.backbone == "vgg":
            [x1, x2, x3, x4, x5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [x1, x2, x3, x4, x5] = self.resnet.forward(inputs)


        cls_branch = self.cls(x5).squeeze()

        x5 = self.FBF_layer4(x5)

        x5_dem_1 = self.x5_dem_1(x5)
        x4_dem_1 = self.x4_dem_1(x4)
        x3_dem_1 = self.x3_dem_1(x3)
        x2_dem_1 = self.x2_dem_1(x2)

        x5_dem_1_up = F.upsample(x5_dem_1, size=x4.size()[2:], mode='bilinear')
        x5_dem_1_up_map1 = self.conv_3(x5_dem_1_up)
        x4_dem_1_map1 = self.conv_3(x4_dem_1)
        x5_dem_1_up_map2 = self.conv_5(x5_dem_1_up)
        x4_dem_1_map2 = self.conv_5(x4_dem_1)
        x5_4 = self.x5_x4(
            abs(x5_dem_1_up - x4_dem_1) + abs(x5_dem_1_up_map1 - x4_dem_1_map1) + abs(x5_dem_1_up_map2 - x4_dem_1_map2))

        x4_dem_1_up = F.upsample(x4_dem_1, size=x3.size()[2:], mode='bilinear')
        x4_dem_1_up_map1 = self.conv_3(x4_dem_1_up)
        x3_dem_1_map1 = self.conv_3(x3_dem_1)
        x4_dem_1_up_map2 = self.conv_5(x4_dem_1_up)
        x3_dem_1_map2 = self.conv_5(x3_dem_1)
        x4_3 = self.x4_x3(
            abs(x4_dem_1_up - x3_dem_1) + abs(x4_dem_1_up_map1 - x3_dem_1_map1) + abs(x4_dem_1_up_map2 - x3_dem_1_map2))

        x3_dem_1_up = F.upsample(x3_dem_1, size=x2.size()[2:], mode='bilinear')
        x3_dem_1_up_map1 = self.conv_3(x3_dem_1_up)
        x2_dem_1_map1 = self.conv_3(x2_dem_1)
        x3_dem_1_up_map2 = self.conv_5(x3_dem_1_up)
        x2_dem_1_map2 = self.conv_5(x2_dem_1)
        x3_2 = self.x3_x2(
            abs(x3_dem_1_up - x2_dem_1) + abs(x3_dem_1_up_map1 - x2_dem_1_map1) + abs(x3_dem_1_up_map2 - x2_dem_1_map2))

        x2_dem_1_up = F.upsample(x2_dem_1, size=x1.size()[2:], mode='bilinear')
        x2_dem_1_up_map1 = self.conv_3(x2_dem_1_up)
        x1_map1 = self.conv_3(x1)
        x2_dem_1_up_map2 = self.conv_5(x2_dem_1_up)
        x1_map2 = self.conv_5(x1)
        x2_1 = self.x2_x1(abs(x2_dem_1_up - x1) + abs(x2_dem_1_up_map1 - x1_map1) + abs(x2_dem_1_up_map2 - x1_map2))

        x5_4_up = F.upsample(x5_4, size=x4_3.size()[2:], mode='bilinear')
        x5_4_up_map1 = self.conv_3(x5_4_up)
        x4_3_map1 = self.conv_3(x4_3)
        x5_4_up_map2 = self.conv_5(x5_4_up)
        x4_3_map2 = self.conv_5(x4_3)
        x5_4_3 = self.x5_x4_x3(abs(x5_4_up - x4_3) + abs(x5_4_up_map1 - x4_3_map1) + abs(x5_4_up_map2 - x4_3_map2))

        x4_3_up = F.upsample(x4_3, size=x3_2.size()[2:], mode='bilinear')
        x4_3_up_map1 = self.conv_3(x4_3_up)
        x3_2_map1 = self.conv_3(x3_2)
        x4_3_up_map2 = self.conv_5(x4_3_up)
        x3_2_map2 = self.conv_5(x3_2)
        x4_3_2 = self.x4_x3_x2(abs(x4_3_up - x3_2) + abs(x4_3_up_map1 - x3_2_map1) + abs(x4_3_up_map2 - x3_2_map2))

        x3_2_up = F.upsample(x3_2, size=x2_1.size()[2:], mode='bilinear')
        x3_2_up_map1 = self.conv_3(x3_2_up)
        x2_1_map1 = self.conv_3(x2_1)
        x3_2_up_map2 = self.conv_5(x3_2_up)
        x2_1_map2 = self.conv_5(x2_1)
        x3_2_1 = self.x3_x2_x1(abs(x3_2_up - x2_1) + abs(x3_2_up_map1 - x2_1_map1) + abs(x3_2_up_map2 - x2_1_map2))

        x5_4_3_up = F.upsample(x5_4_3, size=x4_3_2.size()[2:], mode='bilinear')
        x5_4_3_up_map1 = self.conv_3(x5_4_3_up)
        x4_3_2_map1 = self.conv_3(x4_3_2)
        x5_4_3_up_map2 = self.conv_5(x5_4_3_up)
        x4_3_2_map2 = self.conv_5(x4_3_2)
        x5_4_3_2 = self.x5_x4_x3_x2(
            abs(x5_4_3_up - x4_3_2) + abs(x5_4_3_up_map1 - x4_3_2_map1) + abs(x5_4_3_up_map2 - x4_3_2_map2))

        x4_3_2_up = F.upsample(x4_3_2, size=x3_2_1.size()[2:], mode='bilinear')
        x4_3_2_up_map1 = self.conv_3(x4_3_2_up)
        x3_2_1_map1 = self.conv_3(x3_2_1)
        x4_3_2_up_map2 = self.conv_5(x4_3_2_up)
        x3_2_1_map2 = self.conv_5(x3_2_1)
        x4_3_2_1 = self.x4_x3_x2_x1(
            abs(x4_3_2_up - x3_2_1) + abs(x4_3_2_up_map1 - x3_2_1_map1) + abs(x4_3_2_up_map2 - x3_2_1_map2))

        x5_dem_4 = self.x5_dem_4(x5_4_3_2)
        x5_dem_4_up = F.upsample(x5_dem_4, size=x4_3_2_1.size()[2:], mode='bilinear')
        x5_dem_4_up_map1 = self.conv_3(x5_dem_4_up)
        x4_3_2_1_map1 = self.conv_3(x4_3_2_1)
        x5_dem_4_up_map2 = self.conv_5(x5_dem_4_up)
        x4_3_2_1_map2 = self.conv_5(x4_3_2_1)
        x5_4_3_2_1 = self.x5_x4_x3_x2_x1(
            abs(x5_dem_4_up - x4_3_2_1) + abs(x5_dem_4_up_map1 - x4_3_2_1_map1) + abs(x5_dem_4_up_map2 - x4_3_2_1_map2))

        level4 = x5_4
        level3 = self.level3(x4_3 + x5_4_3)
        level2 = self.level2(x3_2 + x4_3_2 + x5_4_3_2)
        level1 = self.level1(x2_1 + x3_2_1 + x4_3_2_1 + x5_4_3_2_1)
        

        x5_dem_5 = self.x5_dem_5(x5)

        # y=self.FBF_layer4(level1)
        # m1 = self.FBF_layer4(level1)
        # m2 = self.FBF_layer3(level2)
        # m3 = self.FBF_layer2(level3)
        # m4 = self.FBF_layer1(level4)
        # m5 = self.FBF_layer1(x5_dem_5)




        output4 = self.output4(F.upsample(x5_dem_5, size=level4.size()[2:], mode='bilinear') + level4)
        output3 = self.output3(F.upsample(output4, size=level3.size()[2:], mode='bilinear') + level3)
        output2 = self.output2(F.upsample(output3, size=level2.size()[2:], mode='bilinear') + level2)
        output1 = self.output1(F.upsample(output2, size=level1.size()[2:], mode='bilinear') + level1)

        cat = self.upsample_cat(output1, output2, output3, output4)
        
        
        feat = self.last_conv(cat)
        #feat = self.FBF_layer(feat)

        # if self.up_conv != None:
        #     up1 = self.up_conv(up1)

        out = self.seg_decoder(feat)
        out2 = self.seg_decoder2(feat)






        
        #output = F.upsample(output1, size=inputs.size()[2:], mode='bilinear')
        
        #final = self.final(output)


        return out,out2

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
