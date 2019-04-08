import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNormAct(nn.Sequential):
    def __init__(self, in_c, out_c, norm, act, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(ConvNormAct, self).__init__()
        self.add_module('conv', nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False))
        self.add_module('norm', norm(out_c, affine=True))
        self.add_module('act', act(inplace=True))

class ConvBlock(nn.Sequential):
    def __init__(self, in_c, out_c, norm, act):
        super(ConvBlock, self).__init__()
        self.add_module('conv_layer_1', ConvNormAct(in_c, out_c, norm, act))
        self.add_module('conv_layer_2', ConvNormAct(out_c, out_c, norm, act))

class GcnBlock(nn.Module):
    def __init__(self, in_c, out_c, norm, act, ks=7):
        super(GcnBlock, self).__init__()
        self.conv_block_l = nn.Sequential(ConvNormAct(in_c, out_c, norm, act, (ks, 1), padding=(ks // 2, 0)),
                                          ConvNormAct(out_c, out_c, norm, act, (1, ks), padding=(0, ks // 2)))
        self.conv_block_r = nn.Sequential(ConvNormAct(in_c, out_c, norm, act, (1, ks), padding=(ks // 2, 0)),
                                          ConvNormAct(out_c, out_c, norm, act, (ks, 1), padding=(0, ks // 2)))

    def forward(self, x):
        gcn_block_l = self.conv_block_l(x)
        gcn_block_r = self.conv_block_r(x)

        return gcn_block_l + gcn_block_r

class DownModule(nn.Module):
    def __init__(self, in_c, out_c, norm, act):
        super(DownModule, self).__init__()
        self.conv_block = ConvBlock(in_c, out_c, norm, act)
        self.gcn_block = GcnBlock(out_c, out_c, norm, act)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        conv_block = self.conv_block(x)
        gcn_block = self.gcn_block(conv_block)
        pool = self.pool(gcn_block)
        ssc = conv_block + gcn_block

        return ssc, pool

class UpModule(nn.Module):
    def __init__(self, in_c, out_c, norm, act):
        super(UpModule, self).__init__()
        self.up_layer = nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv_block = ConvBlock(out_c * 2, out_c, norm, act)

    def forward(self, x1, x2):
        up = self.up_layer(x2)
        offset = up.shape[2] - x1.shape[2]
        padding = [offset // 2] * 4
        lsc = F.pad(x1, padding)
        up_block = torch.cat([lsc, up], 1)

        return self.conv_block(up_block)

class UnetGcnSsc(nn.Module):
    def __init__(self, feature_scale, n_classes, norm, act):
        super(UnetGcnSsc, self).__init__()
        filters = [64, 128, 256, 512, 1024]
        filters = [x // feature_scale for x in filters]

        self.down_module_1 = DownModule(1, filters[0], norm, act)
        self.down_module_2 = DownModule(filters[0], filters[1], norm, act)
        self.down_module_3 = DownModule(filters[1], filters[2], norm, act)
        self.down_module_4 = DownModule(filters[2], filters[3], norm, act)
        self.bridge_block = ConvBlock(filters[3], filters[4], norm, act)

        self.up_module_4 = UpModule(filters[4], filters[3], norm, act)
        self.up_module_3 = UpModule(filters[3], filters[2], norm, act)
        self.up_module_2 = UpModule(filters[2], filters[1], norm, act)
        self.up_module_1 = UpModule(filters[1], filters[0], norm, act)

        self.final_conv = nn.Conv2d(filters[0], n_classes, 1)

    def forward(self, x):
        lsc1, down1 = self.down_module_1(x)
        lsc2, down2 = self.down_module_2(down1)
        lsc3, down3 = self.down_module_3(down2)
        lsc4, down4 = self.down_module_4(down3)
        bridge = self.bridge_block(down4)

        up4 = self.up_module_4(lsc4, bridge)
        up3 = self.up_module_3(lsc3, up4)
        up2 = self.up_module_2(lsc2, up3)
        up1 = self.up_module_1(lsc1, up2)

        final = self.final_conv(up1)
        return final
