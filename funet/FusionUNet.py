# -*- coding=utf-8 -*-
# @Time: 2023.5.12
# @Author: Zongyi Li
# funet.py


from .utils import ConvBatchNorm, DownBlock, UpBlock_attention

import torch
import torch.nn as nn


class ECA(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class FuseBlock(nn.Module):
    def __init__(self, base_channels):
        super(FuseBlock, self).__init__() 
        self.norm1 = nn.BatchNorm2d(base_channels)
        self.norm2 = nn.BatchNorm2d(base_channels * 2)
        self.norm3 = nn.BatchNorm2d(base_channels * 4)
        self.norm4 = nn.BatchNorm2d(base_channels * 8)


        self.up3 = UpFuseBlock(base_channels=base_channels * 4)
        self.up2 = UpFuseBlock(base_channels=base_channels * 2)
        self.up1 = UpFuseBlock(base_channels=base_channels)

        self.down1 = DownFuseBlock(base_channels=base_channels)
        self.down2 = DownFuseBlock(base_channels=base_channels * 2)
        self.down3 = DownFuseBlock(base_channels=base_channels * 4)

    def forward(self, fp1, fp2, fp3, fp4):
        """

        Args:
            fp1 (torch.Tensor): (B, C, H, W)
            fp2 : (B, C * 2, H // 2, W // 2)
            fp3: (B, C * 4, H // 4, W // 4)
            fp4: (B, C * 8, H //8, W // 8)

        """
        fp4 = self.norm4(fp4)
        fp3 = self.norm3(fp3)
        fp2 = self.norm2(fp2)
        fp1 = self.norm1(fp1)
 
        
        # downsample fuse phase
        fp2 = self.down1(fp1, fp2)
        fp3 = self.down2(fp2, fp3)
        fp4 = self.down3(fp3, fp4)

        # upsample fuse phase
        fp3 = self.up3(fp3, fp4)
        fp2 = self.up2(fp2, fp3)
        fp1 = self.up1(fp1, fp2)


        return fp1, fp2, fp3, fp4



def reshape_downsample(x):
    '''using reshape method to do downsample
       
    Args:
        -x (torch.Tensor): (B, C, H, W)
    Return
        -ret (torch.Tensor): (B, C * 4, H // 2, W // 2)  
    '''
    b, c, h, w = x.shape
    ret = torch.zeros_like(x)
    ret = ret.reshape(b, c * 4, h // 2, -1)
    ret[:, 0::4, :, :] = x[:, :, 0::2, 0::2]
    ret[:, 1::4, :, :] = x[:, :, 0::2, 1::2]
    ret[:, 2::4, :, :] = x[:, :, 1::2, 0::2]
    ret[:, 3::4, :, :] = x[:, :, 1::2, 1::2]

    return ret

def reshape_upsample(x):
    '''using reshape to do upsample
    '''
    b, c, h, w = x.shape
    assert c % 4 == 0, 'number of channels must be multiple of 4'
    ret = torch.zeros_like(x) 
    ret = ret.reshape(b, c // 4, h * 2, w * 2)
    ret[:, :, 0::2, 0::2] = x[:, 0::4, :, :]
    ret[:, :, 0::2, 1::2] = x[:, 1::4, :, :]
    ret[:, :, 1::2, 0::2] = x[:, 2::4, :, :]
    ret[:, :, 1::2, 1::2] = x[:, 3::4, :, :]

    return ret


class DownFuseBlock(nn.Module):
    def __init__(self, base_channels, dropout_rate=0.1):
        super(DownFuseBlock, self).__init__()
        self.eca = ECA(base_channels * 2)
        self.down = reshape_downsample

        # we use group conv here since the reshape downsample split original feature map into
        # 4 pieces and group them in channel dimention. We want each conv group to have 4 channels
        # whicn contains exactly all informations in original HxW feature map
        self.conv1 = nn.Conv2d(base_channels * 4, base_channels * 2, 3, 1, 1, groups=base_channels) 
        self.norm1 = nn.BatchNorm2d(base_channels * 2)

        self.fuse_conv = ConvBatchNorm(base_channels * 2, base_channels * 2)
        self.relu = nn.ReLU()

    def forward(self, fp1, fp2):
        '''
            Args:
                -fp1: (B, C1, H1, W1)
                -fp2: (B, C1 * 2, H1 //2, W1 // 2)
        '''
        down = self.down(fp1)
        down = self.conv1(down)
        down = self.relu(self.norm1(down))

        fp2 = self.fuse_conv(fp2 *0.75 + down * 0.25) + fp2
        fp2 = self.eca(fp2)

        return fp2
        
    
class UpFuseBlock(nn.Module):
    def __init__(self, base_channels, dropout_rate=0.1):
        super(UpFuseBlock, self).__init__()
        self.eca = ECA(base_channels)
        self.up = reshape_upsample

        self.conv1 = nn.Conv2d(base_channels // 2, base_channels, kernel_size=3, stride=1, padding=1, groups=base_channels//2)
        self.norm1 = nn.BatchNorm2d(base_channels)

        self.relu = nn.ReLU()
        self.fuse_conv = ConvBatchNorm(base_channels, base_channels)
        
    def forward(self, fp1, fp2):
        '''
            Args:
                -fp1: (B, C1, H1, W1)
                -fp2: (B, C1 * 2, H1 //2, W1 // 2)
        '''
        up = self.up(fp2)
        up = self.conv1(up) 
        up = self.relu(self.norm1(up))

        fp1 = self.fuse_conv((fp1 * 0.75 + up * 0.25)) + fp1
        fp1 = self.eca(fp1)

        return fp1
        

class FuseModule(nn.Module):
    def __init__(self, base_channel, nb_blocks: int=2):
        super(FuseModule, self).__init__()
        self.base_channel = base_channel
        self.blocks = nn.ModuleList()
        nb_blocks = max(1, nb_blocks)
        for _ in range(nb_blocks):
            self.blocks.append(FuseBlock(base_channel))

    def forward(self, fp1, fp2, fp3, fp4):
        '''
            Args:
                - fp1: torch.Tenosr (B, h1, w1, c1), first level feature map
                - fp2: torch.Tensor (B, h1 // 2, w1 // 2, c1 * 2)
                - fp3 and fp4: like above, shape in (B, h1 // 4, w1 // 4, c1 * 4) and 
                                                    (B, h1 //8, w1 // 8, c1 * 8) respectively
        '''
        for block in self.blocks:
            fp1, fp2, fp3, fp4 = block(fp1, fp2, fp3, fp4)

        return fp1, fp2, fp3, fp4


class FusionUNet(nn.Module):
    def __init__(self, in_channels, n_cls, base_channels, aggre_depth=2):
        '''
            Args:
            - in_channels: int, should be the number of channel of input image
            - n_cls: int, number of segment classes
            - base_channels: int, number of first output skip connection channels
            - aggre_depth: int, number of blocks used to fuse feature map of different scope
        '''
        super(FusionUNet, self).__init__()
        self.in_channel = in_channels
        self.n_cls = n_cls
        
        self.inc = ConvBatchNorm(in_channels, base_channels)
        self.down1 = DownBlock(base_channels, base_channels * 2, nb_Conv=2)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, nb_Conv=2)
        self.down3 = DownBlock(base_channels * 4, base_channels * 8, nb_Conv=2)
        self.down4 = DownBlock(base_channels * 8, base_channels * 8, nb_Conv=2)

        self.fuse = FuseModule(base_channel=base_channels, nb_blocks=aggre_depth)

        self.up4 = UpBlock_attention(base_channels * 16, base_channels * 4, nb_Conv=2)
        self.up3 = UpBlock_attention(base_channels * 8, base_channels * 2, nb_Conv=2)
        self.up2 = UpBlock_attention(base_channels * 4, base_channels, nb_Conv=2)
        self.up1 = UpBlock_attention(base_channels * 2, base_channels, nb_Conv=2)

        self.outc = nn.Conv2d(base_channels, n_cls, kernel_size=1, stride=1)
        self.last_activation = nn.Sigmoid()

    def forward(self, x):
        x = x.float()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x1, x2, x3, x4 = self.fuse(x1, x2, x3, x4)
        
        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        
        if self.n_cls == 1:
            logits = self.last_activation(self.outc(x))
        else:
            logits = self.outc(x)
        return logits