import torch.nn as nn
import torch.nn.functional as F
import torch
from numpy.random import normal
from math import sqrt
import argparse

#channel_dim = 2
#ndf = 32

class GlobalConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size):
        super(GlobalConvBlock, self).__init__()
        pad0 = (kernel_size[0] - 1) // 2
        pad1 = (kernel_size[1] - 1) // 2
        pad2 = (kernel_size[2] - 1) // 2

        self.conv_l1 = nn.Conv3d(in_dim, out_dim, kernel_size=(kernel_size[0], 1, 1),
                                 padding=(pad0, 0, 0))
        self.conv_l2 = nn.Conv3d(out_dim, out_dim, kernel_size=(1, kernel_size[1], 1),
                                 padding=(0, pad1, 0))
        self.conv_l3 = nn.Conv3d(out_dim, out_dim, kernel_size=(1, 1, kernel_size[2]),
                                 padding=(0, 0, pad2))
     
        self.conv_r1 = nn.Conv3d(in_dim, out_dim, kernel_size=(1, kernel_size[1], 1),
                                 padding=(0, pad1, 0))
        self.conv_r2 = nn.Conv3d(out_dim, out_dim, kernel_size=(kernel_size[0], 1, 1),
                                 padding=(pad0, 0, 0))
        self.conv_r3 = nn.Conv3d(out_dim, out_dim, kernel_size=(1, 1, kernel_size[2]),
                                 padding=(0, 0, pad2))

        self.conv_v1 = nn.Conv3d(in_dim, out_dim, kernel_size=(1, 1, kernel_size[2]),
                                 padding=(0, 0, pad2))
        self.conv_v2 = nn.Conv3d(out_dim, out_dim, kernel_size=(kernel_size[0], 1, 1),
                                 padding=(pad0, 0, 0))
        self.conv_v3 = nn.Conv3d(out_dim, out_dim, kernel_size=(1, kernel_size[1], 1),
                                 padding=(0, pad1, 0))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_l = self.conv_l3(x_l)

        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        x_r = self.conv_r3(x_r)

        x_v = self.conv_v1(x)
        x_v = self.conv_v2(x_v)
        x_v = self.conv_v3(x_v)

        #combine two paths
        x = x_l + x_r + x_v
        return x

class ResidualBlock(nn.Module):
    def __init__(self, indim):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(indim, indim*2, kernel_size=1, bias=False)
        self.norm1 = nn.BatchNorm3d(indim*2)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv3d(indim*2, indim*2, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm3d(indim*2)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv3d(indim*2, indim, kernel_size=1, bias=False)
        self.norm3 = nn.BatchNorm3d(indim)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        #parameter initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu1(residual)
        residual = self.conv2(residual)
        residual = self.relu2(residual)
        residual = self.conv3(residual)
        residual = self.relu3(residual)
        out = x + residual
        return out

class ResidualBlock_D(nn.Module):
    def __init__(self, indim):
        super(ResidualBlock_D, self).__init__()
        self.conv1 = nn.Conv3d(indim, indim*2, kernel_size=1, bias=False)
        self.norm1 = nn.BatchNorm3d(indim*2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(indim*2, indim*2, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm3d(indim*2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv3d(indim*2, indim, kernel_size=1, bias=False)
        self.norm3 = nn.BatchNorm3d(indim)
        self.relu3 = nn.ReLU(inplace=True)
        #parameter initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu1(residual)
        residual = self.conv2(residual)
        residual = self.relu2(residual)
        residual = self.conv3(residual)
        residual = self.relu3(residual)
        out = x + residual
        return out


class NetS(nn.Module):
    def __init__(self, ngpu, channel_dim_input, ndf_input):
        super(NetS, self).__init__()
        self.ngpu = ngpu
        self.channel_dim = channel_dim_input
        self.ndf = ndf_input

        self.convblock1 = nn.Sequential(
            # input is (channel_dim) x 64 x 64 x 64
            nn.Conv3d(self.channel_dim, self.ndf, 7, 2, 3, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32 x 32
        )
        self.convblock1_1 = ResidualBlock(self.ndf)
        self.convblock2 = nn.Sequential(
            # state size. (ndf) x 32 x 32 x 32
            nn.Conv3d(self.ndf, self.ndf * 2, 5, 2, 2, bias=False),
            nn.BatchNorm3d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16 x 16
        )
        self.convblock2_1 = ResidualBlock(self.ndf*2)
        self.convblock3 = nn.Sequential(
            # state size. (ndf*2) x 16 x 16 x 16
            nn.Conv3d(self.ndf * 2, self.ndf * 4, 5, 2, 2, bias=False),
            nn.BatchNorm3d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8 x 8
        )
        self.convblock3_1 = ResidualBlock(self.ndf*4)
        self.convblock4 = nn.Sequential(
            # state size. (ndf*4) x 8 x 8 x 8
            nn.Conv3d(self.ndf * 4, self.ndf * 8, 5, 2, 2, bias=False),
            nn.BatchNorm3d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4 x 4
        )
        self.convblock4_1 = ResidualBlock(self.ndf*8)
        self.convblock5 = nn.Sequential(
            # state size. (ndf*8) x 4 x 4 x 4
            nn.Conv3d(self.ndf * 8, self.ndf * 16, 5, 2, 2, bias=False),
            nn.BatchNorm3d(self.ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 2 x 2 x 2
        )
        self.convblock5_1 = ResidualBlock(self.ndf*16)
        self.convblock6 = nn.Sequential(
            # state size. (ndf*16) x 2 x 2 x 2
            nn.Conv3d(self.ndf * 16, self.ndf * 32, 3, 2, 1, bias=False),
            nn.BatchNorm3d(self.ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*32) x 1 x 1 x 1
        )

        self.convblock7 = nn.Sequential(
            # state size. (ndf*32) x 1 x 1 x 1
            nn.Conv3d(self.ndf * 32, self.ndf * 8, kernel_size=1, bias=False),
            nn.BatchNorm3d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 1 x 1 x 1
        )
        	
        self.deconvblock1 = nn.Sequential(
            # state size. (ngf*8) x 1 x 1 x 1
            nn.ConvTranspose3d(self.ndf * 8, self.ndf * 32, kernel_size=1, bias=False),
            nn.BatchNorm3d(self.ndf * 32),
            nn.ReLU(True),
            # state size. (ngf*32) x 1 x 1 x 1
        )

        self.deconvblock2 = nn.Sequential(
            # state size. (cat: ngf*32) x 1 x 1 x 1
            nn.Conv3d(self.ndf * 32 * 2, self.ndf * 16, 3, 1, 1, bias=False),
            nn.BatchNorm3d(self.ndf * 16),
            nn.ReLU(True),
            # state size. (ngf*16) x 2 x 2 x 2  (1024*2)
        )
        self.deconvblock2_1 = nn.Sequential(
            # state size. (ndf*16) x 2 x 2 x 2
            nn.Conv3d(self.ndf * 16, self.ndf * 16, kernel_size=1, bias=False),
            nn.BatchNorm3d(self.ndf * 16),
            nn.ReLU(inplace=True),
            # state size. (ndf*16) x 2 x 2 x 2  (1024*2)
        )

        self.deconvblock3 = nn.Sequential(
            # state size. (cat: ngf*16) x 2 x 2 x 2
            nn.Conv3d(self.ndf * 16 * 2, self.ndf * 8, 3, 1, 1, bias=False),
            nn.BatchNorm3d(self.ndf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4 x 4 (512*4)
        )
        self.deconvblock3_1 = ResidualBlock_D(self.ndf*8)

        self.deconvblock4 = nn.Sequential(
            # state size. (ngf*8) x 4 x 4
            GlobalConvBlock(self.ndf*8*2, self.ndf*4, (7, 7, 7)),
            # nn.ConvTranspose2d(ndf * 8 * 2, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm3d(self.ndf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8 x 8 (256*8)
        )
        self.deconvblock4_1 = ResidualBlock_D(self.ndf*4)

        self.deconvblock5 = nn.Sequential(
            # state size. (ngf*4) x 8 x 8 x 8
            GlobalConvBlock(self.ndf*4*2, self.ndf*2, (7, 7, 7)),
            # nn.ConvTranspose2d(ndf * 8 * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(self.ndf*2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16 x 16 (128*16)
        )
        self.deconvblock5_1 = ResidualBlock_D(self.ndf*2)

        self.deconvblock6 = nn.Sequential(
            # state size. (ngf*4) x 16 x 16 x 16
            GlobalConvBlock(self.ndf*2*2, self.ndf, (9, 9, 9)),
            # nn.ConvTranspose2d(ndf * 4 * 2, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(self.ndf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32 x 32 (64*32)
        )
        self.deconvblock6_1 = ResidualBlock_D(self.ndf)

        self.deconvblock7 = nn.Sequential(
            # state size. (ngf) x 32 x 32 x 32
            GlobalConvBlock(self.ndf*2, self.ndf, (9, 9, 9)),
            # nn.ConvTranspose2d(ndf * 2 * 2,     ndf, 4, 2, 1, bias=False),
            nn.BatchNorm3d(self.ndf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64 x 64
        )
        self.deconvblock7_1 = ResidualBlock_D(self.ndf)

        self.deconvblock8 = nn.Sequential(
            # state size. (ngf) x 64 x 64 x 64
            nn.Conv3d(self.ndf, 1, 5, 1, 2, bias= False),
            # state size. 1 x 64 x 64 x 64
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input):
        # for now it only supports one GPU
        ##if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu is 1:
            #print("netS.forward");
            #print("input", input.size())

            encoder1 = self.convblock1(input)
            encoder1 = self.convblock1_1(encoder1)
            #print("encoder1 ", encoder1.size())

            encoder2 = self.convblock2(encoder1)
            encoder2 = self.convblock2_1(encoder2)
            #print("encoder2 ", encoder2.size())

            encoder3 = self.convblock3(encoder2)
            encoder3 = self.convblock3_1(encoder3)
            #print("encoder3 ", encoder3.size())

            encoder4 = self.convblock4(encoder3)
            encoder4 = self.convblock4_1(encoder4)
            #print("encoder4 ", encoder4.size())

            encoder5 = self.convblock5(encoder4)
            encoder5 = self.convblock5_1(encoder5)
            #print("encoder5 ", encoder5.size())

            encoder6 = self.convblock6(encoder5)
            #print("encoder6  ", encoder6.size())

            encoder7 = self.convblock7(encoder6)
            #print("encoder7  ", encoder7.size())

            #print("decoder")
            decoder1 = self.deconvblock1(encoder7)
            #print("decoder1 1", decoder1.size())
            decoder1 = torch.cat([encoder6,decoder1],1)
            #print("decoder1 2", decoder1.size())
            decoder1 = F.interpolate(decoder1, size = encoder5.size()[2:], mode='trilinear', align_corners=True)
            #print("decoder1 3", decoder1.size())

            decoder2 = self.deconvblock2(decoder1)
            #print("decoder2 1", decoder2.size())
            decoder2 = self.deconvblock2_1(decoder2)
            #print("decoder2 2", decoder2.size())
            # concatenate along depth dimension
            decoder2 = torch.cat([encoder5,decoder2],1)
            #print("decoder2 3", decoder2.size())
            decoder2 = F.interpolate(decoder2, size = encoder4.size()[2:], mode='trilinear', align_corners=True)
            #print("decoder2 4", decoder2.size())

            decoder3 = self.deconvblock3(decoder2)
            #print("decoder3 1", decoder3.size())
            decoder3 = self.deconvblock3_1(decoder3)
            #print("decoder3 2", decoder3.size())
            decoder3 = torch.cat([encoder4,decoder3],1)
            #print("decoder3 3", decoder3.size())
            decoder3 = F.interpolate(decoder3, size = encoder3.size()[2:], mode='trilinear', align_corners=True)
            #print("decoder3 4", decoder3.size())

            decoder4 = self.deconvblock4(decoder3)
            #print("decoder4 1", decoder3.size())
            decoder4 = self.deconvblock4_1(decoder4)
            #print("decoder4 2", decoder3.size())
            decoder4 = torch.cat([encoder3,decoder4],1)
            #print("decoder4 3", decoder4.size())
            decoder4 = F.interpolate(decoder4, size = encoder2.size()[2:], mode='trilinear', align_corners=True)
            #print("decoder4 4", decoder4.size())

            decoder5 = self.deconvblock5(decoder4)
            #print("decoder5 1", decoder5.size())
            decoder5 = self.deconvblock5_1(decoder5)
            #print("decoder5 2", decoder5.size())
            decoder5 = torch.cat([encoder2,decoder5],1)
            #print("decoder5 3", decoder5.size())
            decoder5 = F.interpolate(decoder5, size = encoder1.size()[2:], mode='trilinear', align_corners=True)
            #print("decoder5 4", decoder5.size())

            decoder6 = self.deconvblock6(decoder5)
            #print("decoder6 1", decoder6.size())
            decoder6 = self.deconvblock6_1(decoder6)
            #print("decoder6 2", decoder6.size())
            decoder6 = torch.cat([encoder1,decoder6],1)
            #print("decoder6 3", decoder6.size())
            decoder6 = F.interpolate(decoder6, size = input.size()[2:], mode='trilinear', align_corners=True)
            #print("decoder6 4", decoder6.size())

            decoder7 = self.deconvblock7(decoder6)
            #print("decoder7 1", decoder7.size())
            decoder7 = self.deconvblock7_1(decoder7)
            #print("decoder7 2", decoder7.size())
            decoder8 = self.deconvblock8(decoder7)
            #print("decoder8 ", decoder8.size())
        ##else:
            ##print('For now we only support one GPU')
        
            return decoder8


class NetC(nn.Module):
    def __init__(self, ngpu, channel_dim_input, ndf_input):
        super(NetC, self).__init__()
        self.ngpu = ngpu
        self.channel_dim = channel_dim_input
        self.ndf = ndf_input

        self.convblock1 = nn.Sequential(
            # input is (channel_dim) x 128 x 128
            nn.Conv3d(self.channel_dim, self.ndf, 7, 2, 3, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf) x 64 x 64
        )
        self.convblock1_1 = nn.Sequential(
            # state size. (ndf) x 64 x 64
            GlobalConvBlock(self.ndf, self.ndf * 2, (13, 13, 13)),
            # nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf*2) x 64 x 64
        )
        self.convblock2 = nn.Sequential(
            # state size. (ndf * 2) x 64 x 64
            nn.Conv2d(self.ndf * 1, self.ndf * 2, 5, 2, 2, bias=False),
            nn.BatchNorm3d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf*2) x 32 x 32
        )
        self.convblock2_1 = nn.Sequential(
            # input is (ndf*2) x 32 x 32
            GlobalConvBlock(self.ndf * 2, self.ndf * 4, (11, 11, 11)),
            nn.BatchNorm3d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf*4) x 32 x 32
        )
        self.convblock3 = nn.Sequential(
            # state size. (ndf * 4) x 32 x 32
            nn.Conv3d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf*4) x 16 x 16
        )
        self.convblock3_1 = nn.Sequential(
            # input is (ndf*4) x 16 x 16
            GlobalConvBlock(self.ndf * 4, self.ndf * 8, (9, 9, 9)),
            nn.BatchNorm3d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf * 8) x 16 x 16
        )
        self.convblock4 = nn.Sequential(
            # state size. (ndf*4) x 16 x 16
            nn.Conv3d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm3d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf*8) x 8 x 8
        )
        self.convblock4_1 = nn.Sequential(
            # input is (ndf*8) x 8 x 8
            GlobalConvBlock(self.ndf * 8, self.ndf * 16, (7, 7, 7)),
            nn.BatchNorm3d(self.ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf*16) x 8 x 8
        )
        self.convblock5 = nn.Sequential(
            # state size. (ndf*8) x 8 x 8
            nn.Conv3d(self.ndf * 8, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm3d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf*16) x 4 x 4
        )
        self.convblock5_1 = nn.Sequential(
            # input is (ndf*16) x 4 x 4
            nn.Conv3d(self.ndf * 16, self.ndf * 32, 3, 1, 1, bias=False),
            nn.BatchNorm3d(self.ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf*32) x 4 x 4
        )
        self.convblock6 = nn.Sequential(
            # state size. (ndf*32) x 4 x 4
            nn.Conv3d(self.ndf * 8, self.ndf * 8, 3, 2, 1, bias=False),
            nn.BatchNorm3d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf*32) x 2 x 2
        )
        # self._initialize_weights()
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2]* m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.zero_()

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu is 1:
            #print("Net.C")
            batchsize = input.size()[0]
            #print("input ", input.size())
            out1 = self.convblock1(input)
            #print("out1 ", out1.size())
            # out1 = self.convblock1_1(out1)
            out2 = self.convblock2(out1)
            #print("out2 ", out2.size())
            # out2 = self.convblock2_1(out2)
            out3 = self.convblock3(out2)
            #print("out3 ", out3.size())
            # out3 = self.convblock3_1(out3)
            out4 = self.convblock4(out3)
            #print("out4 ", out4.size())
            # out4 = self.convblock4_1(out4)
            out5 = self.convblock5(out4)
            #print("out5 ", out5.size())
            # out5 = self.convblock5_1(out5)
            out6 = self.convblock6(out5)
            #print("out6 ", out6.size())
            # out6 = self.convblock6_1(out6) + out6
            output = torch.cat((input.view(batchsize,-1),1*out1.view(batchsize,-1),
                                2*out2.view(batchsize,-1),2*out3.view(batchsize,-1),
                                2*out4.view(batchsize,-1),2*out5.view(batchsize,-1),
                                4*out6.view(batchsize,-1)),1)
            #print("output ", output.size())
        else:
            print('For now we only support one GPU')

        return output

