import torch
import torch.nn as nn
import torch.nn.functional as F
from non_local_dot_product import NONLocalBlock2D as NonLocalBlock


class NonLocalUNet(nn.Module):

    def __init__(self, inch,outch,features):
        super(NonLocalUNet, self).__init__()
        self.in_channel = inch
        self.out_channel = outch
        self.features = features
        self.block_sizes = [1, 1, 1]
        self.block_strides = [1, 2, 2]

        # TODO block size is to be variable
        self.conv1 = nn.Conv2d(self.in_channel, self.features, 3, 1, 1)
        self.encoder = Encoder(self.features, [self.features * (2 ** i) for i in range(len(self.block_sizes))],
                               self.block_strides, 1, 3)
        self.bottleneck = NonLocalBlock(self.features * (2 ** (len(self.block_sizes) - 1)))
        self.decoder = Decoder(self.features * (2 ** (len(self.block_sizes) - 1)),
                               [self.features * (2 ** i) for i in reversed(range(len(self.block_sizes) - 1))],
                               self.block_strides[1:], 1, 2)
        self.outconv = nn.Sequential(
            BNRelu(self.features),
            nn.Dropout(0.5),
            nn.Conv2d(self.features, self.out_channel, 1, 1, bias=False)
        )

    def forward(self, x):
        x = self.conv1(x)
        x, skips = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skips)
        x = self.outconv(x)
        return x


class BNRelu(nn.Module):
    def __init__(self, features):
        super(BNRelu, self).__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm2d(features),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, stride, projection_shortcut=True, encode=True):
        super(ResidualBlock, self).__init__()
        self.encode = encode
        self.bnrelu1 = BNRelu(in_features)
        self.projection_shortcut = projection_shortcut
        if self.projection_shortcut:
            self.projection = nn.Conv2d(in_features, out_features, 1 if encode else 3, stride)
        self.conv2d1 = nn.Conv2d(in_features, out_features, 3, stride, 1)
        self.bnrelu2 = BNRelu(out_features)
        self.conv2d2 = nn.Conv2d(out_features, out_features, 3, 1, 1)

    def forward(self, x):
        # print(self.encode,x.shape)
        s = x
        x = self.bnrelu1(x)
        if self.projection_shortcut:
            s = self.projection(x)
        x = self.conv2d1(x)
        x = self.bnrelu2(x)
        x = self.conv2d2(x)
        # print(x.shape,s.shape)
        return x + s


class Encoder(nn.Module):
    def __init__(self, in_feature, features, strides, numblock, numencblock):
        super(Encoder, self).__init__()
        features = [in_feature] + features
        self.blocks = nn.ModuleList([
            nn.Sequential(
                *[ResidualBlock(features[j], features[j + 1], strides[j] if i == 0 else 1, True if i == 0 else False)
                  for i in range(numblock)])
            for j in range(numencblock)])

    def forward(self, x):
        skips = []
        for enc in self.blocks:
            x = enc(x)
            # print(x.shape)
            skips += [x]
        return x, skips[:-1]


class ATTentionBlock(nn.Module):
    def __init__(self, in_feature,out_feature, stride, projection_shortcut):
        super(ATTentionBlock, self).__init__()
        self.bnrelu = BNRelu(in_feature)
        self.projection = nn.ConvTranspose2d(in_feature, out_feature, 3, stride, 1, 1) if projection_shortcut else None
        self.nlb = NonLocalBlock(in_feature)
        self.upsample= nn.ConvTranspose2d(in_feature, out_feature, 3, stride, 1, 1) if projection_shortcut else None
        self.stride=stride
    def forward(self, x):
        s = x
        x = self.bnrelu(x)
        if self.projection is not None:
            s = self.projection(x)
        x = self.nlb(x)
        if self.stride!=1:
            x=self.upsample(x)
        return x + s


class Decoder(nn.Module):
    def __init__(self, in_feature, features, strides, numblocks, numdecblocks):
        super(Decoder, self).__init__()
        features = [in_feature] + features
        self.decoder = nn.ModuleList([
            nn.ModuleList([
                nn.ConvTranspose2d(features[i], features[i + 1], 3, strides[i], 1, 1),
                nn.Sequential(
                    *[ResidualBlock(features[i + 1], features[i + 1], 1, False, False) for _ in range(numblocks)])
            ]) if i != 1 else
            nn.ModuleList([
                ATTentionBlock(features[i],features[i+1], strides[i], True),
                nn.Sequential(*[ResidualBlock(features[i+1], features[i + 1], 1, False, False) for _ in range(numblocks)])
            ])
            for i in range(numdecblocks)
        ])

    def forward(self, x, skips):
        for dec, s in zip(self.decoder, reversed(skips)):
            x = dec[0](x)
            x = dec[1](x + s)
        return x


if __name__ == '__main__':
    model = NonLocalUNet(512)
    optimizer = torch.optim.Adam(model.parameters())
    print(model)
    output = model(torch.randn(1, 3, 256, 256))
    loss = output.mean()
    loss.backward()
    optimizer.step()
    print('END')