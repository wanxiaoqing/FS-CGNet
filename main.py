import torch
import torch.nn as nn
import numbers
from einops import rearrange
import torch.nn.functional as F


##############################################################################################
# Cross-Scale Global Aggregation Module
class SEFO(nn.Module):
    def __init__(self, left_channel):
        super(SEFO, self).__init__()
        self.conv1 = BasicConv2d(left_channel, left_channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(left_channel, left_channel, kernel_size=3, stride=1, padding=1)
        self.conv_cat = BasicConv2d(2 * left_channel, left_channel, kernel_size=3, stride=1, padding=1)
        self.conv_cat_2 = BasicConv2d(3 * left_channel, left_channel, kernel_size=3, stride=1, padding=1)
        self.avg_pool = nn.AvgPool2d((3, 3), stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.upsample = torch.nn.functional.interpolate  # 假设 `cus_sample` 是这里的 `interpolate`
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.cur_b4 = BasicConv2d(left_channel, left_channel, 3, padding=4, dilation=4)
        self.conv1x1 = nn.Sequential(nn.Conv2d(left_channel, left_channel, 1), nn.BatchNorm2d(left_channel),
                                     nn.ReLU(inplace=True), )

    def forward(self, left, right):
        # 判断 left 和 right 的空间尺寸是否一致，不一致则对 right 进行插值上采样。
        if right.shape != left.shape:
            right = self.upsample(right, scale_factor=2)  # right 上采样
        else:
            right = right

        right = self.conv1(right)
        out = self.conv_cat(torch.cat((left, right), 1))

        size = left.size()[2:]
        out_g1 = out + self.cur_b4(out)
        out_g = out_g1 + F.interpolate(self.conv1x1(self.gap(out)), size, mode='bilinear', align_corners=True)
        right = self.conv2(right) * out_g
        left = self.conv2(left) * out_g
        out = self.conv_cat_2(torch.cat((left, right, out), 1))
        out_g2 = self.sigmoid(self.conv2(out - self.avg_pool(out)))
        out = self.conv2(out) * out_g2

        return out


class GAFE(nn.Module):
    def __init__(self, in_channels):
        super(GAFE, self).__init__()
        self.conv_q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv_k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv_v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.delta = nn.Parameter(torch.Tensor([0.1]))  # initiate as 0.1
        self.avg_pool = nn.AvgPool2d((3, 3), stride=1, padding=1)
        self.amg_pool = nn.MaxPool2d((3, 3), stride=1, padding=1)
        self.conv_rbc = BasicConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.conv_rbc_2 = BasicConv2d(2 * in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, ftr):
        B, C, H, W = ftr.size()
        P = H * W
        ftr_q = self.conv_q(ftr).view(B, -1, P).permute(0, 2, 1)  # [B, P, C']
        ftr_k = self.conv_k(ftr).view(B, -1, P)  # [B, C', P]
        ftr_v = self.conv_v(ftr).view(B, -1, P)  # [B, C, P]
        weights = F.softmax(torch.bmm(ftr_q, ftr_k), dim=1)  # column-wise softmax, [B, P, P]
        G = torch.bmm(ftr_v, weights).view(B, C, H, W)
        out = self.delta * G + ftr
        out = self.conv_rbc(out)
        out = torch.sigmoid(out)
        ftr_avg = self.avg_pool(ftr)
        ftr_max = self.amg_pool(ftr)
        ftr_avg = ftr_avg.mul(out)
        ftr_max = ftr_max.mul(out)
        out = torch.cat((ftr_max, ftr_avg), 1)
        out = self.conv_rbc_2(out)

        return out


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CSGAM(nn.Module):
    def __init__(self, in_channels):
        super(CSGAM, self).__init__()
        self.sefo = SEFO(in_channels)
        self.gafe = GAFE(in_channels)

    def forward(self, x):
        x = self.sefo(x, x)
        x = self.gafe(x)
        return x


##############################################################################################
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# Frequency Domain Spectral Feature Module
class FSAS(nn.Module):
    def __init__(self, dim, bias=False):
        super(FSAS, self).__init__()
        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 6, bias=bias)
        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.norm = LayerNorm(dim * 2, LayerNorm_type='WithBias')
        self.gelu = nn.GELU()
        self.patch_size = 1  #####################  17

    def forward(self, x):
        hidden = self.to_hidden(x)
        # q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)
        q, k, v = self.gelu(self.to_hidden_dw(hidden)).chunk(3, dim=1)
        q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        k_patch = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        q_fft = torch.fft.rfft2(q_patch.float())
        k_fft = torch.fft.rfft2(k_patch.float())
        out = q_fft * k_fft
        out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))
        out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                        patch2=self.patch_size)
        out = self.norm(out)
        out = self.gelu(out)
        output = v * out
        output = self.project_out(output)
        return output


##############################################################################################

##############################################################################################
# Multi-Scale Spectral-Channel Fusion Attention Module
class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        self.w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out


class MultiScaleAttention(nn.Module):
    def __init__(self, channel, b=1, gamma=2, scales=[1, 2, 4]):
        super(MultiScaleAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.scales = scales
        self.conv1 = nn.ModuleList([nn.Conv1d(1, 1, kernel_size=s, padding=s // 2, bias=False) for s in scales])
        self.fc_channel = nn.Conv2d(channel, channel, 1, padding=0, bias=True)
        self.fc_spatial = nn.Conv2d(channel, channel, 1, padding=0, bias=True)

        self.sigmoid = nn.Sigmoid()
        self.mix = Mix()

    def forward(self, input):
        # Compute multi-scale channel attention
        channel_attention = []
        for conv in self.conv1:
            x = self.avg_pool(input)
            x1 = conv(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2)
            x2 = self.fc_channel(x).squeeze(-1).transpose(-1, -2)
            out = torch.sum(torch.matmul(x1, x2), dim=1).unsqueeze(-1).unsqueeze(-1)
            channel_attention.append(self.sigmoid(out))

        # Combine multi-scale channel attention
        multi_scale_attention = torch.stack(channel_attention, dim=1).mean(dim=1)

        # Compute spatial attention
        spatial_attention = self.fc_spatial(input)
        spatial_attention = self.sigmoid(spatial_attention)

        # Combine channel and spatial attention
        out = self.mix(multi_scale_attention, spatial_attention)

        return input * out


##############################################################################################
# FS-CGNet: Frequency Spectral-Channel Fusion and Cross-Scale Global Aggregation Network
class FSCGNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=16, dim=64, dropout=0.1):
        super(FSCGNet, self).__init__()
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels, out_channels=8, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )
        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=224, out_channels=64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.fsas = FSAS(dim)
        self.att = MultiScaleAttention(dim)
        self.sefo = SEFO(dim)
        self.gafe = GAFE(dim)

        self.conv = nn.Conv2d(dim, dim // 2, kernel_size=1, stride=1)
        self.fsas1 = FSAS(dim // 2)
        self.att1 = MultiScaleAttention(dim // 2)
        self.sefo1 = SEFO(dim // 2)
        self.gafe1 = GAFE(dim // 2)

        self.conv1 = nn.Conv2d(dim // 2, dim // 4, kernel_size=1, stride=1)
        self.fsas2 = FSAS(dim // 4)
        self.att2 = MultiScaleAttention(dim // 4)
        self.sefo2 = SEFO(dim // 4)
        self.gafe2 = GAFE(dim // 4)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(16, num_classes)
        self.sega = CSGAM(in_channels=64)

    def forward(self, x):
        x = self.conv3d_features(x)  # [64, 8, 28, 11, 11]
        x = rearrange(x, 'b c h w y -> b (c h) w y')
        x = self.conv2d_features(x)  # [64, 64, 9, 9]

        x1 = self.fsas(x)  # [64, 64, 9, 9]
        x2 = self.att(x1)  # [64, 64, 9, 9]
        x3 = self.sefo(x1, x2)  # [64, 64, 9, 9]
        x4 = self.gafe(x3)  # [64, 64, 9, 9]
        x = x2 + x4  # [64, 64, 9, 9]

        x = self.conv(x)  # [64, 32, 9, 9]
        x1 = self.fsas1(x)
        x2 = self.att1(x1)
        x3 = self.sefo1(x1, x2)
        x4 = self.gafe1(x3)
        x = x2 + x4

        x = self.conv1(x)
        x1 = self.fsas2(x)
        x2 = self.att2(x1)
        x3 = self.sefo2(x1, x2)
        x4 = self.gafe2(x3)
        x = x2 + x4

        x = self.pool(x)
        x = x.view(x.size(0), -1)  # 将特征展平为二维，[64, 64]
        x = self.drop(x)  # dropout
        x = self.fc(x)
        return x


if __name__ == '__main__':
    model = FSCGNet(in_channels=1, num_classes=9)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    input = torch.randn(64, 1, 30, 13, 13).to(device)

    y = model(input)
    print(y.size())
