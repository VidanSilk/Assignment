import torch
import torch.nn as nn
import torch.nn.functional as functional
from utils import get_normalization_layer, get_activation_layer


class fcLayer(nn.Module):
    def __init__(self, in_dim, out_dim, norm='bn', act='relu'):
        super(fcLayer, self).__init__()

        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = get_normalization_layer(out_dim, norm=norm)
        self.act = get_activation_layer(act=act)

    def forward(self, x):
        out = self.linear(x).permute(0, 2, 1)
        out = self.norm(out)
        out = self.act(out).permute(0, 2, 1)
        return out


class complexFcLayer(nn.Module):
    def __init__(self, in_dim, out_dim, norm='bn', act='relu'):
        super(complexFcLayer, self).__init__()

        self.linear_r = nn.Linear(in_dim, out_dim)
        self.linear_i = nn.Linear(in_dim, out_dim)

        self.norm_r = get_normalization_layer(out_dim, norm=norm)
        self.act_r = get_activation_layer(act=act)
        self.norm_i = get_normalization_layer(out_dim, norm=norm)
        self.act_i = get_activation_layer(act=act)

    def forward(self, x_r, x_i):
        r2r = self.linear_r(x_r)
        r2i = self.linear_i(x_r)
        i2i = self.linear_i(x_i)
        i2r = self.linear_r(x_i)

        real_out = r2r - i2i
        imag_out = i2r + r2i

        real_out = self.act_r(self.norm_r(real_out.permute(0, 2, 1))).permute(0, 2, 1)
        imag_out = self.act_i(self.norm_i(imag_out.permute(0, 2, 1))).permute(0, 2, 1)

        return real_out, imag_out


def get_padding(kernel_size, dilation):
    return int((kernel_size[0] * dilation[0] - dilation[0]) / 2), int((kernel_size[1] * dilation[1] - dilation[1]))


class causalConv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride=(1, 1), dilation=(1, 1), groups=1, bias=True):
        super(causalConv, self).__init__()
        padding = get_padding(kernel_size, dilation)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=(padding[0], 0),
                              dilation=dilation, groups=groups, bias=bias)
        self.padding = padding[1]

    def forward(self, x):
        x = functional.pad(x, [self.padding, 0, 0, 0])
        out = self.conv(x)
        return out


class causalConvTrans(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride=(1, 1), dilation=(1, 1), padding=(0, 0), output_padding=(0, 0)):
        super(causalConvTrans, self).__init__()
        padding = get_padding(kernel_size, dilation)
        self.conv = nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding,
                                       output_padding=output_padding, dilation=dilation)
        self.padding = padding[1]

    def forward(self, x):
        x = functional.pad(x, [self.padding, 0, 0, 0])
        out = self.conv(x)
        return out


class convLayer(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=(3, 2), stride=(2, 1), dilation=(1, 1), norm='bn', act='relu'):
        super(convLayer, self).__init__()

        self.conv = causalConv(in_dim, out_dim, kernel_size=kernel_size, stride=stride, dilation=dilation)
        self.norm = get_normalization_layer(out_dim, norm=norm, dim=2)
        self.act = get_activation_layer(act=act)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.act(out)
        return out


class complexConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=(3, 2), stride=(2, 1), dilation=(1, 1), norm='bn', act='relu'):
        super(complexConvLayer, self).__init__()

        self.conv_r = causalConv(in_dim, out_dim, kernel_size=kernel_size, stride=stride, dilation=dilation)
        self.conv_i = causalConv(in_dim, out_dim, kernel_size=kernel_size, stride=stride, dilation=dilation)

        self.norm_r = get_normalization_layer(out_dim, norm=norm, dim=2)
        self.act_r = get_activation_layer(act=act)
        self.norm_i = get_normalization_layer(out_dim, norm=norm, dim=2)
        self.act_i = get_activation_layer(act=act)

    def forward(self, x_r, x_i):
        r2r = self.conv_r(x_r)
        r2i = self.conv_i(x_r)
        i2i = self.conv_i(x_i)
        i2r = self.conv_r(x_i)

        real_out = r2r - i2i
        imag_out = i2r + r2i

        real_out = self.act_r(self.norm_r(real_out))
        imag_out = self.act_i(self.norm_i(imag_out))

        return real_out, imag_out


class upConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=(3, 2), stride=(2, 1), dilation=(1, 1), norm='bn', act='relu'):
        super(upConvLayer, self).__init__()

        output_padding = [0, 0]
        if stride[0] > 1:
            output_padding[0] = stride[0] - 1
        else:
            output_padding[0] = 0
        if stride[1] > 1:
            output_padding[1] = stride[1] - 1
        else:
            output_padding[1] = 0

        self.deconv = causalConvTrans(in_dim, out_dim, kernel_size, stride=stride,
                                      padding=get_padding(kernel_size, dilation),
                                      output_padding=output_padding, dilation=dilation)
        self.norm = get_normalization_layer(out_dim, norm=norm, dim=2)
        self.act = get_activation_layer(act=act)

    def forward(self, x):
        out = self.deconv(x)
        out = self.norm(out)
        out = self.act(out)
        return out


class complexUpConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=(3, 2), stride=(2, 1), dilation=(1, 1), norm='bn', act='relu'):
        super(complexUpConvLayer, self).__init__()

        self.conv_r = upConvLayer(in_dim, out_dim, kernel_size=kernel_size, stride=stride, dilation=dilation)
        self.conv_i = upConvLayer(in_dim, out_dim, kernel_size=kernel_size, stride=stride, dilation=dilation)

        self.norm_r = get_normalization_layer(out_dim, norm=norm, dim=2)
        self.act_r = get_activation_layer(act=act)
        self.norm_i = get_normalization_layer(out_dim, norm=norm, dim=2)
        self.act_i = get_activation_layer(act=act)

    def forward(self, x_r, x_i):
        r2r = self.conv_r(x_r)
        r2i = self.conv_i(x_r)
        i2i = self.conv_i(x_i)
        i2r = self.conv_r(x_i)

        real_out = r2r - i2i
        imag_out = i2r + r2i

        real_out = self.act_r(self.norm_r(real_out))
        imag_out = self.act_i(self.norm_i(imag_out))

        return real_out, imag_out
