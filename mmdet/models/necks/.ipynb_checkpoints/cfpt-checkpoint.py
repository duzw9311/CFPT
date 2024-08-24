import torch
import math
import einops
import torch.nn as nn
import copy
import torch.nn.functional as F
from collections import OrderedDict
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mmcv.runner import BaseModule, auto_fp16
from ..builder import NECKS

class LayerNormProxy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')


def BasicConv(filter_in, filter_out, kernel_size, stride=1, pad=None):
    if not pad:
        pad = (kernel_size - 1) // 2 if kernel_size else 0
    else:
        pad = pad

    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.ReLU()),
    ]))

class CrossLayerPosEmbedding3D(nn.Module):
    def __init__(self, num_heads=4, window_size=(5, 3, 1), spatial=True):
        super(CrossLayerPosEmbedding3D, self).__init__()
        self.spatial = spatial
        self.num_heads = num_heads
        self.layer_num = len(window_size)
        if self.spatial:
            self.num_token = sum([i ** 2 for i in window_size])
            self.num_token_per_level = [i ** 2 for i in window_size]
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[0] - 1), num_heads))
            coords_h = [torch.arange(ws) - ws // 2 for ws in window_size]
            coords_w = [torch.arange(ws) - ws // 2 for ws in window_size]
            coords_h = [coords_h[i] * window_size[0] / window_size[i] for i in range(len(coords_h) - 1)] + [
                coords_h[-1]]
            coords_w = [coords_w[i] * window_size[0] / window_size[i] for i in range(len(coords_w) - 1)] + [
                coords_w[-1]]
            coords = [torch.stack(torch.meshgrid([coord_h, coord_w])) for coord_h, coord_w in
                      zip(coords_h, coords_w)]
            coords_flatten = torch.cat([torch.flatten(coord, 1) for coord in coords], dim=-1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += window_size[0] - 1
            relative_coords[:, :, 1] += window_size[0] - 1
            relative_coords[:, :, 0] *= 2 * window_size[0] - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)
            trunc_normal_(self.relative_position_bias_table, std=.02)
        else:
            self.num_token = sum([i for i in window_size])
            self.num_token_per_level = [i for i in window_size]
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[0] - 1), num_heads))
            coords_c = [torch.arange(ws) - ws // 2 for ws in window_size]
            coords_c = [coords_c[i] * window_size[0] / window_size[i] for i in range(len(coords_c) - 1)] + [
                coords_c[-1]]
            coords = torch.cat(coords_c, dim=0)
            coords_flatten = torch.stack([torch.flatten(coord, 0) for coord in coords], dim=-1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += window_size[0] - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)
            trunc_normal_(self.relative_position_bias_table, std=.02)

        self.absolute_position_bias = nn.Parameter(torch.zeros(len(window_size), num_heads, 1, 1, 1))
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self):
        pos_indicies = self.relative_position_index.view(-1)
        pos_indicies_floor = torch.floor(pos_indicies).long()
        pos_indicies_ceil = torch.ceil(pos_indicies).long()
        value_floor = self.relative_position_bias_table[pos_indicies_floor]
        value_ceil = self.relative_position_bias_table[pos_indicies_ceil]
        weights_ceil = pos_indicies - pos_indicies_floor.float()
        weights_floor = 1.0 - weights_ceil

        pos_embed = weights_floor.unsqueeze(-1) * value_floor + weights_ceil.unsqueeze(-1) * value_ceil
        pos_embed = pos_embed.reshape(1, 1, self.num_token, -1, self.num_heads).permute(0, 4, 1, 2, 3)

        pos_embed = pos_embed.split(self.num_token_per_level, 3)
        layer_embed = self.absolute_position_bias.split([1 for i in range(self.layer_num)], 0)
        pos_embed = torch.cat([i + j for (i, j) in zip(pos_embed, layer_embed)], dim=-2)
        return pos_embed


class ConvPosEnc(BaseModule):
    def __init__(self, dim, k=3, act=True):
        super(ConvPosEnc, self).__init__()
        self.proj = nn.Conv2d(dim,
                              dim,
                              to_2tuple(k),
                              to_2tuple(1),
                              to_2tuple(k // 2),
                              groups=dim)
        self.activation = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        feat = self.proj(x)
        x = x + self.activation(feat)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        return x


class Mlp(BaseModule):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

def overlaped_window_partition(x, window_size, stride, pad):
    B, C, H, W = x.shape
    out = torch.nn.functional.unfold(x, kernel_size=(window_size, window_size), stride=stride, padding=pad)
    return out.reshape(B, C, window_size * window_size, -1).permute(0, 3, 2, 1)


def overlaped_window_reverse(x, H, W, window_size, stride, padding):
    B, Wm, Wsm, C = x.shape
    Ws, S, P = window_size, stride, padding
    x = x.permute(0, 3, 2, 1).reshape(B, C * Wsm, Wm)
    out = torch.nn.functional.fold(x, output_size=(H, W), kernel_size=(Ws, Ws), padding=P, stride=S)
    return out

def overlaped_channel_partition(x, window_size, stride, pad):
    B, HW, C, _ = x.shape
    out = torch.nn.functional.unfold(x, kernel_size=(window_size, 1), stride=(stride, 1), padding=(pad, 0))
    out = out.reshape(B, HW, window_size, -1)
    return out


def overlaped_channel_reverse(x, window_size, stride, pad, outC):
    B, C, Ws, HW = x.shape
    x = x.permute(0, 3, 2, 1).reshape(B, HW * Ws, C)
    out = torch.nn.functional.fold(x, output_size=(outC, 1), kernel_size=(window_size, 1), padding=(pad, 0),
                                   stride=(stride, 1))
    return out

class CrossLayerSpatialAttention(BaseModule):
    def __init__(self, in_dim, layer_num=3, beta=1, num_heads=4, mlp_ratio=2, reduction=4):
        super(CrossLayerSpatialAttention, self).__init__()
        assert beta % 2 != 0, "error, beta must be an odd number!"
        self.num_heads = num_heads
        self.reduction = reduction
        self.window_sizes = [(2 ** i + beta) if i != 0 else (2 ** i + beta - 1) for i in range(layer_num)][::-1]
        self.token_num_per_layer = [i ** 2 for i in self.window_sizes]
        self.token_num = sum(self.token_num_per_layer)

        self.stride_list = [2 ** i for i in range(layer_num)][::-1]
        self.padding_list = [[0, 0] for i in self.window_sizes]
        self.shape_list = [[0, 0] for i in range(layer_num)]

        self.hidden_dim = in_dim // reduction
        self.head_dim = self.hidden_dim // num_heads

        self.cpe = nn.ModuleList(
            nn.ModuleList([ConvPosEnc(dim=in_dim, k=3),
                           ConvPosEnc(dim=in_dim, k=3)])
            for i in range(layer_num)
        )

        self.norm1 = nn.ModuleList(LayerNormProxy(in_dim) for i in range(layer_num))
        self.norm2 = nn.ModuleList(nn.LayerNorm(in_dim) for i in range(layer_num))
        self.qkv = nn.ModuleList(
            nn.Conv2d(in_dim, self.hidden_dim * 3, kernel_size=1, stride=1, padding=0)
            for i in range(layer_num)
        )

        mlp_hidden_dim = int(in_dim * mlp_ratio)
        self.mlp = nn.ModuleList(
            Mlp(
                in_features=in_dim,
                hidden_features=mlp_hidden_dim)
            for i in range(layer_num)
        )

        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.ModuleList(
            nn.Conv2d(self.hidden_dim, in_dim, kernel_size=1, stride=1, padding=0) for i in range(layer_num)
        )

        self.pos_embed = CrossLayerPosEmbedding3D(num_heads=num_heads, window_size=self.window_sizes, spatial=True)
        
    def forward(self, x_list, extra=None):
        WmH, WmW = x_list[-1].shape[-2:]
        shortcut_list = []
        q_list, k_list, v_list = [], [], []

        for i, x in enumerate(x_list):
            B, C, H, W = x.shape
            ws_i, stride_i = self.window_sizes[i], self.stride_list[i]
            pad_i = (math.ceil((stride_i * (WmH - 1.) - H + ws_i) / 2.), math.ceil((stride_i * (WmW - 1.) - W + ws_i) / 2.))
            self.padding_list[i] = pad_i
            
            self.shape_list[i] = [H, W]

            x = self.cpe[i][0](x)
            shortcut_list.append(x)
            qkv = self.qkv[i](x)
            qkv_windows = overlaped_window_partition(qkv, ws_i, stride=stride_i, pad=pad_i)
            qkv_windows = qkv_windows.reshape(B, WmH * WmW, ws_i * ws_i, 3, self.num_heads, self.head_dim).permute(3, 0,
                                                                                                                   4, 1,
                                                                                                                   2, 5)
            q_windows, k_windows, v_windows = qkv_windows[0], qkv_windows[1], qkv_windows[2]
            q_list.append(q_windows)
            k_list.append(k_windows)
            v_list.append(v_windows)

        q_stack = torch.cat(q_list, dim=-2)
        k_stack = torch.cat(k_list, dim=-2)
        v_stack = torch.cat(v_list, dim=-2)

        attn = F.normalize(q_stack, dim=-1) @ F.normalize(k_stack, dim=-1).transpose(-1, -2)
        attn = attn + self.pos_embed()
        attn = self.softmax(attn)

        out = attn @ v_stack
        out = out.permute(0, 2, 3, 1, 4).reshape(B, WmH * WmW, self.token_num, self.hidden_dim)

        out_split = out.split(self.token_num_per_layer, dim=-2)
        out_list = []
        for i, out_i in enumerate(out_split):
            ws_i, stride_i, pad_i = self.window_sizes[i], self.stride_list[i], self.padding_list[i]
            H, W = self.shape_list[i]
            out_i = overlaped_window_reverse(out_i, H, W, ws_i, stride_i, pad_i)
            out_i = shortcut_list[i] + self.norm1[i](self.proj[i](out_i))
            out_i = self.cpe[i][1](out_i)
            out_i = out_i.permute(0, 2, 3, 1)
            out_i = out_i + self.mlp[i](self.norm2[i](out_i))
            out_i = out_i.permute(0, 3, 1, 2)
            out_list.append(out_i)
        return out_list


class CrossLayerChannelAttention(BaseModule):
    def __init__(self, in_dim, layer_num=3, alpha=1, num_heads=4, mlp_ratio=2, reduction=4):
        super(CrossLayerChannelAttention, self).__init__()
        assert alpha % 2 != 0, "error, alpha must be an odd number!"
        self.num_heads = num_heads
        self.reduction = reduction
        self.hidden_dim = in_dim // reduction
        self.in_dim = in_dim
        self.window_sizes = [(4 ** i + alpha) if i != 0 else (4 ** i + alpha - 1) for i in range(layer_num)][::-1]
        self.token_num_per_layer = [i for i in self.window_sizes]
        self.token_num = sum(self.token_num_per_layer)

        self.stride_list = [(4 ** i) for i in range(layer_num)][::-1]
        self.padding_list = [0 for i in self.window_sizes]
        self.shape_list = [[0, 0] for i in range(layer_num)]
        self.unshuffle_factor = [(2 ** i) for i in range(layer_num)][::-1]

        self.cpe = nn.ModuleList(
            nn.ModuleList([ConvPosEnc(dim=in_dim, k=3),
                           ConvPosEnc(dim=in_dim, k=3)])
            for i in range(layer_num)
        )
        self.norm1 = nn.ModuleList(LayerNormProxy(in_dim) for i in range(layer_num))
        self.norm2 = nn.ModuleList(nn.LayerNorm(in_dim) for i in range(layer_num))

        self.qkv = nn.ModuleList(
            nn.Conv2d(in_dim, self.hidden_dim * 3, kernel_size=1, stride=1, padding=0)
            for i in range(layer_num)
        )

        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.ModuleList(nn.Conv2d(self.hidden_dim, in_dim, kernel_size=1, stride=1, padding=0) for i in range(layer_num))

        mlp_hidden_dim = int(in_dim * mlp_ratio)
        self.mlp = nn.ModuleList(
            Mlp(
                in_features=in_dim,
                hidden_features=mlp_hidden_dim)
            for i in range(layer_num)
        )

        self.pos_embed = CrossLayerPosEmbedding3D(num_heads=num_heads, window_size=self.window_sizes, spatial=False)
        
    def forward(self, x_list, extra=None):
        shortcut_list, reverse_shape = [], []
        q_list, k_list, v_list = [], [], []
        for i, x in enumerate(x_list):
            B, C, H, W = x.shape
            self.shape_list[i] = [H, W]
            ws_i, stride_i = self.window_sizes[i], self.stride_list[i]
            pad_i = math.ceil((stride_i * (self.hidden_dim - 1.) - (self.unshuffle_factor[i])**2 * self.hidden_dim + ws_i) / 2.)
            self.padding_list[i] = pad_i
            x = self.cpe[i][0](x)
            shortcut_list.append(x)

            qkv = self.qkv[i](x)
            qkv = F.pixel_unshuffle(qkv, downscale_factor=self.unshuffle_factor[i])
            reverse_shape.append(qkv.size(1) // 3)

            qkv_window = einops.rearrange(qkv, "b c h w -> b (h w) c ()")
            qkv_window = overlaped_channel_partition(qkv_window, ws_i, stride=stride_i, pad=pad_i)
            qkv_window = einops.rearrange(qkv_window, "b hw wsm (n nh c) -> n b nh c wsm hw", n=3, nh=self.num_heads)
            q_windows, k_windows, v_windows = qkv_window[0], qkv_window[1], qkv_window[2]
            q_list.append(q_windows)
            k_list.append(k_windows)
            v_list.append(v_windows)

        q_stack = torch.cat(q_list, dim=-2)
        k_stack = torch.cat(k_list, dim=-2)
        v_stack = torch.cat(v_list, dim=-2)
        attn = F.normalize(q_stack, dim=-1) @ F.normalize(k_stack, dim=-1).transpose(-2, -1)

        attn = attn + self.pos_embed()
        attn = self.softmax(attn)
        out = attn @ v_stack
        out = einops.rearrange(out, "b nh c ws hw -> b (nh c) ws hw")

        out_split = out.split(self.token_num_per_layer, dim=-2)
        out_list = []
        for i, out_i in enumerate(out_split):
            ws_i, stride_i, pad_i = self.window_sizes[i], self.stride_list[i], self.padding_list[i]
            out_i = overlaped_channel_reverse(out_i, ws_i, stride_i, pad_i, outC=reverse_shape[i])
            out_i = out_i.permute(0, 2, 1, 3).reshape(B, -1, self.shape_list[-1][0], self.shape_list[-1][1])
            out_i = F.pixel_shuffle(out_i, upscale_factor=self.unshuffle_factor[i])

            out_i = shortcut_list[i] + self.norm1[i](self.proj[i](out_i))
            out_i = self.cpe[i][1](out_i)
            out_i = out_i.permute(0, 2, 3, 1)
            out_i = out_i + self.mlp[i](self.norm2[i](out_i))
            out_i = out_i.permute(0, 3, 1, 2)
            out_list.append(out_i)

        return out_list

@NECKS.register_module()
class CFPT(BaseModule):
    def __init__(self, in_channels=(256, 512, 1024, 2048), out_channels=256, alpha=1, beta=1, num_heads=4, mlp_ratio=2, reduction=4, init_cfg=None):
        super(CFPT, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = len(in_channels)
        
        self.conv2 = BasicConv(in_channels[1], out_channels, 1)
        self.conv3 = BasicConv(in_channels[2], out_channels, 1)
        self.conv4 = BasicConv(in_channels[3], out_channels, 1)
        self.conv5 = BasicConv(in_channels[3], out_channels, 3, 2, 1)
        
        self.cross_layer_spatial_attn = CrossLayerSpatialAttention(
            in_dim=out_channels,
            layer_num=len(in_channels),
            beta=beta,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            reduction=reduction,
        )
        self.cross_layer_channel_attn = CrossLayerChannelAttention(
            in_dim=out_channels,
            layer_num=len(in_channels),
            alpha=alpha,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            reduction=reduction,
        )
        self.extra_conv1 = BasicConv(out_channels, out_channels, kernel_size=3, stride=2, pad=1)
            
    @auto_fp16()
    def forward(self, x_list):
        x1, x2, x3, x4 = x_list
        x1 = self.conv2(x2)
        x2 = self.conv3(x3)
        x3 = self.conv4(x4)
        x4 = self.conv5(x4)
        x_list_ori = [x1, x2, x3, x4]
        x_list = self.cross_layer_channel_attn(x_list_ori)
        x_list = self.cross_layer_spatial_attn(x_list)
        x_list = [(i + j).type(x1.dtype) for step, (i, j) in enumerate(zip(x_list_ori, x_list))]
        x_list.append(self.extra_conv1(x_list[-1]))
        return tuple(x_list)
