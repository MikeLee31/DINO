# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable

import MultiScaleDeformableAttention as MSDA


class MSDeformAttnFunction(Function):
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        output = MSDA.ms_deform_attn_forward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = \
            MSDA.ms_deform_attn_backward(
                value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, grad_output, ctx.im2col_step)

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    # for debug and test only,
    # need to use cuda version instead

    # 1.如何采样点
    N_, S_, M_, D_ = value.shape    # batchsize， key个数， head个数， 维度 
    # sampling_locations = torch.rand(N, Lq, M, L, P, 2).cuda()
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape    # Lq_: query个数， L_:level数， P_: 采样点个数 
    # 把value分割到各个特征层上得到对应的 list value
    # N,H*W,M,D   N,H/2*W/2,M,D  ...
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    # 采样点坐标从[0,1] -> [-1, 1]  F.grid_sample要求采样坐标归一化到[-1, 1]
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        # 这里转换是因为grid_sample 需要  N,C,H_in,W_in 的输入
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, H_, W_)   # 得到每个特征层的value list
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        # 这里转换是因为grid_sample 需要  N,H_out,W_out,2 的grid
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)  # 得到每个特征层的采样点 list
        # N_*M_, D_, Lq_, P_ 采样算法  根据每个特征层采样点到每个特征层的value进行采样  非采样点用0填充
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        # 输出为 N_*M_, D_, Lq_, P_  # 这里意味着在每一层特征层上，也就是 H_*W_ 上采样了  Lq_*P_ 个点的特征作为了value
        sampling_value_list.append(sampling_value_l_)
    # 2.注意力权重计算
    # attention_weights = torch.rand(N, Lq, M, L, P).cuda() + 1e-5
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)

    # 注意力权重 和 采样后的value 进行 weighted sum
    # N_*M_, D_, Lq_, L_ ,P_   ->   N_*M_, D_, Lq_, L_ * P_  * N_*M_, 1, Lq_, L_*P_  -> N_*M_, D_, Lq_
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
    return output.transpose(1, 2).contiguous()  # batchsize,Lq,8 * 32 = 256
