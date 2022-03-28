import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtts.nn.modules.common import LayerNorm
from torchtts.nn.modules.common.functional import sequence_mask
from torchtts.nn.modules.transformer import TransformerEncoder
from torchtts.nn.modules.transformer import TransformerEncoderLayer


DEFAULT_MAX_TARGET_FRAMES = 4096


class SoftLengthRegulator(nn.Module):
    """Soft Length Regulator

    Args:
        in_dim: the number of expected size of input.
        filter_size: the filter size (channels) of convolution.
        kernel: the kernel size of convolution.
        v_transformer_layers: the number of layers of conv, relu, norm and dropout.
        dropout: the dropout value.
    """

    def __init__(self,
                 in_dim,
                 num_heads,
                 ffn_dims,
                 v_transformer_layers=1,
                 enc_ffn_kernels=(9, 1),
                 enc_ffn_dilations=(1, 1),
                 cond_dim=-1,
                 t2t_compatible=True,
                 kernel=3,
                 filter_size_w=8,
                 filter_size_c=8,
                 swish_block_units=(16, 2),
                 dropout=0.1):
        super(SoftLengthRegulator, self).__init__()
        self.filter_size_w = filter_size_w
        self.filter_size_c = filter_size_c
        self.num_heads_upsampling = num_heads * 2
        self.head_dim_upsampling = in_dim // self.num_heads_upsampling

        encoder_layer = TransformerEncoderLayer(model_dim=in_dim,
                                                num_heads=num_heads,
                                                ffn_dims=ffn_dims,
                                                ffn_kernels=enc_ffn_kernels,
                                                ffn_dilations=enc_ffn_dilations,
                                                t2t_compatible=t2t_compatible,
                                                dropout=dropout,
                                                layer_norm_condition_dim=cond_dim)
        encoder_norm = LayerNorm(in_dim, condition_dim=cond_dim)

        self.lconv_substitute = TransformerEncoder(encoder_layer=encoder_layer,
                                                   num_layers=v_transformer_layers,
                                                   norm=encoder_norm)

        self.layers_w_convs = nn.Sequential(nn.Conv1d(in_dim,
                                                      filter_size_w,
                                                      kernel_size=kernel,
                                                      padding=(kernel - 1) // 2),
                                            LayerNorm(filter_size_w, dim=1),
                                            SiLU())

        self.layers_c_convs = nn.Sequential(nn.Conv1d(in_dim,
                                                      filter_size_c,
                                                      kernel_size=kernel,
                                                      padding=(kernel - 1) // 2),
                                            LayerNorm(filter_size_c, dim=1),
                                            SiLU())

        self.swish_block_w = nn.Sequential(nn.Linear(filter_size_w + 2, swish_block_units[0]),
                                           SiLU(),
                                           nn.Linear(swish_block_units[0], swish_block_units[0]),
                                           SiLU())

        self.swish_block_c = nn.Sequential(nn.Linear(filter_size_c + 2, swish_block_units[1]),
                                           SiLU(),
                                           nn.Linear(swish_block_units[1], swish_block_units[1]),
                                           SiLU())

        self.mlp_w_extra = nn.Linear(swish_block_units[0], self.num_heads_upsampling)
        self.mlp_c_extra = nn.Linear(swish_block_units[1], self.head_dim_upsampling)
        self.mha_project_1 = nn.Linear(in_dim, in_dim)
        self.mha_project_2 = nn.Linear(in_dim, in_dim)
        self.mha_project_3 = nn.Linear(in_dim, in_dim)
        self.linear_dur = nn.Sequential(nn.Linear(in_dim, 1),
                                        nn.ReLU())

    def forward(self, phs,
                phs_mask=None,
                layer_norm_condition=None,
                speaking_rate=None,
                total_duration_groundtruth=None):
        v = self.lconv_substitute(phs,
                                  layer_norm_condition=layer_norm_condition,
                                  src_key_padding_mask=~phs_mask if phs_mask is not None else None)  # V=[B,K,in_dim]

        d_log = self.linear_dur(v).squeeze(-1)  # d=[B,K]
        d = torch.clamp(torch.exp(d_log) - 1., min=1e-12)  # d=[B,K]
        if speaking_rate is not None:
            d = d * speaking_rate
        if phs_mask is not None:
            v_padding_mask = ~phs_mask
            d.masked_fill_(v_padding_mask, 0.0)
        if total_duration_groundtruth is not None:
            dur = total_duration_groundtruth.long()
        else:
            dur = torch.clamp(torch.ceil(torch.sum(d, dim=-1)), min=1.0, max=DEFAULT_MAX_TARGET_FRAMES).long()  # s=[B,]
        d_padding_mask = ~sequence_mask(dur)  # D_padding_mask=[B,T]
        k = d.size(-1)
        t = torch.max(dur)
        b = int(len(dur))
        if phs_mask is not None:
            n = v_padding_mask.size(1)  # K
            m = d_padding_mask.size(1)  # T
            v_pad_mask_expand = v_padding_mask.unsqueeze(1).expand(-1, m, n)
            d_pad_mask_expand = d_padding_mask.unsqueeze(2).expand(-1, m, n)
            attn_mask = ~((~v_pad_mask_expand) & (~d_pad_mask_expand))

        e = torch.cumsum(d, dim=-1)  # s=[B,K]
        s = e - d  # e=[B,K]
        s = s.unsqueeze(1).repeat(1, t, 1)  # s=[B, T, K]
        e = e.unsqueeze(1).repeat(1, t, 1)  # e=[B, T, K]
        scale = torch.arange(1, t + 1, device=dur.device).unsqueeze(-1).repeat(b, 1, k)
        if phs_mask is not None:
            s, e = (scale - s).masked_fill(attn_mask, 0), (e - scale).masked_fill(attn_mask, 0)
        else:
            s, e = scale - s, e - scale
        s = s.unsqueeze(-1)  # S=[B, T, K, 1]
        e = e.unsqueeze(-1)  # E=[B, T, K, 1]

        v = self.mha_project_1(v)  # V=[B,K,in_dim]

        w_convs = self.layers_w_convs(v.permute(0, 2, 1)).permute(0, 2, 1)
        w_convs_expand = w_convs.unsqueeze(1).repeat(1, t, 1, 1)
        merged_s_e_w_convs_expand = torch.cat((s, e, w_convs_expand), dim=-1)
        swish_w = self.swish_block_w(merged_s_e_w_convs_expand)
        swish_w_mlp_w_extra = self.mlp_w_extra(swish_w).permute(0, 3, 1, 2)
        if phs_mask is not None:
            swish_w_mlp_w_extra.masked_fill_(v_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        swish_w_mlp_w_extra_softmax = F.softmax(swish_w_mlp_w_extra, dim=-1)

        c_convs = self.layers_c_convs(v.permute(0, 2, 1)).permute(0, 2, 1)
        c_convs_expand = c_convs.unsqueeze(1).repeat(1, t, 1, 1)
        merged_s_e_c_convs_expand = torch.cat((s, e, c_convs_expand), dim=-1)
        swish_c = self.swish_block_c(merged_s_e_c_convs_expand)
        einsum = torch.einsum('bhtk,btkp->bhtp', swish_w_mlp_w_extra_softmax, swish_c)
        swish_c_mlp_c_extra = self.mlp_c_extra(einsum).permute(0, 2, 1, 3)

        v = v.reshape(b, k, self.num_heads_upsampling, self.head_dim_upsampling).permute(0, 2, 1, 3)
        o_left = torch.matmul(swish_w_mlp_w_extra_softmax, v).permute(0, 2, 1, 3)
        o_left = o_left.reshape(b, t, self.num_heads_upsampling * self.head_dim_upsampling)
        o_left = self.mha_project_2(o_left)
        o_right = swish_c_mlp_c_extra.reshape(b, t, self.num_heads_upsampling * self.head_dim_upsampling)
        o_right = self.mha_project_3(o_right)
        out = o_left + o_right  # Out=[B,T,in_dim]

        return out, d, swish_w_mlp_w_extra_softmax


# Cuz Exporting the operator silu to ONNX opset version 12 is not supported
class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)
