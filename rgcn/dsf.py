import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class DSF_embedding(nn.Module):
    def __init__(self, seg_len, dim, dropout_rate=0.3, seg_dim=10):
        super(DSF_embedding, self).__init__()
        self.seg_len = seg_len

        self.dim_linear = nn.Linear(1, seg_dim)
        self.linear = nn.Linear(seg_len, dim)
        self.res_linear = nn.Linear(dim * int(dim/seg_len) * seg_dim, dim)

        self.dropout = torch.nn.Dropout(dropout_rate)
        self.norm_layer = torch.nn.LayerNorm(dim)

    def forward(self, e):
        e = e.unsqueeze(-1)

        e = self.dim_linear(e)
        batch, ts_len, ts_dim = e.shape
        e_segment = rearrange(e, 'b (seg_num seg_len) d -> (b d seg_num) seg_len', seg_len=self.seg_len)

        e_embed = self.linear(e_segment)
        e_embed = rearrange(e_embed, '(b d seg_num) d_model -> b d seg_num d_model', b=batch, d=ts_dim)
        e_embed = rearrange(e_embed, 'b d seg_num d_model -> b (d seg_num d_model)')
        e_embed = self.dropout(e_embed)

        e_embed = self.res_linear(e_embed)
        e_embed = self.norm_layer(e_embed)

        return e_embed