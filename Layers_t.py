import torch
import torch.nn as nn
from Sublayers_t import FeedForward, MultiHeadAttention, Norm

'''d_model 128'''
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)
        # self.dropout_1 = nn.Dropout(dropout)
        # self.dropout_2 = nn.Dropout(dropout)

    '''为什么要使用加法'''
    def forward(self, x,mask):
        x2 = self.norm_1(x)
        # x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x = x + self.attn(x2, x2, x2, mask=mask)
        x2 = self.norm_2(x)
        # x = x + self.dropout_2(self.ff(x2))
        x = x + self.ff(x2)
        return x


# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        # self.dropout_1 = nn.Dropout(dropout)
        # self.dropout_2 = nn.Dropout(dropout)
        # self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model)
        self.attn_2 = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)

    def forward(self, x, e_outputs, mask):
        x2 = self.norm_1(x)
        x = x + self.attn_1(x2, x2, x2, mask=None)
        # x = x + self.attn_1(x2, x2, x2, mask)
        x2 = self.norm_2(x)
        # x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, src_mask))
        x = x + self.attn_2(x2, e_outputs, e_outputs, mask)
        x2 = self.norm_3(x)
        # x = x + self.dropout_3(self.ff(x2))
        x = x + self.ff(x2)
        return x