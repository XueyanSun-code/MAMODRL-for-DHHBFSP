import torch
import torch.nn as nn
from Layers_t import EncoderLayer, DecoderLayer

from Sublayers_t import Norm
import copy


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

'''去掉注意力， 输入工件的全局信息'''
class Encoder(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.N = cfg.N
        # self.embed = Embedder(vocab_size, d_model)
        # self.embedding=nn.Linear(cfg.input_size,cfg.d_model)
        # self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(cfg.d_model, cfg.heads), cfg.N)
        self.norm = Norm(cfg.d_model)

    def forward(self, x,mask):
        # x = self.embedding(x)
        # x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x,mask)
        return self.norm(x)


'''修改attention 让其和pointer 一样'''
class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.N = 1#cfg.N
        # self.embed = Embedder(vocab_size, d_model)
        # self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(cfg.d_model, cfg.heads), self.N)
        self.norm = Norm(cfg.d_model)

    def forward(self, x, e_outputs, mask):
        # x = self.embed(trg)
        # x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, mask)
        return self.norm(x)



