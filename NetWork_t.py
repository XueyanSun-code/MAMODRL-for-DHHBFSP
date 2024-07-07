import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from torch.nn.utils import rnn
from Models_t import Encoder, Decoder

'''输入数据 加工时间 加工机器 该机器可开始加工的相对时间'''


'''Actor'''
class PtrNet1(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = cfg.device
        self.args = cfg
        self.hidden_size=cfg.hidden_size

        self.embedding=nn.Linear(cfg.input_size,cfg.embedding_size)
        self.Vec = nn.Parameter(torch.FloatTensor(self.hidden_size))
        self.Vec2 = nn.Parameter(torch.FloatTensor(self.hidden_size))
        self.W_q = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.W_ref = nn.Conv1d(self.hidden_size , self.hidden_size, 1, 1)
        self.dec_input=nn.Parameter(torch.FloatTensor(self.hidden_size))
        self.W_q2 = nn.Linear(self.hidden_size , self.hidden_size, bias=True)
        self.W_ref2 = nn.Conv1d(self.hidden_size, self.hidden_size, 1, 1)
        self.softmax_T = 1.5  # cfg.softmax_T

        self.Encoder=Encoder(cfg)#input_size, d_model, N, heads
        # self.Decoder=Decoder(cfg)#d_model, N, heads
        self._initialize_weights()

    def _initialize_weights(self, init_min=-0.1, init_max=0.1):
        for param in self.parameters():
            nn.init.uniform_(param.data, init_min, init_max)
    #
    # def make_mask(self,total_operation,max_operation):
    #     mask = torch.ones(total_operation)
    #     for i in range(total_operation):
    #         if i % max_operation==0:
    #             mask[i] = 0
    #     return mask

    '''没有把工序的概念考虑进去'''

    def forward(self, x,mask,train=True,action=None):
        """	x: (batch, job_num, obs_dim)
            enc_h: (batch, city_t, embed)
            dec_input: (batch, 1, embed)
            h: (1, batch, embed)
            return: pi: (batch, city_t), ll: (batch)
            count(batch,job_num)
        """
        batch=x.size(0)
        x_embed=self.embedding(x)  #需要修改，对车间状态特征进行提取
        e_output= self.Encoder(x_embed,mask)
        '''选工件的时候不采用decoder 采用注意力机制 直接选择'''
        query = self.dec_input.repeat(batch,1)
        for i in range(2):
            query = self.glimpse(query, e_output, mask)

            '''思考要不要采用全连接 网络多基层 泛化作用更为明显'''

        probs = self.pointer(query, e_output, mask)
        # log_p = torch.log_softmax(logits, dim=-1) #(batch,total_operations,job_num)
        probs=torch.softmax(probs,dim=-1)
        dist=Categorical(probs)
        if action==None:
            if train:
                action=dist.sample().long()
                return action,dist.log_prob(action)
            else:
                action=torch.argmax(probs, dim=1).long()
            return action

        else:
            action_logprobs=dist.log_prob(action)
            dist_entropy=dist.entropy()
            return action_logprobs,dist_entropy

    # 注意力网络  u1：query   u2：key  ref:value
    def glimpse(self, query, ref, mask, inf=1e9):
        u1 = self.W_q(query).unsqueeze(-1).repeat(1, 1, ref.size(1))  # u1: (batch, 128, city_t)
        u2 = self.W_ref(ref.permute(0, 2, 1))  # u2: (batch, 128, city_t)
        V = self.Vec.unsqueeze(0).unsqueeze(0).repeat(ref.size(0), 1, 1)
        u = torch.bmm(V, torch.tanh(u1 + u2)).squeeze(1)
        # V: (batch, 1, 128) * u1+u2: (batch, 128, city_t) => u: (batch, 1, city_t) => (batch, city_t)
        u = u + inf * mask
        a = F.softmax(u / self.softmax_T, dim=1)
        d = torch.bmm(u2, a.unsqueeze(2)).squeeze(2)
        # # u2: (batch, 128, city_t) * a: (batch, city_t, 1) => d: (batch, 128)
        return d

    def pointer(self, query, ref, mask, inf=1e9):
        u1 = self.W_q2(query).squeeze(1).unsqueeze(-1).repeat(1, 1, ref.size(1))  # u1: (batch, 128, city_t)
        u2 = self.W_ref2(ref.permute(0, 2, 1))  # u2: (batch, 128, city_t)
        V = self.Vec2.unsqueeze(0).unsqueeze(0).repeat(ref.size(0), 1, 1)
        u = torch.bmm(V, self.args.clip_logits * torch.tanh(u1 + u2)).squeeze(1)
        # V: (batch, 1, 128) * u1+u2: (batch, 128, city_t) => u: (batch, 1, city_t) => (batch, city_t)
        u = u + inf * mask
        return u

    # def get_log_likelihood(self, _log_p, pi):
    #     """	args:
    #         _log_p: (batch, city_t, city_t)
    #         pi: (batch, city_t), predicted tour
    #         return: (batch)
    #     """
    #     log_p = torch.gather(input=_log_p, dim=2, index=pi[:, :, None])
    #     return torch.sum(log_p.squeeze(-1), 1)


'''Critic'''
class PtrNet2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = cfg.device
        self.args = cfg
        self.hidden_size=cfg.hidden_size
        #这里的输入包含了权重值
        self.embedding=nn.Linear(cfg.input_size,cfg.embedding_size)
        self.dec_input = nn.Parameter(torch.FloatTensor(self.hidden_size))
        self.Vec = nn.Parameter(torch.FloatTensor(self.hidden_size))
        self.W_q = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.W_ref = nn.Conv1d(self.hidden_size, self.hidden_size, 1, 1)

        self.Encoder=Encoder(cfg)#input_size, d_model, N, heads
        self.final2FC = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(cfg.hidden_size, 2, bias=False))#多目标，产生的Q值有两个
        self._initialize_weights()

    def _initialize_weights(self, init_min=-0.1, init_max=0.1):
        for param in self.parameters():
            nn.init.uniform_(param.data, init_min, init_max)

    def forward(self,  x, mask):
        '''	x: (batch, city_t, 2)
            enc_h: (batch, city_t, embed)
            query(Decoder input): (batch, 1, embed)
            h: (1, batch, embed)
            return: pred_l: (batch)
        '''
        batch=x.size(0)
        x_embed = self.embedding(x)
        e_output = self.Encoder(x_embed,mask)
        query=self.dec_input.repeat(batch,1)
        # query = self.Decoder(dec_input, e_output, mask=None)

        for i in range(2):
            query = self.glimpse(query, e_output)

        pred_l = self.final2FC(query).squeeze(-1).squeeze(-1)
        return pred_l

    def glimpse(self, query, ref):
        """	Args:
            query: the hidden state of the decoder at the current
            (batch, 128)
            ref: the set of hidden states from the encoder.
            (batch, city_t, 128)
        """
        u1 = self.W_q(query).unsqueeze(-1).repeat(1, 1, ref.size(1))  # u1: (batch, 128, city_t)
        u2 = self.W_ref(ref.permute(0, 2, 1))  # u2: (batch, 128, city_t)
        V = self.Vec.unsqueeze(0).unsqueeze(0).repeat(ref.size(0), 1, 1)
        u = torch.bmm(V, torch.tanh(u1 + u2)).squeeze(1)
        # V: (batch, 1, 128) * u1+u2: (batch, 128, city_t) => u: (batch, 1, city_t) => (batch, city_t)
        a = F.softmax(u, dim=1)
        d = torch.bmm(u2, a.unsqueeze(2)).squeeze(2)
        # u2: (batch, 128, city_t) * a: (batch, city_t, 1) => d: (batch, 128)
        return d

class ActorCritic(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.policy=PtrNet1(cfg)
        self.qf=PtrNet2(cfg)

    def forward(self,x,mask,train=True,qf=False,action=None):
        if qf:
            return self.qf(x,mask)
        if action==None:
            if train:
                action,action_logprobs=self.policy(x,mask,train=train,action=None)
                return action,action_logprobs
            else:
                action=self.policy(x,mask,train,action=None)
                return action
        else:
            action_logprobs, dist_entropy=self.policy(x,mask,train=train,action=action)
            return action_logprobs, dist_entropy

    # def qf(self,x,mask):
    #     v=self.qf(x,mask)
    #     return v