import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Models_t import Encoder, Decoder
from torch.distributions import Categorical, Normal

class CriticNetwork(nn.Module):
    def __init__(self, args, alpha, chkpt_dir,name ):
        super(CriticNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.fc1 = nn.Linear(args.input_size*args.max_job+1, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)

        self.q = nn.Linear(args.hidden_size, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
 
        self.to(self.device)



    def forward(self, state, action):
        x = F.relu(self.fc1(T.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))

        q = self.q(x)

        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))


class CriticNetwork_dqn2019(nn.Module):
    def __init__(self, args, alpha, chkpt_dir, name):
        super(CriticNetwork_dqn2019, self).__init__()
        self.chkpt_file = os.path.join(chkpt_dir, name)
        # self.state_size = state_size
        self.action_size = args.max_job
        self.reward_size = args.reward_size

        self.affine1 = nn.Linear(args.input_size*args.max_job+ args.reward_size,
                                 (args.input_size + args.reward_size) * 16)
        self.affine2 = nn.Linear((args.input_size + args.reward_size) * 16,
                                 (args.input_size + args.reward_size) * 32)
        self.affine3 = nn.Linear((args.input_size + args.reward_size) * 32,
                                 (args.input_size + args.reward_size) * 64)
        self.affine4 = nn.Linear((args.input_size + args.reward_size) * 64,
                                 (args.input_size + args.reward_size) * 32)
        self.affine5 = nn.Linear((args.input_size + args.reward_size) * 32,
                                 args.max_job)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)


    def forward(self, state,mask, preference, inf=1e9):
        x = T.cat((state, preference), dim=1)
        x = x.view(x.size(0), -1)
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        x = F.relu(self.affine3(x))
        x = F.relu(self.affine4(x))
        q = self.affine5(x)
        q = q + inf * mask

        hq = q.detach().max(dim=1)[0]

        return hq, q


    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))


class CriticNetwork_AC(nn.Module):
    def __init__(self, args, alpha, chkpt_dir, name):
        super(CriticNetwork_AC, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.fc1 = nn.Linear(args.input_size * args.max_job, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)

        self.q = nn.Linear(args.hidden_size, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        q = self.q(x)

        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, 
                 n_actions,  chkpt_dir, scenario, name):
        super(ActorNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir,scenario, name)

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)

        self.pi = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
 
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        pi = T.softmax(self.pi(x), dim=1)

        return pi

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))


class ActorNetwork_RNN(nn.Module):
    def __init__(self, alpha, input_size, hidden_size, n_actions, chkpt_dir, scenario, name,num_layers=1):
        super(ActorNetwork_RNN, self).__init__()
        # 1 init函数 准备三个层 self.rnn self.linear self.softmax=nn.LogSoftmax(dim=-1)
        self.chkpt_file = os.path.join(chkpt_dir, scenario, name)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = n_actions
        self.num_layers = num_layers

        # 定义rnn层
        self.rnn = nn.RNN(self.input_size, self.hidden_size, self.num_layers)
        # 定义linear层（全连接线性层）
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        # 定义softmax层
        self.softmax = nn.LogSoftmax(dim=-1)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, input, hidden):
        # 让数据经过三个层 返回softmax结果和hn
        # 数据形状 [6,57] -> [6,1,57]
        input = input.unsqueeze(1)
        # 把数据送给模型 提取事物特征
        # 数据形状 [seqlen,1,57],[1,1,128]) -> [seqlen,1,18],[1,1,128]
        rr, hn = self.rnn(input, hidden)
        # 数据形状 [seqlen,1,128] - [1, 128]
        tmprr = rr[-1]##只保留了最后一个的输出结果
        tmprr = self.linear(tmprr)
        return self.softmax(tmprr), hn

    def inithidden(self):
        # 初始化隐藏层输入数据 inithidden()
        return T.zeros(self.num_layers, 1,self.hidden_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        pi = T.softmax(self.pi(x), dim=1)

        return pi

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))


class ActorNetwork_1(nn.Module):
    def __init__(self, cfg,alpha,chkpt_dir, name):
        super().__init__()
        self.device = cfg.device
        self.args = cfg
        self.hidden_size=cfg.hidden_size
        self.chkpt_file = os.path.join(chkpt_dir, name)
        self.embedding=nn.Linear(cfg.input_size,cfg.embedding_size)
        self.Vec = nn.Parameter(T.FloatTensor(self.hidden_size))
        self.Vec2 = nn.Parameter(T.FloatTensor(self.hidden_size))
        self.W_q = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.W_ref = nn.Conv1d(self.hidden_size , self.hidden_size, 1, 1)
        self.dec_input=nn.Parameter(T.FloatTensor(self.hidden_size))
        self.W_q2 = nn.Linear(self.hidden_size , self.hidden_size, bias=True)
        self.W_ref2 = nn.Conv1d(self.hidden_size, self.hidden_size, 1, 1)
        self.softmax_T = 1.5  # cfg.softmax_T

        self.Encoder=Encoder(cfg)#input_size, d_model, N, heads
        # self.Decoder=Decoder(cfg)#d_model, N, heads
        self._initialize_weights()
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def _initialize_weights(self, init_min=-0.1, init_max=0.1):
        for param in self.parameters():
            nn.init.uniform_(param.data, init_min, init_max)
    #
    # def make_mask(self,total_operation,max_operation):
    #     mask = T.ones(total_operation)
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
        # log_p = T.log_softmax(logits, dim=-1) #(batch,total_operations,job_num)
        probs=T.softmax(probs,dim=-1)
        dist=Categorical(probs)
        if action==None:
            if train:
                action=dist.sample().long()
                return action,dist.log_prob(action)
            else:
                action=T.argmax(probs, dim=1).long()
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
        u = T.bmm(V, T.tanh(u1 + u2)).squeeze(1)
        # V: (batch, 1, 128) * u1+u2: (batch, 128, city_t) => u: (batch, 1, city_t) => (batch, city_t)
        u = u + inf * mask
        a = F.softmax(u / self.softmax_T, dim=1)
        d = T.bmm(u2, a.unsqueeze(2)).squeeze(2)
        # # u2: (batch, 128, city_t) * a: (batch, city_t, 1) => d: (batch, 128)
        return d

    def pointer(self, query, ref, mask, inf=1e9):
        u1 = self.W_q2(query).squeeze(1).unsqueeze(-1).repeat(1, 1, ref.size(1))  # u1: (batch, 128, city_t)
        u2 = self.W_ref2(ref.permute(0, 2, 1))  # u2: (batch, 128, city_t)
        V = self.Vec2.unsqueeze(0).unsqueeze(0).repeat(ref.size(0), 1, 1)
        u = T.bmm(V, self.args.clip_logits * T.tanh(u1 + u2)).squeeze(1)
        # V: (batch, 1, 128) * u1+u2: (batch, 128, city_t) => u: (batch, 1, city_t) => (batch, city_t)
        u = u + inf * mask
        return u
    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))
    # def get_log_likelihood(self, _log_p, pi):
    #     """	args:
    #         _log_p: (batch, city_t, city_t)
    #         pi: (batch, city_t), predicted tour
    #         return: (batch)
    #     """
    #     log_p = T.gather(input=_log_p, dim=2, index=pi[:, :, None])
    #     return T.sum(log_p.squeeze(-1), 1)


class CriticNetwork_dqn(nn.Module):
    def __init__(self, cfg,alpha,chkpt_dir, name):
        super().__init__()
        self.device = cfg.device
        self.args = cfg
        self.hidden_size=cfg.hidden_size
        self.chkpt_file = os.path.join(chkpt_dir, name)
        self.embedding=nn.Linear(cfg.input_size,cfg.embedding_size)
        self.Vec = nn.Parameter(T.FloatTensor(self.hidden_size))
        self.Vec2 = nn.Parameter(T.FloatTensor(self.hidden_size))
        self.W_q = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.W_ref = nn.Conv1d(self.hidden_size , self.hidden_size, 1, 1)
        self.dec_input=nn.Parameter(T.FloatTensor(self.hidden_size))
        self.W_q2 = nn.Linear(self.hidden_size , self.hidden_size, bias=True)
        self.W_ref2 = nn.Conv1d(self.hidden_size, self.hidden_size, 1, 1)
        self.softmax_T = 1.5  # cfg.softmax_T

        self.Encoder=Encoder(cfg)#input_size, d_model, N, heads
        # self.Decoder=Decoder(cfg)#d_model, N, heads
        self._initialize_weights()
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def _initialize_weights(self, init_min=-0.1, init_max=0.1):
        for param in self.parameters():
            nn.init.uniform_(param.data, init_min, init_max)
    #
    # def make_mask(self,total_operation,max_operation):
    #     mask = T.ones(total_operation)
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
        # log_p = T.log_softmax(logits, dim=-1) #(batch,total_operations,job_num)
        probs=T.softmax(probs,dim=-1)
        # dist=Categorical(probs)
        # if action==None:
        #     if train:
        #         action=dist.sample().long()
        #         return action,dist.log_prob(action)
        #     else:
        #         action=T.argmax(probs, dim=1).long()
        #     return action
        #
        # else:
        #     action_logprobs=dist.log_prob(action)
        #     dist_entropy=dist.entropy()
        return probs

    # 注意力网络  u1：query   u2：key  ref:value
    def glimpse(self, query, ref, mask, inf=1e9):
        u1 = self.W_q(query).unsqueeze(-1).repeat(1, 1, ref.size(1))  # u1: (batch, 128, city_t)
        u2 = self.W_ref(ref.permute(0, 2, 1))  # u2: (batch, 128, city_t)
        V = self.Vec.unsqueeze(0).unsqueeze(0).repeat(ref.size(0), 1, 1)
        u = T.bmm(V, T.tanh(u1 + u2)).squeeze(1)
        # V: (batch, 1, 128) * u1+u2: (batch, 128, city_t) => u: (batch, 1, city_t) => (batch, city_t)
        u = u + inf * mask
        a = F.softmax(u / self.softmax_T, dim=1)
        d = T.bmm(u2, a.unsqueeze(2)).squeeze(2)
        # # u2: (batch, 128, city_t) * a: (batch, city_t, 1) => d: (batch, 128)
        return d

    def pointer(self, query, ref, mask, inf=1e9):
        u1 = self.W_q2(query).squeeze(1).unsqueeze(-1).repeat(1, 1, ref.size(1))  # u1: (batch, 128, city_t)
        u2 = self.W_ref2(ref.permute(0, 2, 1))  # u2: (batch, 128, city_t)
        V = self.Vec2.unsqueeze(0).unsqueeze(0).repeat(ref.size(0), 1, 1)
        u = T.bmm(V, self.args.clip_logits * T.tanh(u1 + u2)).squeeze(1)
        # V: (batch, 1, 128) * u1+u2: (batch, 128, city_t) => u: (batch, 1, city_t) => (batch, city_t)
        u = u + inf * mask
        return u
    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))
    # def get_log_likelihood(self, _log_p, pi):
    #     """	args:
    #         _log_p: (batch, city_t, city_t)
    #         pi: (batch, city_t), predicted tour
    #         return: (batch)
    #     """
    #     log_p = T.gather(input=_log_p, dim=2, index=pi[:, :, None])
    #     return T.sum(log_p.squeeze(-1), 1)


class CriticNetwork_1(nn.Module):
    def __init__(self, cfg,alpha,chkpt_dir, name):
        super().__init__()
        self.device = cfg.device
        self.args = cfg
        self.hidden_size=cfg.hidden_size
        self.chkpt_file = os.path.join(chkpt_dir, name)
        self.embedding=nn.Linear(cfg.input_size,cfg.embedding_size)
        self.dec_input = nn.Parameter(T.FloatTensor(self.hidden_size))
        self.Vec = nn.Parameter(T.FloatTensor(self.hidden_size))
        self.W_q = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.W_ref = nn.Conv1d(self.hidden_size, self.hidden_size, 1, 1)

        self.Encoder=Encoder(cfg)#input_size, d_model, N, heads
        self.final2FC = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(cfg.hidden_size, 1, bias=False))
        self._initialize_weights()
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

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
        u = T.bmm(V, T.tanh(u1 + u2)).squeeze(1)
        # V: (batch, 1, 128) * u1+u2: (batch, 128, city_t) => u: (batch, 1, city_t) => (batch, city_t)
        a = F.softmax(u, dim=1)
        d = T.bmm(u2, a.unsqueeze(2)).squeeze(2)
        # u2: (batch, 128, city_t) * a: (batch, city_t, 1) => d: (batch, 128)
        return d
