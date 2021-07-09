import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ipdb import set_trace as debug
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from configs_stock import *

device = torch.device("cuda")
class TimeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, cuda_flag=False, bidirectional=False):
        super(TimeLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.cuda_flag = cuda_flag
        self.W_all = nn.Linear(hidden_size, hidden_size * 4)
        self.U_all = nn.Linear(input_size, hidden_size * 4)
        self.W_d = nn.Linear(hidden_size, hidden_size)
        self.bidirectional = bidirectional

    def forward(self, inputs, timestamps, hidden_states, reverse=False):

        b, seq, embed = inputs.size()
        h = hidden_states[0]
        c = hidden_states[1]

        if self.cuda_flag:
            h = h.cuda()
            c = c.cuda()
        outputs = []
        hidden_state_h = []
        hidden_state_c = []

        for s in range(seq):
            c_s1 = torch.tanh(self.W_d(c))  # short term mem
            # discounted short term mem
            c_s2 = c_s1 * timestamps[:, s: s + 1].expand_as(c_s1)
            c_l = c - c_s1  # long term mem
            c_adj = c_l + c_s2  # adjusted = long + disc short term mem
            outs = self.W_all(h) + self.U_all(inputs[:, s])
            f, i, o, c_tmp = torch.chunk(outs, 4, 1)
            f = torch.sigmoid(f)
            i = torch.sigmoid(i)
            o = torch.sigmoid(o)
            c_tmp = torch.sigmoid(c_tmp)
            c = f * c_adj + i * c_tmp
            h = o * torch.tanh(c)
            outputs.append(o)
            hidden_state_c.append(c)
            hidden_state_h.append(h)

        if reverse:
            outputs.reverse()
            hidden_state_c.reverse()
            hidden_state_h.reverse()

        outputs = torch.stack(outputs, 1)
        hidden_state_c = torch.stack(hidden_state_c, 1)
        hidden_state_h = torch.stack(hidden_state_h, 1)

        return outputs, (h, c)

class attn(torch.nn.Module):
    def __init__(self, in_shape, use_attention=True, maxlen=None):
        super(attn, self).__init__()
        self.use_attention = use_attention
        if self.use_attention:
            self.W1 = torch.nn.Linear(in_shape, in_shape).to(device)
            self.W2 = torch.nn.Linear(in_shape, in_shape).to(device)
            self.V = torch.nn.Linear(in_shape, 1).to(device)
        if maxlen != None:
            self.arange = torch.arange(maxlen).to(device)

    def forward(self, full, last, lens=None, dim=1):
        """
        full : B*30*in_shape
        last : B*1*in_shape
        lens: B*1
        """
        if self.use_attention:
            score = self.V(F.tanh(self.W1(last) + self.W2(full)))
            # print(score.shape) -> B*30*1

            if lens != None:
                mask = self.arange[None, :] < lens[:, None]  # B*30
                score[~mask] = float("-inf")

            attention_weights = F.softmax(score, dim=dim)
            context_vector = attention_weights * full
            context_vector = torch.sum(context_vector, dim=dim)
            return context_vector  # B*in_shape
        else:
            if lens != None:
                mask = self.arange[None, :] < lens[:, None]  # B*30
                mask = mask.type(torch.float).unsqueeze(-1).cuda()
                context_vector = full * mask
                context_vector = torch.mean(context_vector, dim=dim)
                return context_vector
            else:
                return torch.mean(full, dim=dim)

class Actor(nn.Module):
    """
    Actor:
        Gets the text: news/tweets about the stocks,
        current balance, price and holds on the stocks.
    """

    def __init__(
        self,
        num_stocks=STOCK_DIM,
        text_embed_dim=TWEETS_EMB,
        intraday_hiddenDim=128,
        interday_hiddenDim=128,
        intraday_numLayers=1,
        interday_numLayers=1,
        use_attn1=False,
        use_attn2=False,
        maxlen=30,
        device=torch.device("cuda"),
    ):
        """
        num_stocks: number of stocks for which the agent is trading
        """
        super(Actor, self).__init__()

        self.lstm1s = [
            TimeLSTM(
                input_size=text_embed_dim,
                hidden_size=intraday_hiddenDim,
            )
            for _ in range(num_stocks)
        ]

        for i, tweet_lstm in enumerate(self.lstm1s):
            self.add_module("lstm1_{}".format(i), tweet_lstm)

        self.lstm1_outshape = intraday_hiddenDim
        self.lstm2_outshape = interday_hiddenDim

        self.attn1s = [
            attn(self.lstm1_outshape, maxlen=maxlen)
            for _ in range(num_stocks)
        ]

        for i, tweet_attn in enumerate(self.attn1s):
            self.add_module("attn1_{}".format(i), tweet_attn)

        self.lstm2s = [
            nn.LSTM(
                input_size=self.lstm1_outshape,
                hidden_size=interday_hiddenDim,
                num_layers=interday_numLayers,
                batch_first=True,
                bidirectional=False,
            )
            for _ in range(num_stocks)
        ]
        for i, day_lstm in enumerate(self.lstm2s):
            self.add_module("lstm2_{}".format(i), day_lstm)

        self.attn2s = [
            attn(self.lstm2_outshape)
            for _ in range(num_stocks)
        ]

        for i, day_attn in enumerate(self.attn2s):
            self.add_module("attn2_{}".format(i), day_attn)

        self.linearx1 = [
            nn.Linear(self.lstm2_outshape, self.lstm2_outshape)
            for _ in range(num_stocks)
        ]
        for i, linear_x in enumerate(self.linearx1):
            self.add_module("linearx1_{}".format(i), linear_x)

        self.linearx2 = [nn.Linear(self.lstm2_outshape, 64)
                         for _ in range(num_stocks)]
        for i, linear_x in enumerate(self.linearx2):
            self.add_module("linearx2_{}".format(i), linear_x)

        self.drop = nn.Dropout(p=0.3)
        self.softmax1 = nn.Softmax(dim=2)
        self.softmax2 = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

        self.device = device
        self.maxlen = maxlen

        self.linear1 = nn.Linear(2 * num_stocks + 1, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear_c = nn.Linear(64 * num_stocks + 32, num_stocks)
        self.tanh = nn.Tanh()
        self.num_stocks = num_stocks
        self.device = device

    def init_hidden(self):
        h = Variable(torch.zeros(self.bs, self.lstm1_outshape)).to(self.device)
        c = Variable(torch.zeros(self.bs, self.lstm2_outshape)).to(self.device)

        return (h, c)

    def forward(self, state):
        state = state.view(-1, FEAT_DIMS)
        stock_feats = state[:, 0: 2 * self.num_stocks + 1].view(
            -1, 2 * self.num_stocks + 1
        )
        sentence_feat = state[:, EMB_IDX:LEN_IDX].view(
            -1, self.num_stocks, N_DAYS, MAX_TWEETS, TWEETS_EMB
        )
        len_tweets = state[:,
                           LEN_IDX:TARGET_IDX].view(-1, self.num_stocks, N_DAYS)
        time_feats = state[:,
                           TIME_IDX:].view(-1, self.num_stocks, N_DAYS, MAX_TWEETS)

        self.bs = sentence_feat.size(0)
        sentence_feat = sentence_feat.permute(1, 2, 0, 3, 4)
        len_tweets = len_tweets.permute(1, 2, 0)
        time_feats = time_feats.permute(1, 2, 0, 3)

        num_days = N_DAYS
        text_out = torch.zeros(self.num_stocks, self.bs, 64).to(self.device)
        for i in range(self.num_stocks):
            h_init, c_init = self.init_hidden()

            lstm1_out = torch.zeros(num_days, self.bs, self.lstm1_outshape).to(
                self.device
            )
            for j in range(num_days):
                temp_sent = sentence_feat[i, j, :, :, :]
                temp_len = len_tweets[i, j, :]
                temp_timefeats = time_feats[i, j, :, :]

                temp_lstmout, (_, _) = self.lstm1s[i](
                    temp_sent, temp_timefeats, (h_init, c_init)
                )

                last_idx = temp_len.type(torch.int).tolist()
                temp_hn = torch.zeros(self.bs, self.lstm1_outshape).to(self.device)
                for k in range(self.bs):
                    if last_idx[k] != 0:
                        temp_hn[k] = temp_lstmout[k, last_idx[k] - 1, :]
                lstm1_out[j] = self.attn1s[i](temp_lstmout, temp_hn, temp_len.to(self.device))

            lstm1_out = lstm1_out.permute(1, 0, 2)
            lstm2_out, (h2_out, _) = self.lstm2s[i](lstm1_out)
            h2_out = h2_out.permute(1, 0, 2)
            x = self.attn2s[i](lstm2_out, h2_out)
            x = self.drop(self.relu(self.linearx1[i](x)))
            x = self.linearx2[i](x)
            text_out[i] = x

        text_out = text_out.permute(1, 0, 2)
        text_out = text_out.view(self.bs, -1)
        x_stock = self.relu(self.linear1(stock_feats))
        x_stock = self.linear2(x_stock)

        full = torch.cat([x_stock, text_out], dim=1)
        full = self.tanh(self.linear_c(full))
        return full

class Critic(nn.Module):
    """
    Actor:
        Gets the text tweets about the stocks,
        current balance, price and holds on the stocks.
    """

    def __init__(
        self,
        num_stocks=STOCK_DIM,
        text_embed_dim=TWEETS_EMB,
        intraday_hiddenDim=128,
        interday_hiddenDim=128,
        intraday_numLayers=1,
        interday_numLayers=1,
        use_attn1=False,
        use_attn2=False,
        maxlen=30,
        device=torch.device("cuda"),
    ):
        """
        num_stocks: number of stocks for which the agent is trading
        """
        super(Critic, self).__init__()

        self.lstm1s = [
            TimeLSTM(
                input_size=text_embed_dim,
                hidden_size=intraday_hiddenDim,
            )
            for _ in range(num_stocks)
        ]

        for i, tweet_lstm in enumerate(self.lstm1s):
            self.add_module("lstm1_{}".format(i), tweet_lstm)

        self.lstm1_outshape = intraday_hiddenDim
        self.lstm2_outshape = interday_hiddenDim

        self.attn1s = [
            attn(self.lstm1_outshape, maxlen=maxlen)
            for _ in range(num_stocks)
        ]

        for i, tweet_attn in enumerate(self.attn1s):
            self.add_module("attn1_{}".format(i), tweet_attn)

        self.lstm2s = [
            nn.LSTM(
                input_size=self.lstm1_outshape,
                hidden_size=interday_hiddenDim,
                num_layers=interday_numLayers,
                batch_first=True,
                bidirectional=False,
            )
            for _ in range(num_stocks)
        ]
        for i, day_lstm in enumerate(self.lstm2s):
            self.add_module("lstm2_{}".format(i), day_lstm)

        self.attn2s = [
            attn(self.lstm2_outshape)
            for _ in range(num_stocks)
        ]

        for i, day_attn in enumerate(self.attn2s):
            self.add_module("attn2_{}".format(i), day_attn)

        self.linearx1 = [
            nn.Linear(self.lstm2_outshape, self.lstm2_outshape)
            for _ in range(num_stocks)
        ]
        for i, linear_x in enumerate(self.linearx1):
            self.add_module("linearx1_{}".format(i), linear_x)

        self.linearx2 = [nn.Linear(self.lstm2_outshape, 64)
                         for _ in range(num_stocks)]
        for i, linear_x in enumerate(self.linearx2):
            self.add_module("linearx2_{}".format(i), linear_x)

        self.drop = nn.Dropout(p=0.3)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.num_stocks = num_stocks
        self.softmax1 = nn.Softmax(dim=2)
        self.softmax2 = nn.Softmax(dim=1)

        self.device = device
        self.maxlen = maxlen

        self.linear1 = nn.Linear(2 * num_stocks + 1, 64)
        self.linear2 = nn.Linear(64, 32)

        self.linear_c = nn.Linear(64 * num_stocks + 32, 32)

        # * Critic Layers
        self.linear_critic = nn.Linear(num_stocks, 32)

        # * Actions and States
        self.linear_sa1 = nn.Linear(64, 32)
        self.linear_sa2 = nn.Linear(32, 1)
        self.device = device
    def init_hidden(self):
        h = Variable(torch.zeros(self.bs, self.lstm1_outshape)).to(self.device)
        c = Variable(torch.zeros(self.bs, self.lstm2_outshape)).to(self.device)

        return (h, c)

    def forward(self, state, actions):
        state = state.view(-1, FEAT_DIMS)
        actions = actions.view(-1, STOCK_DIM)

        stock_feats = state[:, 0: 2 * self.num_stocks + 1].view(
            -1, 2 * self.num_stocks + 1
        )
        sentence_feat = state[:, EMB_IDX:LEN_IDX].view(
            -1, self.num_stocks, N_DAYS, MAX_TWEETS, TWEETS_EMB
        )
        len_tweets = state[:,
                           LEN_IDX:TARGET_IDX].view(-1, self.num_stocks, N_DAYS)
        time_feats = state[:,
                           TIME_IDX:].view(-1, self.num_stocks, N_DAYS, MAX_TWEETS)

        self.bs = sentence_feat.size(0)
        sentence_feat = sentence_feat.permute(1, 2, 0, 3, 4)
        len_tweets = len_tweets.permute(1, 2, 0)
        time_feats = time_feats.permute(1, 2, 0, 3)
        num_days = N_DAYS
        text_out = torch.zeros(self.num_stocks, self.bs, 64).to(self.device)
        for i in range(self.num_stocks):
            h_init, c_init = self.init_hidden()
            lstm1_out = torch.zeros(num_days, self.bs, self.lstm1_outshape).to(
                self.device
            )
            for j in range(num_days):
                temp_sent = sentence_feat[i, j, :, :, :]
                temp_len = len_tweets[i, j, :]
                temp_timefeats = time_feats[i, j, :, :]

                temp_lstmout, (_, _) = self.lstm1s[i](
                    temp_sent, temp_timefeats, (h_init, c_init)
                )
                last_idx = temp_len.type(torch.int).tolist()
                temp_hn = torch.zeros(self.bs, self.lstm1_outshape).to(self.device)
                for k in range(self.bs):
                    if last_idx[k] != 0:
                        temp_hn[k] = temp_lstmout[k, last_idx[k] - 1, :]

                lstm1_out[j] = self.attn1s[i](temp_lstmout, temp_hn, temp_len.to(self.device))

            lstm1_out = lstm1_out.permute(1, 0, 2)
            lstm2_out, (h2_out, _) = self.lstm2s[i](lstm1_out)
            h2_out = h2_out.permute(1, 0, 2)
            x = self.attn2s[i](lstm2_out, h2_out)

            x = self.drop(self.relu(self.linearx1[i](x)))
            x = self.linearx2[i](x)
            text_out[i] = x

        text_out = text_out.permute(1, 0, 2)
        text_out = text_out.view(self.bs, -1)

        x_stock = self.relu(self.linear1(stock_feats))
        x_stock = self.linear2(x_stock)

        full = torch.cat([x_stock, text_out], dim=1)
        full = self.linear_c(full)

        actions = self.linear_critic(actions)

        full = torch.cat([full, actions], dim=1)
        full = self.relu(self.linear_sa1(full))
        full = self.linear_sa2(full)

        return full