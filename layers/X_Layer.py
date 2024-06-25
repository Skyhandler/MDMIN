import math
import torch
import torch.nn.functional as F
from torch import nn

from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer, ProbAttention


# 某条件下的填充长度
def get_P(seq_len:int, kernel_size:int, stride:int):
    if seq_len < kernel_size:
        return kernel_size - seq_len
    else:
        return (stride - (seq_len - kernel_size) % stride) % stride


# 第 i 层卷积后的长度
def get_N(i:int, seq_len:int, kernel_size:int, stride:int):

    P = get_P(seq_len, kernel_size, stride)
    N = int((seq_len + P - kernel_size) / stride + 1)

    if i == 0:
        return N
    else:
        return get_N(i-1, N, kernel_size, stride)


# 多层 1D 卷积
class ML_Conv1d(nn.Module):          
    def __init__(self, seq_len:int, layers_num:int, hidden_channels:int, out_channels:int, kernel_size:int, stride:int):
        
        super(ML_Conv1d, self).__init__()

        self.seq_len = seq_len
        self.layers_num = layers_num
        self.kernel_size = kernel_size
        self.stride = stride

        self.layers = nn.ModuleList()
        for i in range(layers_num):
            in_c = 16
            out_c = 16
            P = 0

            if (i == 0) and (layers_num == 1):
                in_c = 1
                out_c = out_channels
            elif (i == 0) and (layers_num != 1):
                in_c = 1
                out_c = hidden_channels
            elif i == layers_num-1:
                in_c = hidden_channels
                out_c = out_channels
            else:
                in_c = hidden_channels
                out_c = hidden_channels

            self.layers.append(nn.Conv1d(in_channels=in_c, out_channels=out_c, kernel_size=kernel_size, stride=stride))


    def forward(self, x):                               # x: [B, C, L]

        B, C, L = x.shape[0], x.shape[1], x.shape[2]

        x = x.reshape(B*C, 1, L)                        # x: [B, C, L] -> [B*C, 1, L]

        for i in range(self.layers_num):                # x: [B*C, O, N]
            if i == 0:
                P = get_P(self.seq_len, self.kernel_size, self.stride)
            else:
                P = get_P(get_N(i-1, self.seq_len, self.kernel_size, self.stride), self.kernel_size, self.stride)
            x = F.pad(x, pad=(0, P), mode='replicate')
            x = self.layers[i](x)
        
        x = x.permute(0, 2, 1)                          # x: [B*C, N, O]
    
        return x
    
    

class X_Layer(nn.Module):          
    def __init__(self, seq_len:int, pred_len:int, e_layers:int, layers_num:int, hidden_channels:int, out_channels:int, stride1, stride2, stride3,
                 d_model, dropout, factor, output_attention, n_heads, d_ff, activation,
                 pe='sincos', learn_pe=True):
        
        super(X_Layer, self).__init__()

        self.d_model = d_model
        kernel_sets = [2*stride1, 2*stride2, 2*stride3]
        stride_sets = [stride1, stride2, stride3]


        self.models = nn.ModuleList([ML_Conv1d(seq_len, layers_num, hidden_channels, out_channels, kernel_sets[i], stride_sets[i]) 
                                              for i in range(len(kernel_sets))]
        )

        self.embedding = nn.Linear(out_channels, d_model)

        N_sets = [get_N(layers_num-1, seq_len, kernel_sets[i], stride_sets[i])
                        for i in range(len(kernel_sets))]

        N = sum(N_sets)

        self.W_pos = positional_encoding(pe, learn_pe, N, d_model)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=output_attention), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.gate = nn.Linear(N, 1)

        self.leaky_relu = nn.LeakyReLU()

        self.softmax = nn.Softmax(dim=-1)

        self.sigmoid = nn.Sigmoid()

        self.dropout = nn.Dropout(dropout)

        self.flatten = nn.Flatten(start_dim=-2)

        self.linear = nn.Linear(N*d_model, pred_len)


    def forward(self, x):                               # x: [B, C, L]

        B, C, L = x.shape[0], x.shape[1], x.shape[2]

        out = []

        for model in self.models:
            out.append(model(x))

        x = torch.cat(out, dim=-2)                      # x: [B*C, N0+N1+N2, O]

        x = self.embedding(x)                           # x: [B*C, N, D]

        x = x.permute(0,2,1)                            # x: [B*C, D, N]

        gate_score = self.softmax(self.sigmoid(self.gate(x)).reshape(B*C, -1)).unsqueeze(2) # x: [B*C, D, 1]

        x = (x*gate_score).permute(0,2,1)

        x = self.dropout(x + self.W_pos)                # x: [B*C, N, D]

        x, _ = self.encoder(x)                          # x: [B*C, N, D]

        x = x.reshape(B, C, -1, self.d_model)           # x: [B, C, N, D]

        x = self.flatten(x)                             # x: [B, C, N*D]

        x = self.linear(x)                              # x: [B, C, T]

        return x
    




def PositionalEncoding(q_len, d_model, normalize=True):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe * d_model / q_len
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe

SinCosPosEncoding = PositionalEncoding

def Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True, eps=1e-3, verbose=False):
    x = .5 if exponential else 1
    i = 0
    for i in range(100):
        cpe = 2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** x) * (torch.linspace(0, 1, d_model).reshape(1, -1) ** x) - 1
        pv(f'{i:4.0f}  {x:5.3f}  {cpe.mean():+6.3f}', verbose)
        if abs(cpe.mean()) <= eps: break
        elif cpe.mean() > eps: x += .001
        else: x -= .001
        i += 1
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe

def Coord1dPosEncoding(q_len, exponential=False, normalize=True):
    cpe = (2 * (torch.linspace(0, 1, q_len).reshape(-1, 1)**(.5 if exponential else 1)) - 1)
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


def positional_encoding(pe, learn_pe, q_len, d_model):
    # Positional encoding
    if pe == None:
        W_pos = torch.empty((q_len, d_model)) # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'zeros':
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'normal' or pe == 'gauss':
        W_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == 'lin1d': W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
    elif pe == 'exp1d': W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
    elif pe == 'lin2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True)
    elif pe == 'exp2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=True, normalize=True)
    elif pe == 'sincos': W_pos = PositionalEncoding(q_len, d_model, normalize=True)
    else: raise ValueError(f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)")
    return nn.Parameter(W_pos, requires_grad=learn_pe)