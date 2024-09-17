import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, query, key, value, mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        output = torch.matmul(attention, value)
        return output, attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(self.d_k)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)


        query = self.q_linear(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        key = self.k_linear(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        value = self.v_linear(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        x, attn = self.attention(query, key, value, mask=mask)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        x = self.out(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

# 编码器层
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x2 = self.norm1(x)
        x = x + self.dropout(self.self_attn(x2, x2, x2, mask))
        x2 = self.norm2(x)
        x = x + self.dropout(self.ffn(x2))
        return x

# 编码器
class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, num_layers, dropout=0.1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, src, mask=None):
        x = src
        for layer in self.layers:
            x = layer(x, mask)
        return x

# Transformer
class Transformer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, num_layers, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(d_model, n_heads, d_ff, num_layers, dropout)

    def forward(self, src, mask=None):
        enc_output = self.encoder(src, mask)
        return enc_output