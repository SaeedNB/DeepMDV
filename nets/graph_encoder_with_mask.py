import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GraphWithMaskEncoder(nn.Module):
    def __init__(self, n_heads, embed_dim, n_layers, node_dim=None, normalization='batch', feed_forward_hidden=512):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(n_heads, embed_dim, feed_forward_hidden) for _ in range(n_layers)])

    def forward(self, h, mask, distance=None):

        for layer in self.layers:
            h = layer(h, mask)

        return h
        # shape: (batch, problem+1, embedding)


class EncoderLayer(nn.Module):
    def __init__(self, n_heads, embed_dim, feed_forward_hidden=512):
        super().__init__()
        qkv_dim = embed_dim // n_heads
        self.head_num = n_heads
        self.Wq = nn.Linear(embed_dim, n_heads * qkv_dim, bias=False)
        self.Wk = nn.Linear(embed_dim, n_heads * qkv_dim, bias=False)
        self.Wv = nn.Linear(embed_dim, n_heads * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(n_heads * qkv_dim, embed_dim)
        self.norm_factor = 1 / math.sqrt(qkv_dim)
        self.add_n_normalization_1 = AddAndInstanceNormalization(embed_dim)
        self.feed_forward = FeedForward(embed_dim, feed_forward_hidden)
        self.add_n_normalization_2 = AddAndInstanceNormalization(embed_dim)

    def forward(self, input1, mask):

        q = self.reshape_by_heads(self.Wq(input1), head_num=self.head_num)
        k = self.reshape_by_heads(self.Wk(input1), head_num=self.head_num)
        v = self.reshape_by_heads(self.Wv(input1), head_num=self.head_num)
        # qkv shape: (batch, head_num, problem, qkv_dim)

        out_concat = self.multi_head_attention(q, k, v, mask)
        # shape: (batch, problem, head_num*qkv_dim)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem, embedding)

        out1 = self.add_n_normalization_1(input1, multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)

        return out3
        # shape: (batch, problem, embedding)


    def reshape_by_heads(self, qkv, head_num):
        # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

        batch_s = qkv.size(0)
        n = qkv.size(1)

        q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
        # shape: (batch, n, head_num, key_dim)

        q_transposed = q_reshaped.transpose(1, 2)
        # shape: (batch, head_num, n, key_dim)

        return q_transposed

    def multi_head_attention(self, q, k, v, mask=None, rank2_ninf_mask=None, rank3_ninf_mask=None):
        batch_s = q.size(0)
        head_num = q.size(1)
        n = q.size(2)
        key_dim = q.size(3)

        input_s = k.size(2)

        score_scaled = self.norm_factor * torch.matmul(q, k.transpose(2, 3))
        # shape: (batch, head_num, n, problem)

        # score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
        if mask is not None:
            score_scaled[mask[:, None,:, :].expand(batch_s, head_num, n, input_s)] = -math.inf

        if rank2_ninf_mask is not None:
            score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
        if rank3_ninf_mask is not None:
            score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

        weights = nn.Softmax(dim=3)(score_scaled)
        # shape: (batch, head_num, n, problem)

        out = torch.matmul(weights, v)
        # shape: (batch, head_num, n, key_dim)

        out_transposed = out.transpose(1, 2)
        # shape: (batch, n, head_num, key_dim)

        out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
        # shape: (batch, n, head_num*key_dim)

        return out_concat


class AddAndInstanceNormalization(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        added = input1 + input2
        # shape: (batch, problem, embedding)

        transposed = added.transpose(1, 2)
        # shape: (batch, embedding, problem)

        normalized = self.norm(transposed)
        # shape: (batch, embedding, problem)

        back_trans = normalized.transpose(1, 2)
        # shape: (batch, problem, embedding)

        return back_trans


class AddAndBatchNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm_by_EMB = nn.BatchNorm1d(embedding_dim, affine=True)
        # 'Funny' Batch_Norm, as it will normalized by EMB dim

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        batch_s = input1.size(0)
        problem_s = input1.size(1)
        embedding_dim = input1.size(2)

        added = input1 + input2
        normalized = self.norm_by_EMB(added.reshape(batch_s * problem_s, embedding_dim))
        back_trans = normalized.reshape(batch_s, problem_s, embedding_dim)

        return back_trans


class FeedForward(nn.Module):
    def __init__(self, embedding_dim, ff_hidden_dim):
        super().__init__()

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))
