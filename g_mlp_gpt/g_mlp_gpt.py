from math import ceil
from functools import partial
from random import randrange
import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange

from g_mlp_gpt.reversible import ReversibleSequence, SequentialSequence

# functions

def exists(val):
    return val is not None

def cast_tuple(val, num):
    return ((val,) * num) if not isinstance(val, tuple) else val

def pad_to_multiple(tensor, multiple, dim = -1, value = 0):
    seqlen = tensor.shape[dim]
    m = seqlen / multiple
    if m.is_integer():
        return tensor
    remainder = ceil(m) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    return F.pad(tensor, (*pad_offset, 0, remainder), value = value)

def dropout_layers(layers, prob_survival):
    if prob_survival == 1:
        return layers

    num_layers = len(layers)
    to_drop = torch.zeros(num_layers).uniform_(0., 1.) > prob_survival

    # make sure at least one layer makes it
    if all(to_drop):
        rand_index = randrange(num_layers)
        to_drop[rand_index] = False

    layers = [layer for (layer, drop) in zip(layers, to_drop) if not drop]
    return layers

# helper classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        inner_dim = int(dim * mult * 2 / 3)

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim * 2),
            GEGLU(),
            nn.Linear(inner_dim, dim)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim_in, dim_out, dim_inner):
        super().__init__()
        self.scale = dim_inner ** -0.5
        self.to_qkv = nn.Linear(dim_in, dim_inner * 3, bias = False)
        self.to_out = nn.Linear(dim_inner, dim_out)

    def forward(self, x):
        device = x.device
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        mask = torch.ones(sim.shape[-2:], device = device).triu(1).bool()
        sim.masked_fill_(mask[None, ...], -torch.finfo(q.dtype).max)

        attn = sim.softmax(dim = -1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        return self.to_out(out)

class CausalSGU(nn.Module):
    def __init__(
        self,
        dim,
        dim_seq,
        init_eps = 1e-3,
        heads = 4,
        act = nn.Identity()
    ):
        super().__init__()
        dim_out = dim // 2

        self.norm = nn.LayerNorm(dim_out)

        self.heads = heads
        self.weight = nn.Parameter(torch.zeros(heads, dim_seq, dim_seq))
        self.bias = nn.Parameter(torch.zeros(heads, dim_seq))

        init_eps /= dim_seq
        nn.init.uniform_(self.weight, -init_eps, init_eps)
        nn.init.constant_(self.bias, 1.)

        self.act = act
        self.register_buffer('mask', ~torch.ones(dim_seq, dim_seq).triu_(1).bool())

    def forward(self, x, gate_res = None):
        device, n, h = x.device, x.shape[1], self.heads

        res, gate = x.chunk(2, dim = -1)
        gate = self.norm(gate)

        weight, bias = self.weight, self.bias
        weight, bias = weight[:, :n, :n], bias[:, :n]

        weight = weight * self.mask[None, :n, :n].int().float()

        gate = rearrange(gate, 'b n (h d) -> b h n d', h = h)
        gate = einsum('b h n d, h m n -> b h m d', gate, weight)
        gate = gate + rearrange(bias, 'h n -> () h n ()')
        gate = rearrange(gate, 'b h n d -> b n (h d)')

        if exists(gate_res):
            gate = gate + gate_res

        return self.act(gate) * res

class CausalLocalSGU(nn.Module):
    def __init__(
        self,
        dim,
        dim_seq,
        init_eps = 1e-3,
        heads = 4,
        window = 128,
        act = nn.Identity()
    ):
        super().__init__()
        dim_out = dim // 2

        self.norm = nn.LayerNorm(dim_out)

        self.heads = heads
        self.window = window
        self.weight = nn.Parameter(torch.zeros(heads, window, window * 2))
        self.bias = nn.Parameter(torch.zeros(heads, window))

        init_eps /= window
        nn.init.uniform_(self.weight, -init_eps, init_eps)
        nn.init.constant_(self.bias, 1.)

        self.act = act
        self.register_buffer('mask', ~torch.ones(window, window * 2).triu_(window + 1).bool())

    def forward(self, x, **kwargs):
        device, n, h, w = x.device, x.shape[1], self.heads, self.window

        x = pad_to_multiple(x, w, dim = -2)
        x = rearrange(x, 'b (w n) d -> b w n d', n = w)

        res, gate = x.chunk(2, dim = -1)
        gate = self.norm(gate)

        gate = F.pad(gate, (0, 0, 0, 0, 1, 0), value = 0.)
        gate = torch.cat((gate[:, :-1], gate[:, 1:]), dim = 2)

        weight, bias = self.weight, self.bias

        weight = weight * self.mask[None, ...].int().float()

        gate = rearrange(gate, 'b w n (h d) -> b w h n d', h = h)
        gate = einsum('b w h n d, h m n -> b w h m d', gate, weight)
        gate = gate + rearrange(bias, 'h n -> () () h n ()')

        gate = rearrange(gate, 'b w h n d -> b w n (h d)')

        out = self.act(gate) * res
        out = rearrange(out, 'b w n d -> b (w n) d')
        return out[:, :n]

class AxiallyFold(nn.Module):
    def __init__(self, dim, every, fn):
        super().__init__()
        self.fn = fn
        self.every = every
        self.conv = nn.Conv1d(dim, dim, kernel_size = every, groups = dim) if every > 1 else None

    def forward(self, x):
        every = self.every
        if every <= 1:
            return self.fn(x)

        n = x.shape[1]
        x = pad_to_multiple(x, self.every, dim = -2)
        x = rearrange(x, 'b (n e) d -> (b e) n d', e = every)
        x = self.fn(x)

        x = rearrange(x, '(b e) n d -> b d (n e)', e = every)
        x = F.pad(x, (every - 1, 0), value = 0)
        out = self.conv(x)
        out = rearrange(out, 'b d n -> b n d')
        return out[:, :n]

class gMLPBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        seq_len,
        dim_ff,
        heads = 4,
        causal = False,
        window = None,
        attn_dim = None,
        act = nn.Identity()
    ):
        super().__init__()
        SGU = partial(CausalLocalSGU, window = window) if exists(window) and window < seq_len else CausalSGU

        self.attn = Attention(dim_in = dim, dim_inner = attn_dim, dim_out = dim_ff // 2) if exists(attn_dim) else None

        self.proj_in = nn.Sequential(
            nn.Linear(dim, dim_ff),
            nn.GELU()
        )
        self.sgu =  SGU(dim_ff, seq_len, causal, heads = heads, act = act)
        self.proj_out = nn.Linear(dim_ff // 2, dim)

    def forward(self, x):
        gate_res = self.attn(x) if exists(self.attn) else None
        x = self.proj_in(x)
        x = self.sgu(x, gate_res = gate_res)
        x = self.proj_out(x)
        return x

# main classes

class gMLPGPT(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        seq_len,
        heads = 1,
        ff_mult = 4,
        prob_survival = 1.,
        reversible = False,
        window = None,
        attn_dim = None,
        act = nn.Identity()
    ):
        super().__init__()
        dim_ff = dim * ff_mult
        self.seq_len = seq_len
        self.prob_survival = prob_survival

        self.to_embed = nn.Embedding(num_tokens, dim)

        window = cast_tuple(window, depth)
        window = tuple(map(lambda t: t if isinstance(t, tuple) else (t, 1), window))

        attn_dims = cast_tuple(attn_dim, depth)

        assert len(window) == depth, f'num window sizes {len(window)} must be equal to depth {depth}'

        layers = nn.ModuleList([])

        for ind, (w, ax), attn_dim in zip(range(depth), window, attn_dims):
            attn_dim = attn_dim if exists(window) else None
            get_gmlp = lambda: PreNorm(dim, AxiallyFold(dim, ax, gMLPBlock(dim = dim, dim_ff = dim_ff, seq_len = seq_len, heads = heads, window = w, act = act, attn_dim = attn_dim)))

            layer_blocks = nn.ModuleList([
                get_gmlp()
            ])

            if reversible:
                layer_blocks.append(FeedForward(dim, mult = ff_mult))

            layers.append(layer_blocks)

        execute_klass = SequentialSequence if not reversible else ReversibleSequence
        self.net = execute_klass(layers)

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        )

    def forward(self, x):
        layer_dropout = 1. - self.prob_survival

        x = self.to_embed(x)
        out = self.net(x, layer_dropout = layer_dropout)
        return self.to_logits(out)
