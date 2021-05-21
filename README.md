## GPT - gMLP

This repository will attempt to crack long context autoregressive language modeling (GPT) using variations of <a href="https://arxiv.org/abs/2105.08050">gMLPs</a>. Specifically, it will contain a variant that does gMLP for local sliding windows. The hope is to be able to stretch a single GPU to be able to train context lengths of 4096 and above efficiently and well.

GPT is technically a misnomer now, since there will be no attention (transformer) at all contained in the architecture.

## Install

```bash
$ pip install g-mlp-gpt
```

## Usage

```python
import torch
from g_mlp_gpt import gMLPGPT

model = gMLPGPT(
    num_tokens = 20000,
    dim = 512,
    depth = 4,
    seq_len = 1024,
    window_size = (128, 256, 512, 1024) # window sizes for each depth
)

x = torch.randint(0, 20000, (1, 1000))
logits = model(x) # (1, 1000, 20000)
```

16k context length

```python
import torch
from g_mlp_gpt import gMLPGPT

model = gMLPGPT(
    num_tokens = 20000,
    dim = 512,
    seq_len = 16384,
    depth = 8,
    reversible = True,
    window = (128, 128, 256, 512, 1024, 1024, 2048, 2048, 4096, 4096, 8192, 8192),
    axial = (1, 1, 1, 1, 1, 1, 2, 2, 4, 4, 8, 8)
).cuda()

x = torch.randint(0, 20000, (1, 16384)).cuda()
logits = model(x) # (1, 16384, 20000)
```

## Citations

```bibtex
@misc{liu2021pay,
    title   = {Pay Attention to MLPs}, 
    author  = {Hanxiao Liu and Zihang Dai and David R. So and Quoc V. Le},
    year    = {2021},
    eprint  = {2105.08050},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```
