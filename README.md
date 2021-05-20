## GPT - gMLP (wip)

This repository will attempt to crack long context autoregressive language modeling (GPT) using variations of <a href="https://arxiv.org/abs/2105.08050">gMLPs</a>. Specifically, it will contain a variant that does gMLP for local sliding windows. The hope is to be able to stretch a single GPU to be able to train context lengths of 4096 and above efficiently and well.

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
