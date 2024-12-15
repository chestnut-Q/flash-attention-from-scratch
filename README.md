# FlashAttention from Scratch -- Single-GPU and Multi-CPU Implementations

## Introduction

This is an implementation of FlashAttention's forward propagation that doesn't rely on any external library. It's available in two versions: one for a single GPU and another for a multi-CPU cluster. The GPU version is implemented in CUDA, primarily following the algorithm in [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691). The CPU version is implemented using MPI and OpenMP, with partitioning based on the sequence length of Q to enable parallel processing across multiple nodes.

In the current version, Q, K, and V are all set as N×N single-precision matrices, where both the sequence length and hidden size are N. Future versions will gradually introduce support for additional features.

## Getting Started

This guide outlines the steps for compiling and running the implementations in their respective directories.

* GPU Implementation

```bash
cd gpu
make
./opt
```

* CPU Implementation

```bash
cd cpu
make opt
srun -N 4 -n 8 ./opt
```
