import argparse
from typing import List
import os

import torch
import triton
import triton.language as tl

# Local imports
from display import print_end_line
from tensor_type import Float32, Int32
from test_puzzle import test


"""
# Triton Puzzles Lite

Programming for accelerators such as GPUs is critical for modern AI systems.
This often means programming directly in proprietary low-level languages such as CUDA. Triton is 
an alternative open-source language that allows you to code at a higher-level and compile to accelerators 
like GPU.

Coding for Triton is very similar to Numpy and PyTorch in both syntax and semantics. However, as a lower-level 
language there are a lot of details that you need to keep track of. In particular, one area that learners have 
trouble with is memory loading and storage which is critical for speed on low-level devices.

This set is puzzles is meant to teach you how to use Triton from first principles in an interactive fashion. 
You will start with trivial examples and build your way up to real algorithms like Flash Attention and 
Quantized neural networks. These puzzles **do not** need to run on GPU since they use a Triton interpreter.
"""


r"""
## Introduction

To begin with, we will only use `tl.load` and `tl.store` in order to build simple programs.
"""


"""
### Demo 1

Here's an example of load. It takes an `arange` over the memory. By default the indexing of
torch tensors with column, rows, depths or right-to-left. It also takes in a mask as the second
argument. Mask is critically important because all shapes in Triton need to be powers of two.

Expected Results:

[0 1 2 3 4 5 6 7]
[1. 1. 1. 1. 1. 0. 0. 0.]

Explanation:

tl.load(ptr, mask)
tl.load use mask: [0 1 2 3 4 5 6 7] < 5 = [1 1 1 1 1 0 0 0]
"""


@triton.jit
def demo1(x_ptr):
    range = tl.arange(0, 8)
    # print works in the interpreter
    print(range)
    x = tl.load(x_ptr + range, range < 5, 0)
    print(x)


def run_demo1():
    print("Demo1 Output: ")
    demo1[(1, 1, 1)](torch.ones(4, 3))
    print_end_line()


"""
### Demo 2:

You can also use this trick to read in a 2d array.

Expected Results:

[[ 0  1  2  3]
[ 4  5  6  7]
[ 8  9 10 11]
[12 13 14 15]
[16 17 18 19]
[20 21 22 23]
[24 25 26 27]
[28 29 30 31]]
[[1. 1. 1. 0.]
[1. 1. 1. 0.]
[1. 1. 1. 0.]
[1. 1. 1. 0.]
[0. 0. 0. 0.]
[0. 0. 0. 0.]
[0. 0. 0. 0.]
[0. 0. 0. 0.]]

Explanation:

tl.load use mask: i < 4 and j < 3.
"""


@triton.jit
def demo2(x_ptr):
    i_range = tl.arange(0, 8)[:, None]
    j_range = tl.arange(0, 4)[None, :]
    range = i_range * 4 + j_range
    # print works in the interpreter
    print(range)
    x = tl.load(x_ptr + range, (i_range < 4) & (j_range < 3), 0)
    print(x)


def run_demo2():
    print("Demo2 Output: ")
    demo2[(1, 1, 1)](torch.ones(4, 4))
    print_end_line()


"""
### Demo 3

The `tl.store` function is quite similar. It allows you to write to a tensor.

Expected Results:

tensor([[10., 10., 10.],
    [10., 10.,  1.],
    [ 1.,  1.,  1.],
    [ 1.,  1.,  1.]])

Explanation:

tl.store(ptr, value, mask)
here range < 5 corresponds to the 2D-mask

[[1. 1. 1.]
[1. 1. 0.]
[0. 0. 0.]
[0. 0. 0.]]
"""


@triton.jit
def demo3(z_ptr):
    range = tl.arange(0, 8)
    z = tl.store(z_ptr + range, 10, range < 5)


def run_demo3():
    print("Demo3 Output: ")
    z = torch.ones(4, 3)
    demo3[(1, 1, 1)](z)
    print(z)
    print_end_line()


"""
### Demo 4

You can only load in relatively small `blocks` at a time in Triton. To work 
with larger tensors you need to use a program id axis to run multiple blocks in 
parallel. 

Here is an example with one program axis with 3 blocks.

Expected Results:

Print for each [0] [1. 1. 1. 1. 1. 1. 1. 1.]
Print for each [1] [1. 1. 1. 1. 1. 1. 1. 1.]
Print for each [2] [1. 1. 1. 1. 0. 0. 0. 0.]

Explanation:

This program launch 3 blocks in parallel. For each block (pid=0, 1, 2), it loads 8 
elements. Note that similar to demo3, multi-dimensional tensors are flattened when we 
use pointer (i.e. continuous in memory).
"""


@triton.jit
def demo4(x_ptr):
    pid = tl.program_id(0)
    range = tl.arange(0, 8) + pid * 8
    x = tl.load(x_ptr + range, range < 20)
    print("Print for each", pid, x)


def run_demo4():
    print("Demo4 Output: ")
    x = torch.ones(2, 4, 4)
    demo4[(3, 1, 1)](x)
    print_end_line()


r"""
## Puzzle 1: Constant Add

Add a constant to a vector. Uses one program id axis. 
Block size `B0` is always the same as vector `x` with length `N0`.

.. math::
    z_i = 10 + x_i \text{ for } i = 1\ldots N_0
"""


def add_spec(x: Float32[32,]) -> Float32[32,]:
    "This is the spec that you should implement. Uses typing to define sizes."
    return x + 10.0


@triton.jit
def add_kernel(x_ptr, z_ptr, N0, B0: tl.constexpr):
    # We name the offsets of the pointers as "off_"
    off_x = tl.arange(0, B0)
    x = tl.load(x_ptr + off_x)
    # Finish me!
    z = x + 10.0
    tl.store(z_ptr + off_x, z)

    return


r"""
## Puzzle 2: Constant Add Block

Add a constant to a vector. Uses one program block axis (no `for` loops yet). 
Block size `B0` is now smaller than the shape vector `x` which is `N0`.

.. math::
    z_i = 10 + x_i \text{ for } i = 1\ldots N_0
"""


def add2_spec(x: Float32[200,]) -> Float32[200,]:
    return x + 10.0


@triton.jit
def add_mask2_kernel(x_ptr, z_ptr, N0, B0: tl.constexpr):
    # Finish me!
    pid = tl.program_id(0)
    off_x = tl.arange(0, B0) + pid * B0
    x = tl.load(x_ptr + off_x, off_x < N0, 0.0)
    z = x + 10.0
    tl.store(z_ptr + off_x, z, off_x < N0)

    return


r"""
## Puzzle 3: Outer Vector Add

Add two vectors.

Uses one program block axis. Block size `B0` is always the same as vector `x` length `N0`.
Block size `B1` is always the same as vector `y` length `N1`.

.. math::
    z_{j, i} = x_i + y_j\text{ for } i = 1\ldots B_0,\ j = 1\ldots B_1
"""


def add_vec_spec(x: Float32[32,], y: Float32[32,]) -> Float32[32, 32]:
    return x[None, :] + y[:, None]


@triton.jit
def add_vec_kernel(x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr):
    # Finish me!
    # 加载整个 x 向量
    off_x = tl.arange(0, B0)    # [0, 1, 2, ..., N0-1]
    x = tl.load(x_ptr + off_x)  # (N0,)
    
    # 加载整个 y 向量  
    off_y = tl.arange(0, B1)    # [0, 1, 2, ..., N1-1]
    y = tl.load(y_ptr + off_y)  # (N1,)
    
    # 通过广播进行外积加法
    z = x[None, :] + y[:, None]  # (N1, N0)
    
    # 计算二维存储偏移
    i_range = tl.arange(0, B1)[:, None]  # (N1, 1)
    j_range = tl.arange(0, B0)[None, :]  # (1, N0)
    off_z = i_range * N0 + j_range       # (N1, N0)
    
    # 存储结果
    tl.store(z_ptr + off_z, z)

    return


r"""
## Puzzle 4: Outer Vector Add Block

Add a row vector to a column vector.

Uses two program block axes. Block size `B0` is always less than the vector `x` length `N0`.
Block size `B1` is always less than vector `y` length `N1`.

.. math::
    z_{j, i} = x_i + y_j\text{ for } i = 1\ldots N_0,\ j = 1\ldots N_1
"""


def add_vec_block_spec(x: Float32[100,], y: Float32[90,]) -> Float32[90, 100]:
    return x[None, :] + y[:, None]


@triton.jit
def add_vec_block_kernel(
    x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr
):
    block_id_x = tl.program_id(0)
    block_id_y = tl.program_id(1)
    # Finish me!
    off_x = tl.arange(0, B0) + block_id_x * B0
    off_y = tl.arange(0, B1) + block_id_y * B1
    x = tl.load(x_ptr + off_x, off_x < N0, 0)
    y = tl.load(y_ptr + off_y, off_y < N1, 0)
    
    z = x[None, :] + y[:, None]

    i_range = tl.arange(0, B1)[:, None] + block_id_y * B1 # (B1, 1), y
    j_range = tl.arange(0, B0)[None, :] + block_id_x * B0 # (1, B0), x
    off_z = i_range * N0 + j_range

    tl.store(z_ptr + off_z, z, (i_range < N1) & (j_range < N0))

    return


r"""
## Puzzle 5: Fused Outer Multiplication

Multiply a row vector to a column vector and take a relu.

Uses two program block axes. Block size `B0` is always less than the vector `x` length `N0`.
Block size `B1` is always less than vector `y` length `N1`.

.. math::
    z_{j, i} = \text{relu}(x_i \times y_j)\text{ for } i = 1\ldots N_0,\ j = 1\ldots N_1
"""


def mul_relu_block_spec(x: Float32[100,], y: Float32[90,]) -> Float32[90, 100]:
    return torch.relu(x[None, :] * y[:, None])


@triton.jit
def mul_relu_block_kernel(
    x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr
):
    block_id_x = tl.program_id(0)
    block_id_y = tl.program_id(1)
    # Finish me!
    off_x = tl.arange(0, B0) + block_id_x * B0
    off_y = tl.arange(0, B1) + block_id_y * B1
    x = tl.load(x_ptr + off_x, off_x < N0, 0)
    y = tl.load(y_ptr + off_y, off_y < N1, 0)

    product = x[None, :] * y[:, None]
    z = tl.where(product > 0, product, 0.0)

    i_range = tl.arange(0, B1)[:, None] + block_id_y * B1 # (B1, 1), y
    j_range = tl.arange(0, B0)[None, :] + block_id_x * B0 # (1, B0), x
    off_z = i_range * N0 + j_range  # (B1, B0)
    tl.store(z_ptr + off_z, z, (i_range < N1) & (j_range < N0))
    return


r"""
## Puzzle 6: Fused Outer Multiplication - Backwards

Backwards of a function that multiplies a matrix with a row vector and take a relu.

Uses two program blocks. Block size `B0` is always less than the vector `x` length `N0`.
Block size `B1` is always less than vector `y` length `N1`. Chain rule backward `dz`
is of shape `N1` by `N0`

.. math::
    f(x, y) = \text{relu}(x_{j, i} \times y_j)\text{ for } i = 1\ldots N_0,\ j = 1\ldots N_1

.. math::
    dx_{j, i} = f_x'(x, y)_{j, i} \times dz_{j, i}
"""


def mul_relu_block_back_spec(
    x: Float32[90, 100], y: Float32[90,], dz: Float32[90, 100]
) -> Float32[90, 100]:
    x = x.clone()
    y = y.clone()
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    z = torch.relu(x * y[:, None])
    z.backward(dz)
    dx = x.grad
    return dx


@triton.jit
def mul_relu_block_back_kernel(
    x_ptr, y_ptr, dz_ptr, dx_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr
):
    block_id_i = tl.program_id(0)
    block_id_j = tl.program_id(1)
    # Finish me!

    off_y = tl.arange(0, B1) + block_id_j * B1

    y = tl.load(y_ptr + off_y, off_y < N1, 0)

    i_range = tl.arange(0, B1)[:, None] + block_id_j * B1
    j_range = tl.arange(0, B0)[None, :] + block_id_i * B0
    off_matrix = i_range * N0 + j_range

    x = tl.load(x_ptr + off_matrix, (i_range < N1) & (j_range < N0), 0)
    dz = tl.load(dz_ptr + off_matrix, (i_range < N1) & (j_range < N0), 0)

    product = x * y[:, None]
    # dx = (product > 0) ? y * dz : 0
    dx = tl.where(product > 0, y[:, None] * dz, 0.0)

    tl.store(dx_ptr + off_matrix, dx, (i_range < N1) & (j_range < N0))

    return


r"""
## Puzzle 7: Long Sum

Sum of a batch of numbers.

Uses one program blocks. Block size `B0` represents a range of batches of  `x` of length `N0`.
Each element is of length `T`. Process it `B1 < T` elements at a time.  

.. math::
    z_{i} = \sum^{T}_j x_{i,j} =  \text{ for } i = 1\ldots N_0

Hint: You will need a for loop for this problem. These work and look the same as in Python.
"""


def sum_spec(x: Float32[4, 200]) -> Float32[4,]:
    return x.sum(1)


@triton.jit
def sum_kernel(x_ptr, z_ptr, N0, N1, T, B0: tl.constexpr, B1: tl.constexpr):
    # Finish me!
    batch_id = tl.program_id(0)
    
    # 计算当前批次的起始偏移
    batch_offset = batch_id * T  # 每行有 T 个元素
    
    # 初始化累加器
    acc = tl.zeros((B1,), dtype=tl.float32)
    
    # 使用 for 循环处理这一行的所有元素
    for i in range(0, T, B1):  # 每次处理 B1 个元素
        # 计算当前块的偏移
        off_x = tl.arange(0, B1) + i  # 当前块在行内的偏移
        
        # 加载数据
        x_chunk = tl.load(x_ptr + batch_offset + off_x, off_x < T, 0.0)
        
        # 累加
        acc += x_chunk
    
    # 对块内所有元素求和
    result = tl.sum(acc)
    
    # 存储结果
    tl.store(z_ptr + batch_id, result)
    

    return


r"""
## Puzzle 8: Long Softmax

Softmax of a batch of logits.

Uses one program block axis. Block size `B0` represents the batch of `x` of length `N0`.
Block logit length `T`.   Process it `B1 < T` elements at a time.  

.. math::
    z_{i, j} = \text{softmax}(x_{i,1} \ldots x_{i, T}) \text{ for } i = 1\ldots N_0

Note softmax needs to be computed in numerically stable form as in Python. In addition in Triton 
they recommend not using `exp` but instead using `exp2`. You need the identity

.. math::
    \exp(x) = 2^{\log_2(e) x}

Advanced: there one way to do this with 3 loops. You can also do it with 2 loops if you are clever. 
Hint: you will find this identity useful:

.. math::
    \exp(x_i - m) =  \exp(x_i - m/2 - m/2) = \exp(x_i - m/ 2) /  \exp(m/2)
"""


def softmax_spec(x: Float32[4, 200]) -> Float32[4, 200]:
    x_max = x.max(1, keepdim=True)[0]
    x = x - x_max
    x_exp = x.exp()
    return x_exp / x_exp.sum(1, keepdim=True)


@triton.jit
def softmax_kernel(x_ptr, z_ptr, N0, N1, T, B0: tl.constexpr, B1: tl.constexpr):
    """2 loops ver."""
    block_id_i = tl.program_id(0)
    log2_e = 1.44269504
    # Finish me!
    # 检查是否超出批次边界
    if block_id_i >= N0:
        return

    # 计算当前行的起始偏移
    row_offset = block_id_i * T
    
    # 第1个循环：同时找最大值和计算部分和
    max_val = float('-inf')
    sum_exp = 0.0
    
    # 存储所有的块数据，避免重复加载
    chunks = []
    
    for i in range(0, T, B1):
        off_x = tl.arange(0, B1) + i
        mask = off_x < T
        x_chunk = tl.load(x_ptr + row_offset + off_x, mask, float('-inf'))
        
        # 更新最大值
        chunk_max = tl.max(tl.where(mask, x_chunk, float('-inf')))
        old_max = max_val
        max_val = tl.maximum(max_val, chunk_max)
        
        # 使用数学恒等式重新计算之前的和
        # exp(x - new_max) = exp(x - old_max) * exp(old_max - new_max)
        if i > 0:
            correction = tl.exp2(log2_e * (old_max - max_val))
            sum_exp *= correction
        
        # 计算当前块的贡献
        exp_chunk = tl.exp2(log2_e * (x_chunk - max_val))
        exp_chunk = tl.where(mask, exp_chunk, 0.0)
        sum_exp += tl.sum(exp_chunk)
        
        # 存储块数据供后续使用
        chunks.append((x_chunk, mask, off_x))
    
    # 第2个循环：计算并存储最终结果
    for x_chunk, mask, off_x in chunks:
        exp_chunk = tl.exp2(log2_e * (x_chunk - max_val))
        softmax_chunk = exp_chunk / sum_exp
        tl.store(z_ptr + row_offset + off_x, softmax_chunk, mask)
    
    return


@triton.jit
def softmax_kernel_brute_force(
    x_ptr, z_ptr, N0, N1, T, B0: tl.constexpr, B1: tl.constexpr
):
    """3 loops ver."""
    block_id_i = tl.program_id(0)
    log2_e = 1.44269504
    # Finish me!
    row_offset = block_id_i * T
    
    # 第1个循环：找到最大值
    max_val = float('-inf')
    for i in range(0, T, B1):
        off_x = tl.arange(0, B1) + i
        mask = off_x < T
        x_chunk = tl.load(x_ptr + row_offset + off_x, mask, float('-inf'))
        chunk_max = tl.max(x_chunk)
        max_val = tl.maximum(max_val, chunk_max)
    
    # 第2个循环：计算 exp(x - max) 的和
    sum_exp = 0.0
    for i in range(0, T, B1):
        off_x = tl.arange(0, B1) + i
        mask = off_x < T
        x_chunk = tl.load(x_ptr + row_offset + off_x, mask, 0.0)
        # 使用 exp2(log2_e * x) 替代 exp(x)
        exp_chunk = tl.exp2(log2_e * (x_chunk - max_val))
        exp_chunk = tl.where(mask, exp_chunk, 0.0)
        sum_exp += tl.sum(exp_chunk)
    
    # 第3个循环：计算最终的 softmax 值并存储
    for i in range(0, T, B1):
        off_x = tl.arange(0, B1) + i
        mask = off_x < T
        x_chunk = tl.load(x_ptr + row_offset + off_x, mask, 0.0)
        exp_chunk = tl.exp2(log2_e * (x_chunk - max_val))
        softmax_chunk = exp_chunk / sum_exp
        tl.store(z_ptr + row_offset + off_x, softmax_chunk, mask)
    
    return


r"""
## Puzzle 9: Simple FlashAttention

A scalar version of FlashAttention.

Uses zero programs. Block size `B0` represent the batches of `q` to process out of `N0`. Sequence length is `T`. Process it `B1 < T` elements (`k`, `v`) at a time for some `B1`.

.. math::
    z_{i} = \sum_{j=1}^{T} \text{softmax}(q_i k_1, \ldots, q_i k_T)_j v_{j} \text{ for } i = 1\ldots N_0

This can be done in 1 loop using a similar trick from the last puzzle.

Hint: Use `tl.where` to mask `q dot k` to -inf to avoid overflow (NaN).
"""


def flashatt_spec(
    q: Float32[200,], k: Float32[200,], v: Float32[200,]
) -> Float32[200,]:
    x = q[:, None] * k[None, :]
    x_max = x.max(1, keepdim=True)[0]
    x = x - x_max
    x_exp = x.exp()
    soft = x_exp / x_exp.sum(1, keepdim=True)
    return (v[None, :] * soft).sum(1)


@triton.jit
def flashatt_kernel(
    q_ptr, k_ptr, v_ptr, z_ptr, N0, T, B0: tl.constexpr, B1: tl.constexpr
):
    block_id_i = tl.program_id(0)
    log2_e = 1.44269504
    myexp = lambda x: tl.exp2(log2_e * x)
    # Finish me!
    return


r"""
## Puzzle 10: Two Dimensional Convolution

A batched 2D convolution.

Uses one program id axis. Block size `B0` represent the batches to process out of `N0`.
Image `x` is size is `H` by `W` with only 1 channel, and kernel `k` is size `KH` by `KW`.

.. math::
    z_{i, j, l} = \sum_{oj, ol}^{j+oj\le H, l+ol\le W} k_{oj,ol} \times x_{i,j + oj, l + ol} 
    \text{ for } i = 1\ldots N_0 \text{ for } j = 1\ldots H \text{ for } l = 1\ldots W
"""


def conv2d_spec(x: Float32[4, 8, 8], k: Float32[4, 4]) -> Float32[4, 8, 8]:
    z = torch.zeros(4, 8, 8)
    x = torch.nn.functional.pad(x, (0, 4, 0, 4, 0, 0), value=0.0)
    # print(x.shape, k.shape)
    for i in range(8):
        for j in range(8):
            z[:, i, j] = (k[None, :, :] * x[:, i : i + 4, j : j + 4]).sum(1).sum(1)
    return z


@triton.jit
def conv2d_kernel(
    x_ptr, k_ptr, z_ptr, N0, H, W, KH: tl.constexpr, KW: tl.constexpr, B0: tl.constexpr
):
    block_id_i = tl.program_id(0)
    # Finish me!
    # 计算batch偏移（向量化处理多个batch）
    off_i = block_id_i * B0 + tl.arange(0, B0)
    mask_i = off_i < N0

    # 预计算卷积核的偏移
    off_h = tl.arange(0, KH)  # [0, 1, 2, 3]
    off_w = tl.arange(0, KW)  # [0, 1, 2, 3]
    off_hw = off_h[:, None] * KW + off_w[None, :]  # [KH, KW] 卷积核的2D偏移

    # 一次性加载整个卷积核
    k = tl.load(k_ptr + off_hw)

    # 遍历输出特征图的每个位置
    for j in tl.range(0, H):  # 输出高度
        for l in tl.range(0, W):  # 输出宽度
            
            # 计算输入图像中对应的位置范围
            off_j_oj = j + off_h[None, :, None]  # [1, KH, 1] -> [B0, KH, KW]
            off_l_ol = l + off_w[None, None, :]  # [1, 1, KW] -> [B0, KH, KW]
            
            # 计算输入数据的线性偏移
            # off_i[:, None, None] 扩展为 [B0, 1, 1] -> [B0, KH, KW]
            off_x = off_i[:, None, None] * H * W + off_j_oj * W + off_l_ol
            
            # 边界检查（隐式zero padding）
            mask_x = (off_j_oj < H) & (off_l_ol < W) & mask_i[:, None, None]
            
            # 加载输入数据块 [B0, KH, KW]
            x = tl.load(x_ptr + off_x, mask=mask_x, other=0.0)

            # 执行卷积运算（element-wise乘法然后求和）
            # k[None, :, :] 扩展为 [1, KH, KW] -> [B0, KH, KW]
            conv_result = x * k[None, :, :]  # [B0, KH, KW]
            
            # 分步求和：先对 KW 维度求和，再对 KH 维度求和
            z = tl.sum(tl.sum(conv_result, axis=2), axis=1)  # [B0]

            # 计算输出偏移并存储结果
            off_z = off_i * H * W + j * W + l
            tl.store(z_ptr + off_z, z, mask=mask_i)

    return


r"""
## Puzzle 11: Matrix Multiplication

A blocked matrix multiplication.

Uses three program id axes. Block size `B2` represent the batches to process out of `N2`.
Block size `B0` represent the rows of `x` to process out of `N0`. Block size `B1` represent the cols 
of `y` to process out of `N1`. The middle shape is `MID`.

.. math::
    z_{i, j, k} = \sum_{l} x_{i,j, l} \times y_{i, l, k} \text{ for } i = 1\ldots N_2, j = 1\ldots N_0, k = 1\ldots N_1

You are allowed to use `tl.dot` which computes a smaller mat mul.

Hint: the main trick is that you can split a matmul into smaller parts.

.. math::
    z_{i, j, k} = \sum_{l=1}^{L/2} x_{i,j, l} \times y_{i, l, k} +  \sum_{l=L/2}^{L} x_{i,j, l} \times y_{i, l, k}
"""


def dot_spec(x: Float32[4, 32, 32], y: Float32[4, 32, 32]) -> Float32[4, 32, 32]:
    return x @ y


@triton.jit
def dot_kernel(
    x_ptr,
    y_ptr,
    z_ptr,
    N0,
    N1,
    N2,
    MID,
    B0: tl.constexpr,
    B1: tl.constexpr,
    B2: tl.constexpr,
    B_MID: tl.constexpr,
):
    block_id_j = tl.program_id(0)
    block_id_k = tl.program_id(1)
    block_id_i = tl.program_id(2)
    # Finish me!
    # 计算当前块的起始位置
    start_j = block_id_j * B0
    start_k = block_id_k * B1
    start_i = block_id_i * B2
    
    # 创建索引范围
    offs_j = start_j + tl.arange(0, B0)
    offs_k = start_k + tl.arange(0, B1)
    offs_i = start_i + tl.arange(0, B2)
    
    # 初始化累加器
    accumulator = tl.zeros((B2, B0, B1), dtype=tl.float32)
    
    # 沿着 MID 维度进行分块累加
    for mid_start in range(0, MID, B_MID):
        offs_mid = mid_start + tl.arange(0, B_MID)
        
        # 构建内存访问掩码
        mask_x = (offs_i[:, None, None] < N2) & (offs_j[None, :, None] < N0) & (offs_mid[None, None, :] < MID)
        mask_y = (offs_i[:, None, None] < N2) & (offs_mid[None, :, None] < MID) & (offs_k[None, None, :] < N1)
        
        # 计算内存偏移
        x_offsets = offs_i[:, None, None] * N0 * MID + offs_j[None, :, None] * MID + offs_mid[None, None, :]
        y_offsets = offs_i[:, None, None] * MID * N1 + offs_mid[None, :, None] * N1 + offs_k[None, None, :]
        
        # 加载数据块
        x_block = tl.load(x_ptr + x_offsets, mask=mask_x, other=0.0)
        y_block = tl.load(y_ptr + y_offsets, mask=mask_y, other=0.0)
        
        # 执行分块矩阵乘法并累加
        accumulator += tl.dot(x_block, y_block)
    
    # 存储结果
    mask_z = (offs_i[:, None, None] < N2) & (offs_j[None, :, None] < N0) & (offs_k[None, None, :] < N1)
    z_offsets = offs_i[:, None, None] * N0 * N1 + offs_j[None, :, None] * N1 + offs_k[None, None, :]
    tl.store(z_ptr + z_offsets, accumulator, mask=mask_z)
    return


r"""
## Puzzle 12: Quantized Matrix Mult

When doing matrix multiplication with quantized neural networks a common strategy is to store the weight matrix in lower precision, with a shift and scale term.

For this problem our `weight` will be stored in 4 bits. We can store `FPINT` of these in a 32 bit integer. In addition for every `group` weights in order we will store 1 `scale` float value and 1 `shift` 4 bit value. We store these for the column of weight. The `activation`s are stored separately in standard floats.

Mathematically it looks like.

.. math::
    z_{j, k} = \sum_{l} sc_{j, \frac{l}{g}} (w_{j, l} - sh_{j, \frac{l}{g}}) \times y_{l, k} 
    \text{ for } j = 1\ldots N_0, k = 1\ldots N_1

Where `g` is the number of groups (`GROUP`).

However, it is a bit more complex since we need to also extract the 4-bit values into floats to begin.

Note:
- We don't consider batch size, i.e. `i`, in this puzzle.
- Remember to unpack the `FPINT` values into separate 4-bit values. This contains some shape manipulation.
"""

FPINT = 32 // 4
GROUP = 8


def quant_dot_spec(
    scale: Float32[32, 8],
    offset: Int32[32,],
    weight: Int32[32, 8],
    activation: Float32[64, 32],
) -> Float32[32, 32]:
    offset = offset.view(32, 1)

    def extract(x):
        over = torch.arange(8) * 4
        mask = 2**4 - 1
        return (x[..., None] >> over) & mask

    scale = scale[..., None].expand(-1, 8, GROUP).contiguous().view(-1, 64)
    offset = (
        extract(offset)[..., None].expand(-1, 1, 8, GROUP).contiguous().view(-1, 64)
    )
    return (scale * (extract(weight).view(-1, 64) - offset)) @ activation


@triton.jit
def quant_dot_kernel(
    scale_ptr,
    offset_ptr,
    weight_ptr,
    activation_ptr,
    z_ptr,
    N0,
    N1,
    MID,
    B0: tl.constexpr,
    B1: tl.constexpr,
    B_MID: tl.constexpr,
):
    block_id_j = tl.program_id(0)
    block_id_k = tl.program_id(1)
    # Finish me!

    return


def run_demos():
    run_demo1()
    run_demo2()
    run_demo3()
    run_demo4()


def run_puzzles(args, puzzles: List[int]):
    print_log = args.log
    device = args.device

    if 1 in puzzles:
        print("Puzzle #1:")
        ok = test(
            add_kernel,
            add_spec,
            nelem={"N0": 32},
            print_log=print_log,
            device=device,
        )
        print_end_line()
        if not ok:
            return
    if 2 in puzzles:
        print("Puzzle #2:")
        ok = test(
            add_mask2_kernel,
            add2_spec,
            nelem={"N0": 200},
            print_log=print_log,
            device=device,
        )
        print_end_line()
        if not ok:
            return
    if 3 in puzzles:
        print("Puzzle #3:")
        ok = test(
            add_vec_kernel,
            add_vec_spec,
            nelem={"N0": 32, "N1": 32},
            print_log=print_log,
            device=device,
        )
        print_end_line()
        if not ok:
            return
    if 4 in puzzles:
        print("Puzzle #4:")
        ok = test(
            add_vec_block_kernel,
            add_vec_block_spec,
            nelem={"N0": 100, "N1": 90},
            print_log=print_log,
            device=device,
        )
        print_end_line()
        if not ok:
            return
    if 5 in puzzles:
        print("Puzzle #5:")
        ok = test(
            mul_relu_block_kernel,
            mul_relu_block_spec,
            nelem={"N0": 100, "N1": 90},
            print_log=print_log,
            device=device,
        )
        print_end_line()
        if not ok:
            return
    if 6 in puzzles:
        print("Puzzle #6:")
        ok = test(
            mul_relu_block_back_kernel,
            mul_relu_block_back_spec,
            nelem={"N0": 100, "N1": 90},
            print_log=print_log,
            device=device,
        )
        print_end_line()
        if not ok:
            return
    if 7 in puzzles:
        print("Puzzle #7:")
        ok = test(
            sum_kernel,
            sum_spec,
            B={"B0": 1, "B1": 32},
            nelem={"N0": 4, "N1": 32, "T": 200},
            print_log=print_log,
            device=device,
        )
        print_end_line()
        if not ok:
            return
    if 8 in puzzles:
        print("Puzzle #8:")
        ok = test(
            softmax_kernel,
            softmax_spec,
            B={"B0": 1, "B1": 32},
            nelem={"N0": 4, "N1": 32, "T": 200},
            print_log=print_log,
            device=device,
        )
        print_end_line()
        if not ok:
            return
    if 9 in puzzles:
        print("Puzzle #9:")
        ok = test(
            flashatt_kernel,
            flashatt_spec,
            B={"B0": 64, "B1": 32},
            nelem={"N0": 200, "T": 200},
            print_log=print_log,
            device=device,
        )
        print_end_line()
        if not ok:
            return
    if 10 in puzzles:
        print("Puzzle #10:")
        ok = test(
            conv2d_kernel,
            conv2d_spec,
            B={"B0": 1},
            nelem={"N0": 4, "H": 8, "W": 8, "KH": 4, "KW": 4},
            print_log=print_log,
            device=device,
        )
        print_end_line()
        if not ok:
            return
    if 11 in puzzles:
        print("Puzzle #11:")
        ok = test(
            dot_kernel,
            dot_spec,
            B={"B0": 16, "B1": 16, "B2": 1, "B_MID": 16},
            nelem={"N0": 32, "N1": 32, "N2": 4, "MID": 32},
            print_log=print_log,
            device=device,
        )
        print_end_line()
        if not ok:
            return
    if 12 in puzzles:
        print("Puzzle #12:")
        ok = test(
            quant_dot_kernel,
            quant_dot_spec,
            B={"B0": 16, "B1": 16, "B_MID": 64},
            nelem={"N0": 32, "N1": 32, "MID": 64},
            print_log=print_log,
            device=device,
        )
        print_end_line()
        if not ok:
            return
    print("All tests passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--puzzle", type=int, metavar="N", help="Run Puzzle #N")
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Run all Puzzles. Stop at first failure.",
    )
    parser.add_argument("-l", "--log", action="store_true", help="Print log messages.")
    parser.add_argument(
        "-i",
        "--intro",
        action="store_true",
        help="Run all demos in the introduction part.",
    )

    args = parser.parse_args()

    if os.getenv("TRITON_INTERPRET", "0") == "1":
        torch.set_default_device("cpu")
        args.device = "cpu"
    else:  # GPU mode
        torch.set_default_device("cuda")
        args.device = "cuda"

    if args.intro:
        run_demos()
    elif args.all:
        run_puzzles(args, list(range(1, 13)))
    elif args.puzzle:
        run_puzzles(args, [int(args.puzzle)])
    else:
        parser.print_help()
