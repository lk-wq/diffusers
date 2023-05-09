# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import math

import flax.linen as nn
import jax
import jax.numpy as jnp


def _query_chunk_attention(query, key, value, precision, key_chunk_size: int = 4096):
    """Multi-head dot product attention with a limited number of queries."""
    num_kv, num_heads, k_features = key.shape[-3:]
    v_features = value.shape[-1]
    key_chunk_size = min(key_chunk_size, num_kv)
    query = query / jnp.sqrt(k_features)

    @functools.partial(jax.checkpoint, prevent_cse=False)
    def summarize_chunk(query, key, value):
        attn_weights = jnp.einsum("...qhd,...khd->...qhk", query, key, precision=precision)

        max_score = jnp.max(attn_weights, axis=-1, keepdims=True)
        max_score = jax.lax.stop_gradient(max_score)
        exp_weights = jnp.exp(attn_weights - max_score)

        exp_values = jnp.einsum("...vhf,...qhv->...qhf", value, exp_weights, precision=precision)
        max_score = jnp.einsum("...qhk->...qh", max_score)

        return (exp_values, exp_weights.sum(axis=-1), max_score)

    def chunk_scanner(chunk_idx):
        # julienne key array
        key_chunk = jax.lax.dynamic_slice(
            operand=key,
            start_indices=[0] * (key.ndim - 3) + [chunk_idx, 0, 0],  # [...,k,h,d]
            slice_sizes=list(key.shape[:-3]) + [key_chunk_size, num_heads, k_features],  # [...,k,h,d]
        )

        # julienne value array
        value_chunk = jax.lax.dynamic_slice(
            operand=value,
            start_indices=[0] * (value.ndim - 3) + [chunk_idx, 0, 0],  # [...,v,h,d]
            slice_sizes=list(value.shape[:-3]) + [key_chunk_size, num_heads, v_features],  # [...,v,h,d]
        )

        return summarize_chunk(query, key_chunk, value_chunk)

    chunk_values, chunk_weights, chunk_max = jax.lax.map(f=chunk_scanner, xs=jnp.arange(0, num_kv, key_chunk_size))

    global_max = jnp.max(chunk_max, axis=0, keepdims=True)
    max_diffs = jnp.exp(chunk_max - global_max)

    chunk_values *= jnp.expand_dims(max_diffs, axis=-1)
    chunk_weights *= max_diffs

    all_values = chunk_values.sum(axis=0)
    all_weights = jnp.expand_dims(chunk_weights, -1).sum(axis=0)

    return all_values / all_weights


def jax_memory_efficient_attention(
    query, key, value, precision=jax.lax.Precision.HIGHEST, query_chunk_size: int = 1024, key_chunk_size: int = 4096
):
    r"""
    Flax Memory-efficient multi-head dot product attention. https://arxiv.org/abs/2112.05682v2
    https://github.com/AminRezaei0x443/memory-efficient-attention

    Args:
        query (`jnp.ndarray`): (batch..., query_length, head, query_key_depth_per_head)
        key (`jnp.ndarray`): (batch..., key_value_length, head, query_key_depth_per_head)
        value (`jnp.ndarray`): (batch..., key_value_length, head, value_depth_per_head)
        precision (`jax.lax.Precision`, *optional*, defaults to `jax.lax.Precision.HIGHEST`):
            numerical precision for computation
        query_chunk_size (`int`, *optional*, defaults to 1024):
            chunk size to divide query array value must divide query_length equally without remainder
        key_chunk_size (`int`, *optional*, defaults to 4096):
            chunk size to divide key and value array value must divide key_value_length equally without remainder

    Returns:
        (`jnp.ndarray`) with shape of (batch..., query_length, head, value_depth_per_head)
    """
    num_q, num_heads, q_features = query.shape[-3:]

    def chunk_scanner(chunk_idx, _):
        # julienne query array
        query_chunk = jax.lax.dynamic_slice(
            operand=query,
            start_indices=([0] * (query.ndim - 3)) + [chunk_idx, 0, 0],  # [...,q,h,d]
            slice_sizes=list(query.shape[:-3]) + [min(query_chunk_size, num_q), num_heads, q_features],  # [...,q,h,d]
        )

        return (
            chunk_idx + query_chunk_size,  # unused ignore it
            _query_chunk_attention(
                query=query_chunk, key=key, value=value, precision=precision, key_chunk_size=key_chunk_size
            ),
        )

    _, res = jax.lax.scan(
        f=chunk_scanner, init=0, xs=None, length=math.ceil(num_q / query_chunk_size)  # start counter  # stop counter
    )

    return jnp.concatenate(res, axis=-3)  # fuse the chunked result back


class FlaxAttention(nn.Module):
    r"""
    A Flax multi-head attention module as described in: https://arxiv.org/abs/1706.03762

    Parameters:
        query_dim (:obj:`int`):
            Input hidden states dimension
        heads (:obj:`int`, *optional*, defaults to 8):
            Number of heads
        dim_head (:obj:`int`, *optional*, defaults to 64):
            Hidden states dimension inside each head
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        use_memory_efficient_attention (`bool`, *optional*, defaults to `False`):
            enable memory efficient attention https://arxiv.org/abs/2112.05682
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`

    """
    query_dim: int
    heads: int = 8
    dim_head: int = 64
    dropout: float = 0.0
    use_memory_efficient_attention: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        inner_dim = self.dim_head * self.heads
        self.scale = self.dim_head**-0.5

        # Weights were exported with old names {to_q, to_k, to_v, to_out}
        self.query = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype, name="to_q")
        self.key = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype, name="to_k")
        self.value = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype, name="to_v")

        self.proj_attn = nn.Dense(self.query_dim, dtype=self.dtype, name="to_out_0")

    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = jnp.transpose(tensor, (0, 2, 1, 3))
        tensor = tensor.reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = jnp.transpose(tensor, (0, 2, 1, 3))
        tensor = tensor.reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def __call__(self, hidden_states, context=None, deterministic=True):
        context = hidden_states if context is None else context

        query_proj = self.query(hidden_states)
        key_proj = self.key(context)
        value_proj = self.value(context)

        query_states = self.reshape_heads_to_batch_dim(query_proj)
        key_states = self.reshape_heads_to_batch_dim(key_proj)
        value_states = self.reshape_heads_to_batch_dim(value_proj)

        if self.use_memory_efficient_attention:
            query_states = query_states.transpose(1, 0, 2)
            key_states = key_states.transpose(1, 0, 2)
            value_states = value_states.transpose(1, 0, 2)

            # this if statement create a chunk size for each layer of the unet
            # the chunk size is equal to the query_length dimension of the deepest layer of the unet

            flatten_latent_dim = query_states.shape[-3]
            if flatten_latent_dim % 64 == 0:
                query_chunk_size = int(flatten_latent_dim / 64)
            elif flatten_latent_dim % 16 == 0:
                query_chunk_size = int(flatten_latent_dim / 16)
            elif flatten_latent_dim % 4 == 0:
                query_chunk_size = int(flatten_latent_dim / 4)
            else:
                query_chunk_size = int(flatten_latent_dim)

            hidden_states = jax_memory_efficient_attention(
                query_states, key_states, value_states, query_chunk_size=query_chunk_size, key_chunk_size=4096 * 4
            )

            hidden_states = hidden_states.transpose(1, 0, 2)
        else:
            # compute attentions
            attention_scores = jnp.einsum("b i d, b j d->b i j", query_states, key_states)
            attention_scores = attention_scores * self.scale
            attention_probs = nn.softmax(attention_scores, axis=2)

            # attend to values
            hidden_states = jnp.einsum("b i j, b j d -> b i d", attention_probs, value_states)

        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        hidden_states = self.proj_attn(hidden_states)
        return attention_probs

class FlaxAttention2(nn.Module):
    r"""
    A Flax multi-head attention module as described in: https://arxiv.org/abs/1706.03762

    Parameters:
        query_dim (:obj:`int`):
            Input hidden states dimension
        heads (:obj:`int`, *optional*, defaults to 8):
            Number of heads
        dim_head (:obj:`int`, *optional*, defaults to 64):
            Hidden states dimension inside each head
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        use_memory_efficient_attention (`bool`, *optional*, defaults to `False`):
            enable memory efficient attention https://arxiv.org/abs/2112.05682
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`

    """
    query_dim: int
    heads: int = 8
    dim_head: int = 64
    dropout: float = 0.0
    use_memory_efficient_attention: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        inner_dim = self.dim_head * self.heads
        self.scale = self.dim_head**-0.5

        # Weights were exported with old names {to_q, to_k, to_v, to_out}
#         self.query = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype, name="to_q")
#         self.key = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype, name="to_k")
#         self.value = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype, name="to_v")

#         self.proj_attn = nn.Dense(self.query_dim, dtype=self.dtype, name="to_out_0")

    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = jnp.transpose(tensor, (0, 2, 1, 3))
        tensor = tensor.reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = jnp.transpose(tensor, (0, 2, 1, 3))
        tensor = tensor.reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def __call__(self, query, key, value, deterministic=True,save=False):
#         context = hidden_states if context is None else context

#         query_proj = self.query(hidden_states)
#         key_proj = self.key(context)
#         value_proj = self.value(context)
        
        print("query k v " , query.shape, key.shape, value.shape)
        query = jnp.transpose(query,(0,2,1,3)).reshape(query.shape[0], query.shape[1],query.shape[2]*query.shape[3])
        query_states = self.reshape_heads_to_batch_dim(query)
        key = jnp.transpose(key,(0,2,1,3)).reshape(key.shape[0], key.shape[1],key.shape[2]*key.shape[3])

        key_states = self.reshape_heads_to_batch_dim(key)
#         value_states = self.reshape_heads_to_batch_dim(value_proj)

        # compute attentions
        print("q s vs ks", query_states.shape , key_states.shape )
#         attention_scores = jnp.einsum("b i d, b j d->b i j", query_states, key_states)
        attention_scores = jnp.einsum("b d i, b d j->b i j", query_states, key_states)

        attention_scores = attention_scores * self.scale
        attention_probs = nn.softmax(attention_scores, axis=2)

            # attend to values
        value = jnp.transpose(value,(0,2,1,3)).reshape(value.shape[0], value.shape[1],value.shape[2]*value.shape[3])
        value_states = self.reshape_heads_to_batch_dim(value)

        print("a vs v",attention_probs.shape, value_states.shape)
        hidden_states = jnp.einsum("b i j, b d j -> b i d", attention_probs, value_states)

#         hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
#         hidden_states = self.proj_attn(hidden_states)
        return attention_probs

class FlaxBasicTransformerBlock(nn.Module):
    r"""
    A Flax transformer block layer with `GLU` (Gated Linear Unit) activation function as described in:
    https://arxiv.org/abs/1706.03762


    Parameters:
        dim (:obj:`int`):
            Inner hidden states dimension
        n_heads (:obj:`int`):
            Number of heads
        d_head (:obj:`int`):
            Hidden states dimension inside each head
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        only_cross_attention (`bool`, defaults to `False`):
            Whether to only apply cross attention.
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
        use_memory_efficient_attention (`bool`, *optional*, defaults to `False`):
            enable memory efficient attention https://arxiv.org/abs/2112.05682
    """
    dim: int
    n_heads: int
    d_head: int
    dropout: float = 0.0
    only_cross_attention: bool = False
    dtype: jnp.dtype = jnp.float32
    use_memory_efficient_attention: bool = False

    def setup(self):
        # self attention (or cross_attention if only_cross_attention is True)
        self.attn1 = FlaxAttention2(
            self.dim, self.n_heads, self.d_head, self.dropout, self.use_memory_efficient_attention, dtype=self.dtype
        )
        self.ff = FlaxFeedForward(dim=self.dim, dropout=self.dropout, dtype=self.dtype)
        self.norm1 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)
        self.norm2 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)
        self.norm3 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)

    def __call__(self, hidden_states, context, deterministic=True):
        # self attention
        residual = hidden_states
        if self.only_cross_attention:
            hidden_states = self.attn1(self.norm1(hidden_states), context, deterministic=deterministic)
        else:
            hidden_states = self.attn1(self.norm1(hidden_states), deterministic=deterministic)
            
        hidden_states = jnp.einsum("b i j, b j d -> b i d", attention_probs, value_states)

        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        hidden_states = self.proj_attn(hidden_states)

        
        hidden_states = hidden_states + residual

        # cross attention
#         residual = hidden_states
#         hidden_states = self.attn2(self.norm2(hidden_states), context, deterministic=deterministic)
#         hidden_states = hidden_states + residual

        # feed forward
        residual = hidden_states
        hidden_states = self.ff(self.norm3(hidden_states), deterministic=deterministic)
        hidden_states = hidden_states + residual

        return hidden_states
    
def head_to_batch_dim( tensor):
    batch_size, seq_len, dim = tensor.shape
    head_size = self.heads
    tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
    tensor = jnp.transpose(tensor, (0, 2, 1, 3))
    tensor = tensor.reshape(batch_size * head_size, seq_len, dim // head_size)
    return tensor

def batch_to_head_dim( tensor):
    batch_size, seq_len, dim = tensor.shape
    head_size = self.heads
    tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
    tensor = jnp.transpose(tensor, (0, 2, 1, 3))
    tensor = tensor.reshape(batch_size // head_size, seq_len, dim * head_size)
    return tensor
from jax.experimental import io_callback
from jax import debug
import numpy as np
def save_(x,name):
    debug.callback(lambda x: np.save(name,x),x ) #np.save('post_conv1.npy',np.asarray(hidden_states))
    return x

class FlaxBasicTransformerBlock2(nn.Module):
    r"""
    A Flax transformer block layer with `GLU` (Gated Linear Unit) activation function as described in:
    https://arxiv.org/abs/1706.03762


    Parameters:
        dim (:obj:`int`):
            Inner hidden states dimension
        n_heads (:obj:`int`):
            Number of heads
        d_head (:obj:`int`):
            Hidden states dimension inside each head
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        only_cross_attention (`bool`, defaults to `False`):
            Whether to only apply cross attention.
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
        use_memory_efficient_attention (`bool`, *optional*, defaults to `False`):
            enable memory efficient attention https://arxiv.org/abs/2112.05682
    """
    dim: int
    n_heads: int
    d_head: int
    dropout: float = 0.0
    only_cross_attention: bool = False
    dtype: jnp.dtype = jnp.float32
    use_memory_efficient_attention: bool = False

    def setup(self):
        # self attention (or cross_attention if only_cross_attention is True)
        self.attn1 = FlaxAttention2(
            self.dim, self.n_heads, self.d_head, self.dropout, self.use_memory_efficient_attention, dtype=self.dtype
        )
#         self.ff = FlaxFeedForward(dim=self.dim, dropout=self.dropout, dtype=self.dtype)
#         self.norm1 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)
#         self.norm2 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)
#         self.norm3 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)
    def batch_to_head_dim(self, tensor):
        head_size = self.d_head
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = jnp.transpose(tensor,(0, 2, 1, 3)).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def head_to_batch_dim(self, tensor, out_dim=3):
        head_size = self.d_head
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = jnp.transpose(tensor,(0, 2, 1, 3))

        if out_dim == 3:
            tensor = tensor.reshape(batch_size * head_size, seq_len, dim // head_size)

        return tensor

    def __call__(self, hidden_states, encoder_hidden_states,attn, deterministic=True,save=False):
        r2 = hidden_states
        if save:
            save_(hidden_states, 'attn1.npy')
        hidden_states = jnp.transpose(hidden_states, (0,3,1,2))
        residual = hidden_states
        print("hs",hidden_states.shape)
#         encoder_hidden_states = jnp.ones((1,77,4096))
        hidden_states = jnp.transpose(hidden_states.reshape(hidden_states.shape[0], hidden_states.shape[1],hidden_states.shape[-2]* hidden_states.shape[-1]),(0,2, 1))
        if save:
            save_(hidden_states, 'attn2.npy')

        batch_size, sequence_length, _ = hidden_states.shape
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
#             print("encoder size",encoder_hidden_states)
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)
        if save:
            save_(encoder_hidden_states, 'attn2.npy')

        hidden_states = jnp.transpose( jnp.transpose(attn.group_norm(hidden_states),(0,2, 1)), (0,2, 1) )
        if save:
            save_(hidden_states, 'attn3.npy')

        query = attn.to_q(hidden_states)
        if save:
            save_(query, 'attn4.npy')

        print("query v hs", query.shape, hidden_states.shape)
        query = self.head_to_batch_dim(query, out_dim=4)
        if save:
            save_(query, 'attn5.npy')

        print("enc h s",encoder_hidden_states.shape)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        if save:
            save_(encoder_hidden_states_key_proj, 'attn6.npy')

        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
        if save:
            save_(encoder_hidden_states_value_proj, 'attn7.npy')

        encoder_hidden_states_key_proj = self.head_to_batch_dim(encoder_hidden_states_key_proj, out_dim=4)
        if save:
            save_(encoder_hidden_states_key_proj, 'attn8.npy')

        encoder_hidden_states_value_proj = self.head_to_batch_dim(encoder_hidden_states_value_proj, out_dim=4)
        if save:
            save_(encoder_hidden_states_key_proj, 'attn9.npy')

        key = attn.to_k(hidden_states)
        if save:
           save_(key, 'attn10.npy')

        value = attn.to_v(hidden_states)
        if save:
           save_(value, 'attn11.npy')

        key = self.head_to_batch_dim(key, out_dim=4)
        if save:
           save_(value, 'attn12.npy')

        value = self.head_to_batch_dim(value, out_dim=4)
        if save:
           save_(value, 'attn13.npy')

        key = jnp.concatenate([encoder_hidden_states_key_proj, key], axis=2)
        if save:
           save_(key, 'attn14.npy')

        value = jnp.concatenate([encoder_hidden_states_value_proj, value], axis=2)
        if save:
           save_(value, 'attn15.npy')

        attn_weight = nn.softmax(query @ jnp.transpose(key,(0,1,3,2))/jnp.sqrt(query.shape[-1]), axis=-1)
        if save:
           save_(attn_weight, 'attn16.npy')
        hidden_states = attn_weight @ value
        if save:
           save_(hidden_states, 'attn17.npy')
        
        #self.attn1(query, key, value)#attention_mask)
        print("hidden after self attn1",hidden_states.shape)
        fill = 1
        for i in hidden_states.shape:
            fill *= i 
        fill = fill // (batch_size * residual.shape[1])
        hidden_states = jnp.transpose(hidden_states, (0,2, 1,3)).reshape(batch_size, -1, residual.shape[1])
        if save:
           save_(hidden_states, 'attn18.npy')

        print("res shape",residual.shape)
#         fill = 1
#         for i in hidden_states.shape:
#             fill *= i
#         fill = fill//(batch_size * residual.shape[-1])

#         hidden_states = jnp.transpose(hidden_states,(0,2, 1)).reshape(batch_size, hidden_states.shape[2], residual.shape[-1])

#         hidden_states = self.batch_to_head_dim(hidden_states)
        
        hidden_states = attn.to_out[0](hidden_states)
        if save:
           save_(hidden_states, 'attn19.npy')

        hidden_states = jnp.transpose(hidden_states, (0,2, 1)).reshape(residual.shape)

        if save:
           save_(hidden_states, 'attn20.npy')

        hidden_states = hidden_states + residual
        if save:
           save_(hidden_states, 'attn21.npy')

        return hidden_states.reshape(r2.shape)

class FlaxTransformer2DModel2(nn.Module):
    r"""
    A Spatial Transformer layer with Gated Linear Unit (GLU) activation function as described in:
    https://arxiv.org/pdf/1506.02025.pdf


    Parameters:
        in_channels (:obj:`int`):
            Input number of channels
        n_heads (:obj:`int`):
            Number of heads
        d_head (:obj:`int`):
            Hidden states dimension inside each head
        depth (:obj:`int`, *optional*, defaults to 1):
            Number of transformers block
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        use_linear_projection (`bool`, defaults to `False`): tbd
        only_cross_attention (`bool`, defaults to `False`): tbd
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
        use_memory_efficient_attention (`bool`, *optional*, defaults to `False`):
            enable memory efficient attention https://arxiv.org/abs/2112.05682
    """
    in_channels: int
    n_heads: int
    d_head: int
    depth: int = 1
    dropout: float = 0.0
    use_linear_projection: bool = False
    only_cross_attention: bool = False
    dtype: jnp.dtype = jnp.float32
    use_memory_efficient_attention: bool = False

    def setup(self):
#         self.norm_cross = nn.GroupNorm(num_groups=32, epsilon=1e-5)

        inner_dim = self.n_heads * self.d_head
#         if self.use_linear_projection:
#             self.proj_in = nn.Dense(inner_dim, dtype=self.dtype)
#         else:
#             self.proj_in = nn.Conv(
#                 inner_dim,
#                 kernel_size=(1, 1),
#                 strides=(1, 1),
#                 padding="VALID",
#                 dtype=self.dtype,
        self.to_out = [nn.Dense(inner_dim)]
#         self.to_out.append()
#         self.to_out.append(nn.Dropout(dropout))
        self.to_k = nn.Dense(inner_dim)
        self.to_q = nn.Dense(inner_dim)
        self.to_v = nn.Dense(inner_dim)
        self.group_norm = nn.GroupNorm(num_groups=32, epsilon=1e-5)
        self.norm_cross = nn.GroupNorm(num_groups=32, epsilon=1e-5)
        self.add_k_proj = nn.Dense(inner_dim)
        self.add_v_proj = nn.Dense(inner_dim)

        self.transformer_blocks = [
            FlaxBasicTransformerBlock2(
                inner_dim,
                self.n_heads,
                self.d_head,
                dropout=self.dropout,
                only_cross_attention=self.only_cross_attention,
                dtype=self.dtype,
                use_memory_efficient_attention=self.use_memory_efficient_attention,
            )
            for _ in range(1)
        ]

#         if self.use_linear_projection:
#             self.to_out = nn.Dense(inner_dim, dtype=self.dtype)
#         else:
#             self.to_out = nn.Conv(
#                 inner_dim,
#                 kernel_size=(1, 1),
#                 strides=(1, 1),
#                 padding="VALID",
#                 dtype=self.dtype,
#             )

    def __call__(self, hidden_states, context, deterministic=True,save=False):
#         batch, height, width, channels = hidden_states.shape
#         residual = hidden_states
#         hidden_states = self.norm_cross(hidden_states)
#         if self.use_linear_projection:
#             hidden_states = hidden_states.reshape(batch, height * width, channels)
#             hidden_states = self.proj_in(hidden_states)
#         else:
#             hidden_states = self.proj_in(hidden_states)
#             hidden_states = hidden_states.reshape(batch, height * width, channels)
        r = hidden_states
        for transformer_block in self.transformer_blocks:
            hidden_states = transformer_block(hidden_states, context,self, deterministic=deterministic,save=save)

#         if self.use_linear_projection:
#             hidden_states = self.to_out(hidden_states)
#             hidden_states = hidden_states.reshape(batch, height, width, channels)
#         else:
#             hidden_states = hidden_states.reshape(batch, height, width, channels)
#             hidden_states = self.to_out(hidden_states)

#         hidden_states = hidden_states + residual
        return hidden_states.reshape(r.shape)

class FlaxTransformer2DModel(nn.Module):
    r"""
    A Spatial Transformer layer with Gated Linear Unit (GLU) activation function as described in:
    https://arxiv.org/pdf/1506.02025.pdf


    Parameters:
        in_channels (:obj:`int`):
            Input number of channels
        n_heads (:obj:`int`):
            Number of heads
        d_head (:obj:`int`):
            Hidden states dimension inside each head
        depth (:obj:`int`, *optional*, defaults to 1):
            Number of transformers block
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        use_linear_projection (`bool`, defaults to `False`): tbd
        only_cross_attention (`bool`, defaults to `False`): tbd
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
        use_memory_efficient_attention (`bool`, *optional*, defaults to `False`):
            enable memory efficient attention https://arxiv.org/abs/2112.05682
    """
    in_channels: int
    n_heads: int
    d_head: int
    depth: int = 1
    dropout: float = 0.0
    use_linear_projection: bool = False
    only_cross_attention: bool = False
    dtype: jnp.dtype = jnp.float32
    use_memory_efficient_attention: bool = False

    def setup(self):
        self.norm_cross = nn.GroupNorm(num_groups=32, epsilon=1e-5)

        inner_dim = self.n_heads * self.d_head
        if self.use_linear_projection:
            self.proj_in = nn.Dense(inner_dim, dtype=self.dtype)
        else:
            self.proj_in = nn.Conv(
                inner_dim,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="VALID",
                dtype=self.dtype,
            )

        self.transformer_blocks = [
            FlaxBasicTransformerBlock2(
                inner_dim,
                self.n_heads,
                self.d_head,
                dropout=self.dropout,
                only_cross_attention=self.only_cross_attention,
                dtype=self.dtype,
                use_memory_efficient_attention=self.use_memory_efficient_attention,
            )
            for _ in range(self.depth)
        ]

        if self.use_linear_projection:
            self.to_out = nn.Dense(inner_dim, dtype=self.dtype)
        else:
            self.to_out = nn.Conv(
                inner_dim,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="VALID",
                dtype=self.dtype,
            )

    def __call__(self, hidden_states, context, deterministic=True):
#         batch, height, width, channels = hidden_states.shape
#         residual = hidden_states
#         hidden_states = self.norm_cross(hidden_states)
#         if self.use_linear_projection:
#             hidden_states = hidden_states.reshape(batch, height * width, channels)
#             hidden_states = self.proj_in(hidden_states)
#         else:
#             hidden_states = self.proj_in(hidden_states)
#             hidden_states = hidden_states.reshape(batch, height * width, channels)

        for transformer_block in self.transformer_blocks:
            hidden_states = transformer_block(hidden_states, context, deterministic=deterministic)

        if self.use_linear_projection:
            hidden_states = self.to_out(hidden_states)
            hidden_states = hidden_states.reshape(batch, height, width, channels)
        else:
            hidden_states = hidden_states.reshape(batch, height, width, channels)
            hidden_states = self.to_out(hidden_states)

        hidden_states = hidden_states + residual
        return hidden_states


class FlaxFeedForward(nn.Module):
    r"""
    Flax module that encapsulates two Linear layers separated by a non-linearity. It is the counterpart of PyTorch's
    [`FeedForward`] class, with the following simplifications:
    - The activation function is currently hardcoded to a gated linear unit from:
    https://arxiv.org/abs/2002.05202
    - `dim_out` is equal to `dim`.
    - The number of hidden dimensions is hardcoded to `dim * 4` in [`FlaxGELU`].

    Parameters:
        dim (:obj:`int`):
            Inner hidden states dimension
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """
    dim: int
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # The second linear layer needs to be called
        # net_2 for now to match the index of the Sequential layer
        self.net_0 = FlaxGEGLU(self.dim, self.dropout, self.dtype)
        self.net_2 = nn.Dense(self.dim, dtype=self.dtype)

    def __call__(self, hidden_states, deterministic=True):
        hidden_states = self.net_0(hidden_states)
        hidden_states = self.net_2(hidden_states)
        return hidden_states


class FlaxGEGLU(nn.Module):
    r"""
    Flax implementation of a Linear layer followed by the variant of the gated linear unit activation function from
    https://arxiv.org/abs/2002.05202.

    Parameters:
        dim (:obj:`int`):
            Input hidden states dimension
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """
    dim: int
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        inner_dim = self.dim * 4
        self.proj = nn.Dense(inner_dim * 2, dtype=self.dtype)

    def __call__(self, hidden_states, deterministic=True):
        hidden_states = self.proj(hidden_states)
        hidden_linear, hidden_gelu = jnp.split(hidden_states, 2, axis=2)
        return hidden_linear * nn.gelu(hidden_gelu)
