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
import math

import flax.linen as nn
import jax.numpy as jnp

from jax.experimental import io_callback
from jax import debug
import numpy as np
def save_(x,name):
    debug.callback(lambda x: np.save(name,x),x ) #np.save('post_conv1.npy',np.asarray(hidden_states))
    return x

import numpy as np
def get_sinusoidal_embeddings(
    timesteps: jnp.ndarray,
    embedding_dim: int,
    freq_shift: float = 1,
    min_timescale: float = 1,
    max_timescale: float = 1.0e4,
    flip_sin_to_cos: bool = False,
    scale: float = 1.0,
) -> jnp.ndarray:
    """Returns the positional encoding (same as Tensor2Tensor).

    Args:
        timesteps: a 1-D Tensor of N indices, one per batch element.
        These may be fractional.
        embedding_dim: The number of output channels.
        min_timescale: The smallest time unit (should probably be 0.0).
        max_timescale: The largest time unit.
    Returns:
        a Tensor of timing signals [N, num_channels]
    """
    assert timesteps.ndim == 1, "Timesteps should be a 1d-array"
    assert embedding_dim % 2 == 0, f"Embedding dimension {embedding_dim} should be even"
    num_timescales = float(embedding_dim // 2)
    log_timescale_increment = math.log(max_timescale / min_timescale) / (num_timescales - freq_shift)
    inv_timescales = min_timescale * jnp.exp(jnp.arange(num_timescales, dtype=jnp.float32) * -log_timescale_increment)
    emb = jnp.expand_dims(timesteps, 1) * jnp.expand_dims(inv_timescales, 0)

    # scale embeddings
    scaled_time = scale * emb

    if flip_sin_to_cos:
        signal = jnp.concatenate([jnp.cos(scaled_time), jnp.sin(scaled_time)], axis=1)
    else:
        signal = jnp.concatenate([jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=1)
    signal = jnp.reshape(signal, [jnp.shape(timesteps)[0], embedding_dim])
    return signal


class FlaxTimestepEmbedding(nn.Module):
    r"""
    Time step Embedding Module. Learns embeddings for input time steps.

    Args:
        time_embed_dim (`int`, *optional*, defaults to `32`):
                Time step embedding dimension
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
                Parameters `dtype`
    """
    time_embed_dim: int = 32
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, temb):
        temb = nn.Dense(self.time_embed_dim, dtype=self.dtype, name="linear_1")(temb)
        temb = nn.silu(temb)
        temb = nn.Dense(self.time_embed_dim, dtype=self.dtype, name="linear_2")(temb)
        return temb
import numpy as np

class FlaxTimesteps(nn.Module):
    r"""
    Wrapper Module for sinusoidal Time step Embeddings as described in https://arxiv.org/abs/2006.11239

    Args:
        dim (`int`, *optional*, defaults to `32`):
                Time step embedding dimension
    """
    dim: int = 32
    flip_sin_to_cos: bool = False
    freq_shift: float = 1

    @nn.compact
    def __call__(self, timesteps):
        return get_sinusoidal_embeddings(
            timesteps, embedding_dim=self.dim, flip_sin_to_cos=self.flip_sin_to_cos, freq_shift=self.freq_shift
        )

class FlaxTextTimeEmbedding(nn.Module):
#     def __init__(self, encoder_dim: int, time_embed_dim: int, num_heads: int = 64):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(encoder_dim)
#         self.pool = FlaxAttentionPooling(num_heads, encoder_dim)
#         self.proj = nn.Dense(encoder_dim, time_embed_dim)
#         self.norm2 = nn.LayerNorm(time_embed_dim)
    
    encoder_dim: int = 0
    time_embed_dim: int = 0
    num_heads: int = 0

    def setup(self):
        self.norm1 = nn.LayerNorm()
        self.pool = FlaxAttentionPooling(num_heads=self.num_heads, embed_dim=self.encoder_dim)
        self.proj = nn.Dense(self.time_embed_dim)
        self.norm2 = nn.LayerNorm()

    def __call__(self, hidden_states):
        print("h0 ",hidden_states,hidden_states.shape)
        print("h0 vars",jnp.var(hidden_states[0,0,:]) , jnp.mean(hidden_states[0,0,:]) )

        hidden_states = self.norm1(hidden_states.astype(jnp.float32))
        print("h1 ", hidden_states)
        print("h1 vars",jnp.var(hidden_states[0,0,:]) , jnp.mean(hidden_states[0,0,:]) )
#         np.save('hidden_states.npy',hidden_states)
        hidden_states = self.pool(hidden_states)
        hidden_states = self.proj(hidden_states)
        hidden_states = self.norm2(hidden_states)
        return hidden_states


class FlaxAttentionPooling(nn.Module):
    # Copied from https://github.com/deep-floyd/IF/blob/2f91391f27dd3c468bf174be5805b4cc92980c0b/deepfloyd_if/model/nn.py#L54

#     def __init__(self, num_heads, embed_dim, dtype=None):
#         super().__init__()
# #         self.dtype = dtype
#         self.positional_embedding = jnp.asarray(np.load('positional_embedding.npy'))#jax.random.normal(prng_seed, shape=latents_shape, dtype=jnp.float32) #nn.Parameter(torch.randn(1, embed_dim) / embed_dim**0.5)
#         self.k_proj = nn.Dense(embed_dim)#, embed_dim, dtype=self.dtype)
#         self.q_proj = nn.Dense(embed_dim)#, embed_dim, dtype=self.dtype)
#         self.v_proj = nn.Dense(embed_dim)#, embed_dim, dtype=self.dtype)
#         self.num_heads = num_heads
#         self.dim_per_head = embed_dim // self.num_heads
    num_heads: int = 0
    embed_dim: int = 0
    
    def setup(self):
        self.positional_embedding = jnp.asarray(np.load('positional_embedding.npy'))#jax.random.normal(prng_seed, shape=latents_shape, dtype=jnp.float32) #nn.Parameter(torch.randn(1, embed_dim) / embed_dim**0.5)
        self.k_proj = nn.Dense(self.embed_dim)#, embed_dim, dtype=self.dtype)
        self.q_proj = nn.Dense(self.embed_dim)#, embed_dim, dtype=self.dtype)
        self.v_proj = nn.Dense(self.embed_dim)#, embed_dim, dtype=self.dtype)
#         self.num_heads = self.num_heads
        self.dim_per_head = self.embed_dim // self.num_heads

    
    def __call__(self, x):
        print("raw incoming",x.shape,self.embed_dim)
        bs, length, width = x.shape#()
        raw_width = width

        og = x
        print("pre length",length)
        def shape(x):
            og = x
            bs, length, width = x.shape#()

            print("x in shape " , x.shape )
            print("length in shape", length)
            # (bs, length, width) --> (rrribs, length, n_heads, dim_per_head)
#             fill = (bs*length *width)//(length*self.num_heads*self.dim_per_head)
            x = x.reshape(bs, length, self.num_heads, self.dim_per_head)
            # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
            x = jnp.transpose(x, (0, 2,1,3))
            # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
#             fill = (bs*self.n_heads *self.dim_per_head)//(length*self.num_heads*self.dim_per_head)

            x = x.reshape(bs * self.num_heads, length, self.dim_per_head)
            # (bs*n_heads, length, dim_per_head) --> (bs*n_heads, dim_per_head, length)
            x = jnp.transpose(x,(0,2, 1))
            return x
        print("x mean vs self pos",jnp.mean(x,axis=1, keepdims=True).shape ,self.positional_embedding.shape)
        if raw_width > 2816:
#             save_( 'raw_greater_2816.npy' , self.positional_embedding)
            class_token = jnp.mean(x,axis=1, keepdims=True) + self.positional_embedding
        else:
            class_token = jnp.mean(x,axis=1, keepdims=True)
        print("class token size",class_token.shape)
        x = jnp.concatenate([class_token, x], axis=1)  # (bs, length+1, width)

        # (bs*n_heads, class_token_length, dim_per_head)
        q = shape(self.q_proj(class_token))
        print("q",q.shape)
        # (bs*n_heads, length+class_token_length, dim_per_head)
        k = shape(self.k_proj(x))
        print("k",k.shape)
        v = shape(self.v_proj(x))
        print("v",v.shape)
        # (bs*n_heads, class_token_length, length+class_token_length):
        scale = 1 / math.sqrt(math.sqrt(self.dim_per_head))
        weight = jnp.einsum("bct,bcs->bts", q * scale, k * scale)  # More stable with f16 than dividing afterwards
        print("weight 1 ",weight.shape)
        weight = nn.softmax(weight.astype(jnp.float32), axis=-1)#.type(weight.dtype)
        print("weight 2 ",weight.shape)
        # (bs*n_heads, dim_per_head, class_token_length)
        a = jnp.einsum("bts,bcs->bct", weight, v)
        print("a 1",a.shape)
        # (bs, length+1, width)
        if raw_width > 768:
            fill = 1
            for i in a.shape:
                print("dis i", i)
                fill = fill * i
            fill = fill//bs
            print("fill ->",fill)
            a = jnp.transpose(a.reshape(bs, fill, 1),(0,2, 1))
            print("a 2", a.shape)
            return a[:, 0, :]  # cls_token
        else:
            return og
