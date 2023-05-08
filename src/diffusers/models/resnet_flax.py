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
import flax.linen as nn
import jax
import jax.numpy as jnp


class FlaxUpsample2D(nn.Module):
    out_channels: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.conv = nn.Conv(
            self.out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

    def __call__(self, hidden_states):
        batch, height, width, channels = hidden_states.shape
        hidden_states = jax.image.resize(
            hidden_states,
            shape=(batch, height * 2, width * 2, channels),
            method="nearest",
        )
        hidden_states = self.conv(hidden_states)
        return hidden_states

class FlaxUpsample2D2(nn.Module):
    out_channels: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.conv = nn.Conv(
            self.out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

    def __call__(self, hidden_states):
        batch, height, width, channels = hidden_states.shape
        hidden_states = jax.image.resize(
            hidden_states,
            shape=(batch, height * 2, width * 2, channels),
            method="nearest",
        )
        
        
        
        return hidden_states

class FlaxDownsample2D(nn.Module):
    out_channels: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.conv = nn.Conv(
            self.out_channels,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=((1, 1), (1, 1)),  # padding="VALID",
            dtype=self.dtype,
        )

    def __call__(self, hidden_states):
        # pad = ((0, 0), (0, 1), (0, 1), (0, 0))  # pad height and width dim
        # hidden_states = jnp.pad(hidden_states, pad_width=pad)
        hidden_states = self.conv(hidden_states)
        return hidden_states

class FlaxResnetBlock2D(nn.Module):
    in_channels: int
    out_channels: int = None
    dropout_prob: float = 0.0
    use_nin_shortcut: bool = None
    dtype: jnp.dtype = jnp.float32
    downsample: bool = None
    upsample: bool = None

    def setup(self):
        out_channels = self.in_channels if self.out_channels is None else self.out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, epsilon=1e-5)
        self.conv1 = nn.Conv(
            out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

        self.time_emb_proj = nn.Dense(2*out_channels, dtype=self.dtype)

        self.norm2 = nn.GroupNorm(num_groups=32, epsilon=1e-5)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.conv2 = nn.Conv(
            out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

        use_nin_shortcut = self.in_channels != out_channels if self.use_nin_shortcut is None else self.use_nin_shortcut

        self.conv_shortcut = None
        if use_nin_shortcut:
            self.conv_shortcut = nn.Conv(
                out_channels,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="VALID",
                dtype=self.dtype,
            )
        if self.downsample:
#             self.downsample = FlaxDownsample2D(pool=True)
            self.down = True
        elif self.upsample:
            self.up = True

    def __call__(self, hidden_states, temb, deterministic=True):
        residual = hidden_states
#         print("residual -------->",residual)
        hidden_states = self.norm1(hidden_states)
#         print("post norm1 ----->",hidden_states)
        hidden_states = nn.swish(hidden_states)
        if self.downsample:
            hidden_states = nn.avg_pool(hidden_states,window_shape=(2,2),strides=(2,2))
            residual = nn.avg_pool(residual,window_shape=(2,2),strides=(2,2))
        if self.upsample:
            batch, height, width, channels = hidden_states.shape
            hidden_states = jax.image.resize(
                hidden_states,
                shape=(batch, height * 2, width * 2, channels),
                method="nearest",
            )
            residual = jax.image.resize(
                residual,
                shape=(batch, height * 2, width * 2, channels),
                method="nearest",
            )


            # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
#             if hidden_states.shape[0] >= 64:
#                 input_tensor = input_tensor.contiguous()
#                 hidden_states = hidden_states.contiguous()
#         print("pre conv1 -------------------------------------->",hidden_states)

        hidden_states = self.conv1(hidden_states)
#         print("post conv1 -------------------------------------->",hidden_states)

        temb = self.time_emb_proj(nn.swish(temb))
        
        

#         temb = jnp.expand_dims(jnp.expand_dims(temb, 1), 1)
#         hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)
        scale, shift = jnp.split(temb, 2, axis=1)
        hidden_states = hidden_states * (1 +  jnp.expand_dims(jnp.expand_dims(scale,axis=1),axis=2)) + jnp.expand_dims(jnp.expand_dims(shift,axis=1),axis=2)

        hidden_states = nn.swish(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)

        return hidden_states + residual
