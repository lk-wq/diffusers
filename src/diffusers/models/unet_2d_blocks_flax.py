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
import jax.numpy as jnp

from .attention_flax import FlaxTransformer2DModel, FlaxTransformer2DModel2
from .resnet_flax import FlaxDownsample2D, FlaxResnetBlock2D, FlaxUpsample2D
import numpy as np
from jax.experimental import io_callback
from jax import debug
def save_(x,name):
    debug.callback(lambda x: np.save(name,x),x ) #np.save('post_conv1.npy',np.asarray(hidden_states))
    return x

class FlaxCrossAttnDownBlock2D(nn.Module):
    r"""
    Cross Attention 2D Downsizing block - original architecture from Unet transformers:
    https://arxiv.org/abs/2103.06104

    Parameters:
        in_channels (:obj:`int`):
            Input channels
        out_channels (:obj:`int`):
            Output channels
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        num_layers (:obj:`int`, *optional*, defaults to 1):
            Number of attention blocks layers
        attn_num_head_channels (:obj:`int`, *optional*, defaults to 1):
            Number of attention heads of each spatial transformer block
        add_downsample (:obj:`bool`, *optional*, defaults to `True`):
            Whether to add downsampling layer before each final output
        use_memory_efficient_attention (`bool`, *optional*, defaults to `False`):
            enable memory efficient attention https://arxiv.org/abs/2112.05682
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """
    in_channels: int
    out_channels: int
    dropout: float = 0.0
    num_layers: int = 1
    attn_num_head_channels: int = 1
    add_downsample: bool = True
    use_linear_projection: bool = False
    only_cross_attention: bool = False
    use_memory_efficient_attention: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        resnets = []
        attentions = []

        for i in range(self.num_layers):
            in_channels = self.in_channels if i == 0 else self.out_channels

            res_block = FlaxResnetBlock2D(
                in_channels=in_channels,
                out_channels=self.out_channels,
                dropout_prob=self.dropout,
                dtype=self.dtype,
            )
            resnets.append(res_block)

            attn_block = FlaxTransformer2DModel2(
                in_channels=self.out_channels,
                n_heads=self.attn_num_head_channels,
                d_head=self.out_channels // self.attn_num_head_channels,
                depth=1,
                use_linear_projection=self.use_linear_projection,
                only_cross_attention=self.only_cross_attention,
                use_memory_efficient_attention=self.use_memory_efficient_attention,
                dtype=self.dtype,
            )
            attentions.append(attn_block)

        self.resnets = resnets
        self.attentions = attentions

        if self.add_downsample:
            self.downsamplers_0 = FlaxResnetBlock2D(
                in_channels=in_channels,
                out_channels=self.out_channels,
                dropout_prob=self.dropout,
                downsample=True,
                dtype=self.dtype,
            )#FlaxDownsample2D(self.out_channels, dtype=self.dtype)

    def __call__(self, hidden_states, temb, encoder_hidden_states, deterministic=True,save=False):
        output_states = ()

        for ix, (resnet, attn) in enumerate(zip(self.resnets, self.attentions)):
            if ix == 1 and save:
#                 print("pre attn",ix,jnp.transpose( hidden_states, (0,2,3,1)  ),hidden_states.shape)
                save = True
                print("saving blocks ------------------------------------------------------->")
                hidden_states = resnet(hidden_states, temb, deterministic=deterministic,save=save)

            else:
                
                hidden_states = resnet(hidden_states, temb, deterministic=deterministic)
                
#             if ix == 0 and save:
#                 save_(hidden_states,'0_states.npy')
            if ix == 0 and save:
                hidden_states = attn(hidden_states, encoder_hidden_states, deterministic=deterministic,save=save)
            else:
                hidden_states = attn(hidden_states, encoder_hidden_states, deterministic=deterministic)

#             hidden_states = jnp.transpose(hidden_states,(0,2,1,3))
#             if ix == 0 and save:
#                 save_(hidden_states,'0_states2.npy')

#             if ix == 1:
#                 print("post attn",ix,jnp.transpose( hidden_states, (0,2,3,1)  ),hidden_states.shape)
#             hidden_states = jnp.transpose(hidden_states,(0,2,1,3))
            output_states += (hidden_states,)

        if self.add_downsample:
            hidden_states = self.downsamplers_0(hidden_states,temb,deterministic=deterministic)
#             print("post down sample",hidden_states(
            output_states += (hidden_states,)

        return hidden_states, output_states


class FlaxDownBlock2D(nn.Module):
    r"""
    Flax 2D downsizing block

    Parameters:
        in_channels (:obj:`int`):
            Input channels
        out_channels (:obj:`int`):
            Output channels
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        num_layers (:obj:`int`, *optional*, defaults to 1):
            Number of attention blocks layers
        add_downsample (:obj:`bool`, *optional*, defaults to `True`):
            Whether to add downsampling layer before each final output
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """
    in_channels: int
    out_channels: int
    dropout: float = 0.0
    num_layers: int = 1
    add_downsample: bool = True
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        resnets = []

        for i in range(self.num_layers):
            in_channels = self.in_channels if i == 0 else self.out_channels

            res_block = FlaxResnetBlock2D(
                in_channels=in_channels,
                out_channels=self.out_channels,
                dropout_prob=self.dropout,
                downsample=False,
                dtype=self.dtype,
            )
            resnets.append(res_block)
        self.resnets = resnets

        if self.add_downsample:
            self.downsamplers_0 = FlaxResnetBlock2D(
                in_channels=in_channels,
                out_channels=self.out_channels,
                dropout_prob=self.dropout,
                downsample=True,
                dtype=self.dtype,
            )#FlaxDownsample2D(self.out_channels, dtype=self.dtype)

    def __call__(self, hidden_states, temb, deterministic=True):
        output_states = ()

        for ix, resnet in enumerate(self.resnets):
            if ix == 2:
                hidden_states = resnet(hidden_states, temb, deterministic=deterministic,display=True)
            else:
                hidden_states = resnet(hidden_states, temb, deterministic=deterministic)

            if ix == 1:
                print("rezzy 0 ------------------------------------------------------>",hidden_states)
            output_states += (hidden_states,)
#         save_(hidden_states,'fin.npy')
        if self.add_downsample:
            hidden_states = self.downsamplers_0(hidden_states,temb,deterministic=deterministic)
            output_states += (hidden_states,)
#         save_(hidden_states,'fin_d.npy')

        return hidden_states, output_states


class FlaxCrossAttnUpBlock2D(nn.Module):
    r"""
    Cross Attention 2D Upsampling block - original architecture from Unet transformers:
    https://arxiv.org/abs/2103.06104

    Parameters:
        in_channels (:obj:`int`):
            Input channels
        out_channels (:obj:`int`):
            Output channels
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        num_layers (:obj:`int`, *optional*, defaults to 1):
            Number of attention blocks layers
        attn_num_head_channels (:obj:`int`, *optional*, defaults to 1):
            Number of attention heads of each spatial transformer block
        add_upsample (:obj:`bool`, *optional*, defaults to `True`):
            Whether to add upsampling layer before each final output
        use_memory_efficient_attention (`bool`, *optional*, defaults to `False`):
            enable memory efficient attention https://arxiv.org/abs/2112.05682
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """
    in_channels: int
    out_channels: int
    prev_output_channel: int
    dropout: float = 0.0
    num_layers: int = 1
    attn_num_head_channels: int = 1
    add_upsample: bool = True
    use_linear_projection: bool = False
    only_cross_attention: bool = False
    use_memory_efficient_attention: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        resnets = []
        attentions = []

        for i in range(self.num_layers):
            res_skip_channels = self.in_channels if (i == self.num_layers - 1) else self.out_channels
            resnet_in_channels = self.prev_output_channel if i == 0 else self.out_channels

            res_block = FlaxResnetBlock2D(
                in_channels=resnet_in_channels + res_skip_channels,
                out_channels=self.out_channels,
                dropout_prob=self.dropout,
                dtype=self.dtype,
            )
            resnets.append(res_block)

            attn_block = FlaxTransformer2DModel2(
                in_channels=self.out_channels,
                n_heads=self.attn_num_head_channels,
                d_head=self.out_channels // self.attn_num_head_channels,
                depth=1,
                use_linear_projection=self.use_linear_projection,
                only_cross_attention=self.only_cross_attention,
                use_memory_efficient_attention=self.use_memory_efficient_attention,
                dtype=self.dtype,
            )
            attentions.append(attn_block)

        self.resnets = resnets
        self.attentions = attentions
        if self.add_upsample:
            self.upsamplers_0 = FlaxResnetBlock2D(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                dropout_prob=self.dropout,
                upsample=True,
                dtype=self.dtype,
            )#FlaxUpsample2D(self.out_channels, dtype=self.dtype)

    def __call__(self, hidden_states, res_hidden_states_tuple, temb, encoder_hidden_states, deterministic=True):
        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = jnp.concatenate((hidden_states, res_hidden_states), axis=-1)

            hidden_states = resnet(hidden_states, temb, deterministic=deterministic)
            hidden_states = attn(hidden_states, encoder_hidden_states, deterministic=deterministic)

        if self.add_upsample:
            hidden_states = self.upsamplers_0(hidden_states,temb,deterministic=deterministic)

        return hidden_states


class FlaxUpBlock2D(nn.Module):
    r"""
    Flax 2D upsampling block

    Parameters:
        in_channels (:obj:`int`):
            Input channels
        out_channels (:obj:`int`):
            Output channels
        prev_output_channel (:obj:`int`):
            Output channels from the previous block
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        num_layers (:obj:`int`, *optional*, defaults to 1):
            Number of attention blocks layers
        add_downsample (:obj:`bool`, *optional*, defaults to `True`):
            Whether to add downsampling layer before each final output
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """
    in_channels: int
    out_channels: int
    prev_output_channel: int
    dropout: float = 0.0
    num_layers: int = 1
    add_upsample: bool = True
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        resnets = []

        for i in range(self.num_layers):
            res_skip_channels = self.in_channels if (i == self.num_layers - 1) else self.out_channels
            resnet_in_channels = self.prev_output_channel if i == 0 else self.out_channels

            res_block = FlaxResnetBlock2D(
                in_channels=resnet_in_channels + res_skip_channels,
                out_channels=self.out_channels,
                dropout_prob=self.dropout,
                dtype=self.dtype,
            )
            resnets.append(res_block)

        self.resnets = resnets

        if self.add_upsample:
            self.upsamplers_0 = FlaxResnetBlock2D(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                dropout_prob=self.dropout,
                upsample=True,
                dtype=self.dtype,
            )

    def __call__(self, hidden_states, res_hidden_states_tuple, temb, deterministic=True):
        for resnet in self.resnets:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = jnp.concatenate((hidden_states, res_hidden_states), axis=-1)

            hidden_states = resnet(hidden_states, temb, deterministic=deterministic)

        if self.add_upsample:
            hidden_states = self.upsamplers_0(hidden_states,temb,deterministic=deterministic)

        return hidden_states


class FlaxUNetMidBlock2DCrossAttn(nn.Module):
    r"""
    Cross Attention 2D Mid-level block - original architecture from Unet transformers: https://arxiv.org/abs/2103.06104

    Parameters:
        in_channels (:obj:`int`):
            Input channels
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        num_layers (:obj:`int`, *optional*, defaults to 1):
            Number of attention blocks layers
        attn_num_head_channels (:obj:`int`, *optional*, defaults to 1):
            Number of attention heads of each spatial transformer block
        use_memory_efficient_attention (`bool`, *optional*, defaults to `False`):
            enable memory efficient attention https://arxiv.org/abs/2112.05682
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """
    in_channels: int
    dropout: float = 0.0
    num_layers: int = 1
    attn_num_head_channels: int = 1
    use_linear_projection: bool = False
    use_memory_efficient_attention: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # there is always at least one resnet
        resnets = [
            FlaxResnetBlock2D(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                dropout_prob=self.dropout,
                dtype=self.dtype,
            )
        ]

        attentions = []

        for _ in range(self.num_layers):
            attn_block = FlaxTransformer2DModel2(
                in_channels=self.in_channels,
                n_heads=self.attn_num_head_channels,
                d_head=self.in_channels // self.attn_num_head_channels,
                depth=1,
                use_linear_projection=self.use_linear_projection,
                use_memory_efficient_attention=self.use_memory_efficient_attention,
                dtype=self.dtype,
            )
            attentions.append(attn_block)

            res_block = FlaxResnetBlock2D(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                dropout_prob=self.dropout,
                dtype=self.dtype,
            )
            resnets.append(res_block)

        self.resnets = resnets
        self.attentions = attentions

    def __call__(self, hidden_states, temb, encoder_hidden_states, deterministic=True):
        hidden_states = self.resnets[0](hidden_states, temb)
#         save_(hidden_states,'sample_mid_0.npy')
        for ix, (attn, resnet) in enumerate(zip(self.attentions, self.resnets[1:])):
            hidden_states = attn(hidden_states, encoder_hidden_states, deterministic=deterministic)
#             save_(hidden_states,'sample_mid_postattn_'+str(ix)+'.npy')

            hidden_states = resnet(hidden_states, temb, deterministic=deterministic)
#             save_(hidden_states,'sample_mid_postresnet_'+str(ix)+'.npy')

        return hidden_states
