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
from typing import Tuple, Union

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from ..configuration_utils import ConfigMixin, flax_register_to_config
from ..utils import BaseOutput
from .embeddings_flax import FlaxTimestepEmbedding, FlaxTimesteps,FlaxTextTimeEmbedding
from .modeling_flax_utils import FlaxModelMixin
from .unet_2d_blocks_flax import (
    FlaxCrossAttnDownBlock2D,
    FlaxCrossAttnUpBlock2D,
    FlaxDownBlock2D,
    FlaxUNetMidBlock2DCrossAttn,
    FlaxUpBlock2D,
)

from jax.experimental import io_callback
from jax import debug
import numpy as np
def save_(x,name):
    debug.callback(lambda x: np.save(name,x),x ) #np.save('post_conv1.npy',np.asarray(hidden_states))
    return x

@flax.struct.dataclass
class FlaxUNet2DConditionOutput(BaseOutput):
    """
    Args:
        sample (`jnp.ndarray` of shape `(batch_size, num_channels, height, width)`):
            Hidden states conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: jnp.ndarray


@flax_register_to_config
class FlaxUNet2DConditionModel(nn.Module, FlaxModelMixin, ConfigMixin):
    r"""
    FlaxUNet2DConditionModel is a conditional 2D UNet model that takes in a noisy sample, conditional state, and a
    timestep and returns sample shaped output.

    This model inherits from [`FlaxModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the models (such as downloading or saving, etc.)

    Also, this model is a Flax Linen [flax.linen.Module](https://flax.readthedocs.io/en/latest/flax.linen.html#module)
    subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to
    general usage and behavior.

    Finally, this model supports inherent JAX features such as:
    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

    Parameters:
        sample_size (`int`, *optional*):
            The size of the input sample.
        in_channels (`int`, *optional*, defaults to 4):
            The number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 4):
            The number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D")`):
            The tuple of downsample blocks to use. The corresponding class names will be: "FlaxCrossAttnDownBlock2D",
            "FlaxCrossAttnDownBlock2D", "FlaxCrossAttnDownBlock2D", "FlaxDownBlock2D"
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D",)`):
            The tuple of upsample blocks to use. The corresponding class names will be: "FlaxUpBlock2D",
            "FlaxCrossAttnUpBlock2D", "FlaxCrossAttnUpBlock2D", "FlaxCrossAttnUpBlock2D"
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        attention_head_dim (`int` or `Tuple[int]`, *optional*, defaults to 8):
            The dimension of the attention heads.
        cross_attention_dim (`int`, *optional*, defaults to 768):
            The dimension of the cross attention features.
        dropout (`float`, *optional*, defaults to 0):
            Dropout probability for down, up and bottleneck blocks.
        flip_sin_to_cos (`bool`, *optional*, defaults to `True`):
            Whether to flip the sin to cos in the time embedding.
        freq_shift (`int`, *optional*, defaults to 0): The frequency shift to apply to the time embedding.
        use_memory_efficient_attention (`bool`, *optional*, defaults to `False`):
            enable memory efficient attention https://arxiv.org/abs/2112.05682

    """

    sample_size: int = 32
    in_channels: int = 4
    out_channels: int = 4
    down_block_types: Tuple[str] = (
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D",
    )
    up_block_types: Tuple[str] = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D")
    only_cross_attention: Union[bool, Tuple[bool]] = False
    block_out_channels: Tuple[int] = (320, 640, 1280, 1280)
    layers_per_block: int = 2
    attention_head_dim: Union[int, Tuple[int]] = 8
    cross_attention_dim: int = 1280
    dropout: float = 0.0
    use_linear_projection: bool = False
    dtype: jnp.dtype = jnp.float32
    flip_sin_to_cos: bool = True
    freq_shift: int = 0
    use_memory_efficient_attention: bool = False

    def init_weights(self, rng: jax.random.KeyArray) -> FrozenDict:
        # init input tensors
        sample_shape = (1, self.in_channels, self.sample_size, self.sample_size)
        sample = jnp.zeros(sample_shape, dtype=jnp.float32)
        timesteps = jnp.ones((1,), dtype=jnp.int32)
        encoder_hidden_states = jnp.zeros((1, 1, self.cross_attention_dim), dtype=jnp.float32)

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}
        sample = jax.device_put(sample, device=jax.devices("cpu")[0])
        timesteps = jax.device_put(timesteps, device=jax.devices("cpu")[0])
        encoder_hidden_states = jax.device_put(encoder_hidden_states, device=jax.devices("cpu")[0])

        return self.init(rngs, sample, timesteps, encoder_hidden_states,)["params"]

    def setup(self):
        block_out_channels = self.block_out_channels
        time_embed_dim = block_out_channels[0] * 4

        # input
        self.conv_in = nn.Conv(
            block_out_channels[0],
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

        # time
        self.time_proj = FlaxTimesteps(
            block_out_channels[0], flip_sin_to_cos=self.flip_sin_to_cos, freq_shift=self.config.freq_shift
        )
        self.time_embedding = FlaxTimestepEmbedding(time_embed_dim, dtype=self.dtype)

        only_cross_attention = self.only_cross_attention
        if isinstance(only_cross_attention, bool):
            only_cross_attention = (only_cross_attention,) * len(self.down_block_types)

        attention_head_dim = self.attention_head_dim
        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(self.down_block_types)

        # down
        down_blocks = []
        output_channel = block_out_channels[0]
        self.encoder_hid_proj = nn.Dense(time_embed_dim) #fix hardcode
        for i, down_block_type in enumerate(self.down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            if i > 0:
                down_block = FlaxCrossAttnDownBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    dropout=self.dropout,
                    num_layers=self.layers_per_block,
                    attn_num_head_channels=attention_head_dim[i],
                    add_downsample=not is_final_block,
                    use_linear_projection=self.use_linear_projection,
                    only_cross_attention=only_cross_attention[i],
                    use_memory_efficient_attention=self.use_memory_efficient_attention,
                    dtype=self.dtype,
                )
            else:
                down_block = FlaxDownBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    dropout=self.dropout,
                    num_layers=self.layers_per_block,
                    add_downsample=not is_final_block,
                    dtype=self.dtype,
                )

            down_blocks.append(down_block)
        self.down_blocks = down_blocks

        # mid
        self.mid_block = FlaxUNetMidBlock2DCrossAttn(
            in_channels=block_out_channels[-1],
            dropout=self.dropout,
            attn_num_head_channels=attention_head_dim[-1],
            use_linear_projection=self.use_linear_projection,
            use_memory_efficient_attention=self.use_memory_efficient_attention,
            dtype=self.dtype,
        )

        # up
        up_blocks = []
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_attention_head_dim = list(reversed(attention_head_dim))
        only_cross_attention = list(reversed(only_cross_attention))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(self.up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            is_final_block = i == len(block_out_channels) - 1

            if not is_final_block:#up_block_type == "CrossAttnUpBlock2D":
                up_block = FlaxCrossAttnUpBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channel=prev_output_channel,
                    num_layers=self.layers_per_block + 1,
                    attn_num_head_channels=reversed_attention_head_dim[i],
                    add_upsample=not is_final_block,
                    dropout=self.dropout,
                    use_linear_projection=self.use_linear_projection,
                    only_cross_attention=only_cross_attention[i],
                    use_memory_efficient_attention=self.use_memory_efficient_attention,
                    dtype=self.dtype,
                )
            else:
                up_block = FlaxUpBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channel=prev_output_channel,
                    num_layers=self.layers_per_block + 1,
                    add_upsample=not is_final_block,
                    dropout=self.dropout,
                    dtype=self.dtype,
                )

            up_blocks.append(up_block)
            prev_output_channel = output_channel
        self.up_blocks = up_blocks
        self.add_embedding = FlaxTextTimeEmbedding(
            encoder_dim=4096, time_embed_dim=time_embed_dim, num_heads=64
        )#fix hardcode

        # out
        self.conv_norm_out = nn.GroupNorm(num_groups=32, epsilon=1e-5)
        self.conv_out = nn.Conv(
            self.out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

    def __call__(
        self,
        sample,
        timesteps,
        encoder_hidden_states,
        down_block_additional_residuals=None,
        mid_block_additional_residual=None,
        return_dict: bool = True,
        train: bool = False,
        index: int = 0,
    ) -> Union[FlaxUNet2DConditionOutput, Tuple]:
        r"""
        Args:
            sample (`jnp.ndarray`): (batch, channel, height, width) noisy inputs tensor
            timestep (`jnp.ndarray` or `float` or `int`): timesteps
            encoder_hidden_states (`jnp.ndarray`): (batch_size, sequence_length, hidden_size) encoder hidden states
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition_flax.FlaxUNet2DConditionOutput`] instead of a
                plain tuple.
            train (`bool`, *optional*, defaults to `False`):
                Use deterministic functions and disable dropout when not training.

        Returns:
            [`~models.unet_2d_condition_flax.FlaxUNet2DConditionOutput`] or `tuple`:
            [`~models.unet_2d_condition_flax.FlaxUNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`.
            When returning a tuple, the first element is the sample tensor.
        """
        # 1. time
        if not isinstance(timesteps, jnp.ndarray):
            timesteps = jnp.array([timesteps], dtype=jnp.int32)
        elif isinstance(timesteps, jnp.ndarray) and len(timesteps.shape) == 0:
            timesteps = timesteps.astype(dtype=jnp.float32)
            timesteps = jnp.expand_dims(timesteps, 0)

        save_(timesteps,'timesteps_'+str(index)+'.npy')
        t_emb = self.time_proj(timesteps)
        save_(t_emb,'post_time_proj_'+str(index)+'.npy')

        t_emb = self.time_embedding(t_emb)
        save_(t_emb,'t_emb_stage_1_'+str(index)+'.npy')

        save_(sample,'sample_incoming_'+str(index)+'.npy')

        # 2. pre-process
#         print("s0",sample , sample.shape )
        sample = jnp.transpose(sample, (0, 2, 3, 1))
#         print("s1",sample , sample.shape )

        sample = self.conv_in(sample)
        save_(sample,'conv_in_'+str(index)+'.npy')

#         print("s2",sample , sample.shape )
        sample2 = jnp.transpose(sample, (0, 3, 1, 2))
#         print("s3",sample2 , sample2.shape )

#         print("enc huuh",encoder_hidden_states.shape)
        if encoder_hidden_states.shape[-1] == 768:
            t_emb2 = t_emb + self.add_embedding(encoder_hidden_states)
        else:
#             print("pre t_emb",t_emb.shape)
#             print("pre emb val",t_emb)
            save_(t_emb,'t_emb_pre_addition_'+str(index)+'.npy')
            add_ = self.add_embedding(encoder_hidden_states)
            save_(add_,'add_'+str(index)+'.npy')
            print("add ------------------------>", add_.shape)
            print("t_emb ------------------------>", t_emb.shape)
            print("pre ------------------------>", encoder_hidden_states.shape)

            t_emb = t_emb + add_
#             print("post t_emb",t_emb.shape)
        save_(t_emb,'t_emb'+str(index)+'.npy')
            
#         except Exception as e:
#             print("EXCEPTION",e)

    # 3. down
        save_(encoder_hidden_states,'encoder_hidden_states_incoming_'+str(index)+'.npy')
        encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)
        save_(encoder_hidden_states,'encoder_hidden_states_enc_hp_'+str(index)+'.npy')

        down_block_res_samples = (sample,)
#         print('sample.npy',sample)
#         print('t_emb.npy 1',t_emb)
#         print('encoder_hidden_states.npy',encoder_hidden_states)
        for ix , down_block in enumerate(self.down_blocks):
#             print("db",down_block)
            if isinstance(down_block, FlaxCrossAttnDownBlock2D):
                if ix == 1:
                    save = False
                    print("savig -------------------------------------> condition")
                else:
                    save = False
                sample, res_samples = down_block(sample, t_emb, encoder_hidden_states, deterministic=not train,save=save)
#                 print("post down_block",sample.shape)

            else:
#                 try:
#                     print("not cross pre down_block",sample.shape , res_samples.shape)
#                 except:
#                      print("not cross pre down_block",sample.shape)
#                 print("entering init sample", sample )
#                 print(" ")
#                 print("entering init temb ", t_emb )

                sample, res_samples = down_block(sample, t_emb, deterministic=not train)
#                 print("not cross post down_block",sample.shape)
#             if ix == 0:
#                 print("sample 1 --------------->",sample)
            down_block_res_samples += res_samples
        save_(sample,'sample_down'+str(index)+'.npy')
        if down_block_additional_residuals is not None:
            new_down_block_res_samples = ()
#             print("residuals ????")
            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample += down_block_additional_residual
                new_down_block_res_samples += (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # 4. mid
#         print("going into mid block sample v t_emb vs enc hs", sample.shape,t_emb.shape, encoder_hidden_states.shape)
#         print("before mid_block")
#         print('sample.npy',sample)
#         print('t_emb.npy 1',t_emb)
#         print('encoder_hidden_states.npy',encoder_hidden_states)

        import numpy as np
#         np.save('sample.npy',sample)
#         np.save('t_emb.npy',t_emb)
#         np.save('encoder_hidden_states.npy',encoder_hidden_states)
#         print('sample.npy',sample)
#         print('t_emb.npy',t_emb)
#         print('encoder_hidden_states.npy',encoder_hidden_states)
        sample = self.mid_block(sample, t_emb, encoder_hidden_states, deterministic=not train)

        if mid_block_additional_residual is not None:
            sample += mid_block_additional_residual
#         save_(sample,'sample_mid'+str(index)+'_'+str(index)+'.npy')
        # 5. up
        for i , up_block in enumerate(self.up_blocks):
            res_samples = down_block_res_samples[-(self.layers_per_block + 1) :]
            down_block_res_samples = down_block_res_samples[: -(self.layers_per_block + 1)]
            if isinstance(up_block, FlaxCrossAttnUpBlock2D):
                sample = up_block(
                    sample,
                    temb=t_emb,
                    encoder_hidden_states=encoder_hidden_states,
                    res_hidden_states_tuple=res_samples,
                    deterministic=not train,
                )
#                 save_(sample,'up_sample_round_'+str(i)+'_'+str(index)+'.pth')
            else:
                sample = up_block(sample, temb=t_emb, res_hidden_states_tuple=res_samples, deterministic=not train)
#                 save_(sample,'up_sample_round_'+str(i)+'_'+str(index)+'.pth')

#         save_(sample,'sample_up_final'+str(index)+'.npy')
        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = nn.silu(sample)
        sample = self.conv_out(sample)
        sample = jnp.transpose(sample, (0, 3, 1, 2))
#         save_(sample,'final_sample'+str(index)+'.npy')
        if not return_dict:
            return (sample,)

        return FlaxUNet2DConditionOutput(sample=sample)
