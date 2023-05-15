import argparse
import logging
import math
import os
import random
from pathlib import Path
from typing import Optional
from jax.experimental import mesh_utils
from diffusers import DiffusionPipeline

import numpy as np
import torch
import torch.utils.checkpoint
import flax

import jax
import jax.numpy as jnp
import optax
import transformers
from datasets import load_dataset
from diffusers import (
    FlaxAutoencoderKL,
    FlaxDDPMScheduler,
    FlaxPNDMScheduler,
    FlaxStableDiffusionPipeline,
    FlaxUNet2DConditionModel,
    FlaxDDIMScheduler,
)
from diffusers.pipelines.stable_diffusion import FlaxStableDiffusionSafetyChecker
from flax import jax_utils
from flax.training import train_state
from flax.training.common_utils import shard
from huggingface_hub import HfFolder, Repository, whoami
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTokenizer, FlaxCLIPTextModel, set_seed

from flax.core.frozen_dict import freeze, unfreeze
from flax.training.common_utils import onehot, stack_forest
from partitions_simple import set_partitions

from typing import Dict
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from PIL import Image
from torchvision import transforms
from einops import rearrange
# from ldm.util import instantiate_from_config
from datasets import load_dataset
resolution = 512
import functools
import inspect
from typing import Callable, Dict, Union, NamedTuple, Optional, Iterable, Sequence

# from partitions_text import set_partitions_text
from transformers import T5EncoderModel, FlaxT5EncoderModel

import chex
from jax_smi import initialise_tracking

"""
Stochastically rounded operations between JAX tensors.
You can find an up-to-date source and full description here: https://github.com/nestordemeure/jochastic
"""
import jax
import jax.numpy as jnp

#----------------------------------------------------------------------------------------
# BUILDING BLOCKS
from jax.experimental.maps import xmap
from jax.experimental.pjit import pjit
from jax.sharding import Mesh

def _compute_error(x, y, result):
    """
    Computes the error introduced during a floating point addition (x+y=result) using the TwoSum error-free transformation.
    In infinite precision (associative maths) this function should return 0.
    WARNING: 
    - the order of the operations *matters*, do not change this operation in a way that would alter the order of operations
    - requires rounding to nearest (the default on modern processors) and assumes that floating points follow the IEEE-754 norm 
      (but, it has been tested with alternative types such as bfloat16)
    """
    # NOTE: computing this quantity via a cast to higher precision would be faster for low precisions
    y2 = result - x
    x2 = result - y2
    error_y = y - y2
    error_x = x - x2
    return error_x + error_y

def _misround_result(result, error):
    """
    Given the result of a floating point operation and the numerical error introduced during that operation
    returns the floating point number on the other side of the interval containing the analytical result of the operation.
    NOTE: the output of this function will be of the type of result, the type of error does not matter.
    """
    # computes the direction in which the misrounded result lies
    finfo = jnp.finfo(result.dtype)
    direction = jnp.where(error > 0, finfo.max, finfo.min)
    # goes one ULP in that direction
    return jnp.nextafter(result, direction)

def _pseudorandom_bool(prngKey, result, alternative_result, error, is_biased=True):
    """
    Takes  the result of a floating point operation, 
    the floating point number on the other side of the interval containing the analytical result of the operation
    and the numerical error introduced during that operation
    returns a randomly generated boolean.
    If is_biased is True, the random number generator is biased according to the relative error of the operation
    else, it will round up 50% of the time and down the other 50%.
    """
    if is_biased:
        # gets a random number in [0;1]
        random_unitary_float = jax.random.uniform(key=prngKey, shape=result.shape, dtype=result.dtype)
        # draws a boolean randomly, biasing the draw as a function of the ratio of the error and one ULP
        ulp = jnp.abs(alternative_result - result)
        abs_error = jnp.abs(error)
        result = random_unitary_float * ulp > abs_error
    else:
        # NOTE: we do not deal with the error==0 case as it is too uncommon to bias the results significantly
        result = jax.random.bernoulli(key=prngKey, shape=result.shape)
    return result

#----------------------------------------------------------------------------------------
# OPERATIONS

def add_mixed_precision(prngKey, x, y, is_biased=True):
    """
    Returns the sum of two tensors x and y pseudorandomly rounded up or down to the nearest representable floating-point number.
    Assumes that one of the inputs is higher precision than the other, return a result in the *lowest* precision of the inputs.
    If is_biased is True, the random number generator is biased according to the relative error of the addition
    else, it will round up 50% of the time and down the other 50%.
    """
    # insures that x is lower precision than y
    dtype_x = x.dtype
    dtype_y = y.dtype
    bits_x = jnp.finfo(dtype_x).bits
    bits_y = jnp.finfo(dtype_y).bits
    if bits_x > bits_y: 
        return add_mixed_precision(prngKey, y, x, is_biased)
    assert(dtype_x != dtype_y)
    # performs the addition
    result_high_precision = x.astype(dtype_y) + y
    result = result_high_precision.astype(dtype_x)
    # computes the numerical error
    result_rounded = result.astype(dtype_y)
    error = result_high_precision - result_rounded
    # picks the result to be returned
    alternative_result = _misround_result(result, error)
    useResult = _pseudorandom_bool(prngKey, result, alternative_result, error, is_biased)
    return jnp.where(useResult, result, alternative_result)

def add(prngKey, x, y, is_biased=True):
    """
    Returns the sum of two tensors x and y pseudorandomly rounded up or down to the nearest representable floating-point number.
    This function will delegate to `add_mixed_precision` if the inputs have different precisions.
    It will then return a result of the *lowest* precision of the inputs.
    If is_biased is True, the random number generator is biased according to the relative error of the addition
    else, it will round up 50% of the time and down the other 50%.
    """
    # use a specialized function if one of the inputs is higher precision than than the other
    if (x.dtype != y.dtype):
        return add_mixed_precision(prngKey, x, y, is_biased)
    # computes both the result and the result that would have been obtained with another rounding
    result = x + y 
    error = _compute_error(x, y, result)
    alternative_result = _misround_result(result, error)
    # picks the values for which we will use the other rounding
    use_result = _pseudorandom_bool(prngKey, result, alternative_result, error, is_biased)
    return jnp.where(use_result, result, alternative_result)

#----------------------------------------------------------------------------------------
# TREE OPERATIONS

def _random_split_like_tree(prngKey, tree):
    """
    Takes a random number generator key and a tree, splits the key into a properly structured tree.
    credit: https://github.com/google/jax/discussions/9508#discussioncomment-2144076
    """
    tree_structure = jax.tree_structure(tree)
    key_leaves = jax.random.split(prngKey, tree_structure.num_leaves)
    return jax.tree_unflatten(tree_structure, key_leaves)

def tree_add(prngKey, tree_x, tree_y, is_biased=True):
    """
    Returns the sum of two pytree tree_x and tree_y pseudorandomly rounded up or down to the nearest representable floating-point number.
    If the inputs have different precisions, it will return a result of the *lowest* precision of the inputs.
    If is_biased is True, the random number generator is biased according to the relative error of the addition
    else, it will round up 50% of the time and down the other 50%.
    """
    # split the key into a tree
    tree_prngKey = _random_split_like_tree(prngKey, tree_x)
    # applies the addition to all pair of leaves
    def add_leaf(prngKey, x, y): return add(prngKey, x, y, is_biased)
    return jax.tree_util.tree_map(add_leaf, tree_prngKey, tree_x, tree_y)

from optax._src import base
from optax._src import numerics
train_transforms = transforms.Compose(
    [
        transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution) if True else transforms.RandomCrop(resolution),
        transforms.RandomHorizontalFlip() if True else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)
from jax.experimental import io_callback
from jax import debug
import numpy as np

def save_(x,name):
    # print(name , "---------------------------------------------------->",x)
#     debug.callback(lambda x: np.save(name,x),x ) #np.save('post_conv1.npy',np.asarray(hidden_states))
    debug.callback(lambda x: print(x),x ) #np.save('post_conv1.npy',np.asarray(hidden_states))

    #     print(x)
    return x

from typing import Any, Callable

from flax import core
from flax import struct
import optax

class TrainState(struct.PyTreeNode):
  """Simple train state for the common case with a single Optax optimizer.

  Synopsis::

      state = TrainState.create(
          apply_fn=model.apply,
          params=variables['params'],
          tx=tx)
      grad_fn = jax.grad(make_loss_fn(state.apply_fn))
      for batch in data:
        grads = grad_fn(state.params, batch)
        state = state.apply_gradients(grads=grads)

  Note that you can easily extend this dataclass by subclassing it for storing
  additional data (e.g. additional variable collections).

  For more exotic usecases (e.g. multiple optimizers) it's probably best to
  fork the class and modify it.

  Args:
    step: Counter starts at 0 and is incremented by every call to
      `.apply_gradients()`.
    apply_fn: Usually set to `model.apply()`. Kept in this dataclass for
      convenience to have a shorter params list for the `train_step()` function
      in your training loop.
    params: The parameters to be updated by `tx` and used by `apply_fn`.
    tx: An Optax gradient transformation.
    opt_state: The state for `tx`.
  """
  step: int
  apply_fn: Callable = struct.field(pytree_node=False)
  params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
  c0: core.FrozenDict[str, Any] = struct.field(pytree_node=True)

  tx: optax.GradientTransformation = struct.field(pytree_node=False)
  opt_state: optax.OptState = struct.field(pytree_node=True)

  def apply_gradients(self, *, grads,**kwargs):
    """Updates `step`, `params`, `opt_state` and `**kwargs` in return value.

    Note that internally this function calls `.tx.update()` followed by a call
    to `optax.apply_updates()` to update `params` and `opt_state`.

    Args:
      grads: Gradients that have the same pytree structure as `.params`.
      **kwargs: Additional dataclass attributes that should be `.replace()`-ed.

    Returns:
      An updated instance of `self` with `step` incremented by one, `params`
      and `opt_state` updated by applying `grads`, and additional attributes
      replaced as specified by `kwargs`.
    """
    updates, new_opt_state = self.tx.update(
        grads, self.opt_state, self.params)
#     new_params = tree_add( rng,self.params, updates,is_biased=True)
    updates = jax.tree_util.tree_map(
          lambda update, c: update - c,
          updates, self.c0)
    
    new_params = optax.apply_updates(self.params, updates)
    ctemp = jax.tree_util.tree_map(
          lambda np, p: np - p,
          new_params, self.params)
    c0 = jax.tree_util.tree_map(
          lambda ct, update: ct - update,
          ctemp, updates)
    return self.replace(
        step=self.step + 1,
        params=new_params,
        c0=c0,
        opt_state=new_opt_state,
        **kwargs,
    )

  @classmethod
  def create(cls, *, apply_fn, params,c0, tx,**kwargs):
    """Creates a new instance with `step=0` and initialized `opt_state`."""
    opt_state = tx.init(params)
    return cls(
        step=0,
        apply_fn=apply_fn,
        params=params,
        c0=c0,
        tx=tx,
        opt_state=opt_state,
        **kwargs,
    )



import random
from google.cloud import storage
class FolderData(Dataset):
    def __init__(self,
        root_dir,
        token_dir,
        caption_file=None,
        image_transforms=[],
        ext="jpg",
        default_caption="",
        postprocess=None,
        return_paths=False,
        negative_prompt="",
        restart_from=0,
        section0=0,
        section1=0,
        if_=None,
        ip=None,
        resolution=768,
        resolution2=1536,
        drop=False,
        resize=False,
        center=False,
        tokenizer_folder="",
        ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = root_dir
        self.default_caption = default_caption
        self.return_paths = return_paths
#         l = []
#         for i in range(9):
#           with open(root_dir+args.img_folder, "r") as f:
#               l = f.readlines()
#               lines = [json.loads(x) for x in lines]
#               l.extend(lines)
            # captions = {x["file_name"]: x["text"].strip("\n") for x in lines}
        # rs = restart_from % len(lines)
#         with open(root_dir+if_, "r") as f:
#           l = f.readlines()
#           lines = [json.loads(x) for x in l]
#         captions = {x["file_name"]: x["text"].strip("\n") for x in lines}

        import glob
        print("stuff--------------------->",root_dir+if_+'/*')
#         self.captions = glob.glob(root_dir+if_+'/*')
        with open(if_, "r") as f:
          l = f.readlines()
          self.captions = [json.loads(x) for x in l]

        import random
#         random.shuffle(l)
#         self.captions = lines  #[rs:] + lines[:rs]

        # Only used if there is no caption file
        # self.paths = []
        # if isinstance(image_transforms, ListConfig):
        #     image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
        image_transforms.extend([transforms.ToTensor(),
                                 transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        image_transforms = transforms.Compose(image_transforms)
        print("resolution ", resolution)
#         resolution = 768
        self.tform0 = transforms.Compose(
            [
        transforms.CenterCrop(resolution),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
            ]
        )
        if resize:
            self.tform1 = transforms.Compose(
                [
            transforms.Resize( resolution2, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
                ]
            )
        elif center:
            self.tform1 = transforms.Compose(
                [ transforms.CenterCrop(resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
                ]
            )

        else:
            self.tform1 = transforms.Compose(
                [
    #         transforms.Resize( resolution2, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
                ]
            )

#         self.tokenizer = CLIPTokenizer.from_pretrained(token_dir, subfolder="tokenizer")
        self.negative_prompt = negative_prompt
        self.instance_prompt = ip
        self.tokenizer = pipe = DiffusionPipeline.from_pretrained(
            tokenizer_folder, 
            text_encoder=None, # pass the previously instantiated 8bit text encoder
            unet=None, 
        ).tokenizer
        prompt_ids = torch.load('prompt_embeds.pth',map_location='cpu')
#         prompt_ids = jnp.asarray(prompt_ids)
        self.data = prompt_ids[0].unsqueeze(0)
        self.drop = drop
    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        data = {}
#         print("fn",self.captions[index])
        filename = self.captions[index]['file_name']
        
        im = Image.open(filename)
        im = self.process_im(im)
        data["image"] = im
        caption = self.instance_prompt + self.captions[index]['text']
        list_ = [i for i in range(100)] 
        choice = random.choice(list_)
        if self.drop:
            if choice <= 15:
              caption = ""
        data["txt"] = self.tokenize_captions(caption)
        data['fixed'] = self.data
        # if self.postprocess is not None:
        #     data = self.postprocess(data)

        return data
    
    def tokenize_captions(self,captions, is_train=True):
#         inputs = self.tokenizer(captions, max_length=77, padding="max_length", truncation=True ,add_special_tokens=True,return_tensors="pt")
        text_inputs = self.tokenizer(
                captions,
                padding="max_length",
                max_length=77,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(captions, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, 77 - 1 : -1])
        attention_mask = text_inputs.attention_mask

        return (text_input_ids, attention_mask)
    
    def process_im(self, im):
        i = random.choice([0,1])
        if False:
            im = im.convert("RGB")
            return self.tform1(im)     
        else:
            im = im.convert("RGB")
            return self.tform1(im)     

logger = logging.getLogger(__name__)
import random
from jax.experimental import io_callback
from jax import debug
import numpy as np

def save_(x,name):
    # print(name , "---------------------------------------------------->",x)
    debug.callback(lambda x: np.save(name,x),x ) #np.save('post_conv1.npy',np.asarray(hidden_states))
    return x

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_folder",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained tokenizer from huggingface.co/models.",
    )

    
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument("--train_text_encoder", action="store_true", help="Whether to train the text encoder")
    parser.add_argument("--stochastic_rounding", action="store_true", help="Whether to use stochastic rounding")

    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default="a beautiful art work by mmdd111",
        help="instance prompt.",
    )

    parser.add_argument(
        "--img_folder",
        type=str,
        default="images",
        help="instance prompt.",
    )


    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument("--bucketname", type=str, default='buck', help="Name of bucket.")
    parser.add_argument("--bucketdir", type=str, default='buck', help="Bucket directory.")

    parser.add_argument("--restart_from", type=int, default=0, help="Steps to restart from")
    parser.add_argument("--save_frequency", type=int, default=5120, help="How frequently to save")
    parser.add_argument("--accumulation_frequency", type=int, default=1, help="How frequently to save")
    parser.add_argument("--ema_frequency", type=int, default=10, help="How frequently to perform ema")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt for unconditional generation")

    parser.add_argument("--scheduling", type=str, default="constant", help="scheduling")
    parser.add_argument("--warmup_steps", type=int, default=0, help="warm up steps")
    parser.add_argument("--section0", type=int, default=0, help="section 0")
    parser.add_argument("--section1", type=int, default=0, help="section 1")
    
    parser.add_argument(
        "--resolution",
        type=int,
        default=768,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help="Whether to center crop images before resizing to resolution (if not set, random crop will be used)",
    )
    parser.add_argument(
        "--drop",
        action="store_true",
        help="Whether to prompt drop",
    )
    parser.add_argument(
        "--resize",
        action="store_true",
        help="Whether to prompt drop",
    )


    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--init_learning_rate",
        type=float,
        default=1e-6,
        help="Initial learning rate before warmup.",
    )
    parser.add_argument(
        "--min_decay",
        type=float,
        default=.9999,
        help="minimum decay",
    )

    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--resolution2",
        type=int,
        default=1536,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )

    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    return args


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


dataset_name_mapping = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}

def get_params_to_save(params):
    return jax.device_get(jax.tree_util.tree_map(lambda x: x[0], params))

def get_params_to_avg(params):
    return jax.tree_util.tree_map(lambda x: x[0].astype(jnp.float32), params)

def get_zero(params):
    return jax.tree_util.tree_map(lambda x: x[0] * 0, params)


def main():
#     initialise_tracking()

    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
    if jax.process_index() == 0:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if jax.process_index() == 0:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    dataset = FolderData(args.train_data_dir,args.pretrained_model_name_or_path,negative_prompt=args.negative_prompt,section0=args.section0,section1=args.section1,if_=args.img_folder,ip=args.instance_prompt,resolution=args.resolution,resolution2=args.resolution2,drop=args.drop,resize=args.resize,center=args.center_crop,tokenizer_folder=args.tokenizer_folder)

    def tokenize_captions(captions, is_train=True):
#         captions = [].
        inputs = tokenizer(captions, max_length=tokenizer.model_max_length, padding="do_not_pad", truncation=True)
        input_ids = inputs.input_ids
        return input_ids

    if jax.process_index() == 0:
        if args.max_train_samples is not None:
            pass#dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset = dataset#["train"]#.with_transform(preprocess_train)
    import random
    
    def collate_fn(examples):
        pixel_values = torch.stack([example["image"] for example in examples] )#+ [example['class_images'] for example in examples] )
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.cat([example["txt"][0] for example in examples])
        attention_mask = torch.cat([example["txt"][1] for example in examples])
        fixed = [example["fixed"] for example in examples]
        fixed = torch.cat(fixed)

        batch = {
            "input_ids": input_ids,
            "attention_mask":attention_mask,
            "pixel_values": pixel_values,
            'fixed':fixed,
        }
        batch = {k: v.numpy() for k, v in batch.items()}
        return batch

    total_train_batch_size = args.train_batch_size * 1#jax.local_device_count()
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=total_train_batch_size, drop_last=True
    )

    weight_dtype = jnp.bfloat16
    if args.mixed_precision == "fp16":
        weight_dtype = jnp.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = jnp.bfloat16
    #shard(
    # Load models and create wrapper for stable diffusion
    print("weight type ----->",weight_dtype)
#     text_encoder = FlaxCLIPTextModel.from_pretrained(
#         args.pretrained_model_name_or_path, subfolder="text_encoder",revision='bf16', dtype=weight_dtype
#     )
    
    import flatdict
    def unflatten(dictionary):
        resultDict = dict()
        for key, value in dictionary.items():
            parts = key.split(".")
            d = resultDict
            for part in parts[:-1]:
                if part not in d:
                    d[part] = dict()
                d = d[part]
            d[parts[-1]] = value
        return resultDict

    if args.scale_lr:
        args.learning_rate = args.learning_rate * total_train_batch_size
    if args.scheduling != "constant":
        def warmup_linear_schedule(
            init_value: float,
            peak_value: float,
            warmup_steps: int,
        ) -> base.Schedule:
          schedules = [
              optax.linear_schedule(
                  init_value=init_value,
                  end_value=peak_value,
                  transition_steps=warmup_steps),
              optax.constant_schedule(
                  peak_value)]
          return optax.join_schedules(schedules, [warmup_steps])
        scheduler = warmup_linear_schedule(init_value=args.init_learning_rate, peak_value=args.learning_rate,warmup_steps=args.warmup_steps)

    else:
        scheduler = optax.constant_schedule(args.learning_rate)
        scheduler_text = optax.constant_schedule(args.learning_rate)

    adamw = optax.adamw(
        learning_rate=scheduler,
        b1=args.adam_beta1,
        b2=args.adam_beta2,
        eps=args.adam_epsilon,
        weight_decay=args.adam_weight_decay,
    )

    adamw2 = optax.adamw(
        learning_rate=scheduler_text,
        b1=args.adam_beta1,
        b2=args.adam_beta2,
        eps=args.adam_epsilon,
        weight_decay=args.adam_weight_decay,
    )

    optimizer_ = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        adamw,
    )
    optimizer_2 = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        adamw2,
    )

#     optimizer = optax.MultiSteps(
#         optimizer_, args.accumulation_frequency
#     )
    optimizer2_ = optax.MultiSteps(
        optimizer_2, args.accumulation_frequency
    )
#     optimizer3_ = optax.MultiSteps(
#         optimizer_2, args.accumulation_frequency*2
#     )

    def flattened_traversal(fn):
      """Returns function that is called with `(path, param)` instead of pytree."""
      def mask(tree):
        flat = flax.traverse_util.flatten_dict(tree)
        return flax.traverse_util.unflatten_dict(
            {k: fn(k, v) for k, v in flat.items()})
      return mask
    label_fn = flattened_traversal(
        lambda path, _: 'adam' if check_str(path) else 'none')
    def check_str(path):
      for s in path:
        if '23' in s:
            print("success ---> " , path )
            return True
#       print("fail ----> ", path )      
      return False
    def create_key(seed=0):
        return jax.random.PRNGKey(seed)
    rng = create_key(args.seed)
    weight_dtype = jnp.bfloat16
    unet, params = FlaxUNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet",dtype=weight_dtype
    )
    
    text_encoder, text_params = FlaxT5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder",dtype=weight_dtype
    )
    
#     text_params = jax.tree_util.tree_map(lambda x: jax.device_put(x ,NamedSharding(mesh , partition_shape(x.shape)) ), text_params)
    flat = flax.traverse_util.flatten_dict( text_params )
    
    d = {}
    for i in flat.keys():
        key = ".".join(list(i))
        if '23' in key:
            d[key] = 'adam'
        else:
            d[key] = 'none'
    fd = flax.core.frozen_dict.freeze({"params":d})
    optimizer = optax.multi_transform(
      {'adam': optimizer2_, 'none': optax.set_to_zero()}, label_fn )

    params = jax.tree_util.tree_map(lambda x: np.asarray(x), params)
    text_params = jax.tree_util.tree_map(lambda x: np.asarray(x), text_params)
    
#     mesh_devices = np.array(jax.devices()).reshape(1, jax.local_device_count())
    mesh_devices = mesh_utils.create_device_mesh((4, 2))
        
    print("starting -----------------------------------------------------------> ",mesh_devices)
    print("starting -----------------------------------------------------------> ")
    print("starting -----------------------------------------------------------> ")
    print("starting -----------------------------------------------------------> ")
    print("starting -----------------------------------------------------------> ")
    print("starting -----------------------------------------------------------> ")
    print("starting -----------------------------------------------------------> ")
    print("starting -----------------------------------------------------------> ")
    print("starting -----------------------------------------------------------> ")
    print("starting -----------------------------------------------------------> ")
    import gc 
    gc.collect()

    def partition_shape(shape):
      for i in shape:
        if 6 in shape:
            if len(shape) == 1:
                return P(None)
            if len(shape) == 4:
                return P(None,None,None,None)
      if len(shape) == 1:
        if shape[0] % 4 == 0:
          return P("dp")
        else:
          return P("mp")
      if len(shape) == 2:
        if shape[0] % 4 == 0 and shape[1] % 2 == 0 and shape[0] > shape[1]:
          return P("dp","mp")
        if shape[0] % 2 == 0 and shape[1] % 4 == 0:
          return P("mp","dp")
        if shape[0] % 4 == 0:# and shape[1] % 2 == 0:
          return P("dp",None)
        if shape[1] % 4 == 0:# and shape[1] % 2 == 0:
          return P(None,"dp")
        if shape[0] % 2 == 0 and shape[1] % 2 == 0:
          return P("mp",None)
      if len(shape) == 4:
        if shape[-2] % 4 == 0 and shape[-1] % 2 == 0:
          return P(None,None,"dp","mp")
        if shape[-2] % 2 == 0 and shape[-1] % 4 == 0:
          return P(None,None,"mp","dp")
        if shape[-2] % 4 == 0:# and shape[1] % 2 == 0:
          return P(None,None,"dp",None)
        if shape[-1] % 4 == 0:# and shape[1] % 2 == 0:
          return P(None,None,None,"dp")
        if shape[-1] % 2 == 0 and shape[-2] % 2 == 0:
          return P(None,None,"mp",None)
        
      print("fail")
      return P()
    from jax.experimental import PartitionSpec as P 
    from jax.sharding import NamedSharding

    mesh = Mesh(mesh_devices , axis_names=('dp','mp'))
#     text_opt_state = optimizer.init(text_params)
#     text_opt_state = jax.tree_util.tree_map(lambda x: jax.device_put(x ,NamedSharding(mesh , partition_shape(x.shape)) ), text_opt_state)
#     text_opt_state = optimizer.init(text_params)
#     text_opt_state = text_opt_state#.inner_states 

#     text_opt_state_spec = jax.tree_util.tree_map(lambda x : partition_shape(x.shape), text_opt_state )

    text_param_spec = jax.tree_util.tree_map(lambda x: partition_shape(x.shape) , text_params)
    param_spec = jax.tree_util.tree_map(lambda x: partition_shape(x.shape) , params )
#     flat = flax.traverse_util.flatten_dict( text_params )
#     fk = flat.keys()
#     k = random.choice(list(fk))

#     save_(flat[k],'text_param_keys2.npy')

    text_params = jax.tree_util.tree_map(lambda x: jax.device_put(x ,NamedSharding(mesh , partition_shape(x.shape)) ), text_params)
    unet_params = jax.tree_util.tree_map(lambda x: jax.device_put(x ,NamedSharding(mesh , partition_shape(x.shape)) ), params)
#     text_opt_state = jax.tree_util.tree_map(lambda x: jax.device_put(x ,NamedSharding(mesh , partition_shape(x.shape)) ), text_opt_state)
#     flat = flax.traverse_util.flatten_dict( text_params )
#     fk = flat.keys()

#     save_(flat[k],'text_param_keys2.npy')
    noise_scheduler = FlaxDDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    noise_scheduler_state = noise_scheduler[1]#.create_state()

    # Initialize our training
    rng = jax.random.PRNGKey(args.seed)
    rng, dropout_rng = jax.random.split(rng)

    train_rngs = jax.random.PRNGKey(args.seed)
#     train_rngs = jax.random.split(rng, jax.local_device_count())
    import random
    def train_step(unet_params,text_params, input_ids, pixels, mask, train_rng):
        dropout_rng, sample_rng, new_train_rng = jax.random.split(train_rng, 3)
        params = {"text_encoder": text_params, "unet": unet_params}

        def compute_loss(params):
            # Convert images to latent space
#             latents = vae_outputs.latent_dist.sample(sample_rng)
            # (NHWC) -> (NCHW)
            latents = pixels#batch["pixel_values"]
#             latents = jnp.transpose(latents, (0, 3, 1, 2))
#             latents = latents * 0.18215

            # Sample noise that we'll add to the latents
            noise_rng, timestep_rng = jax.random.split(sample_rng)
            noise = jax.random.normal(noise_rng, latents.shape)
            # Sample a random timestep for each image
            bsz = latents.shape[0]
            timesteps = jax.random.randint(
                timestep_rng,
                (bsz,),
                0,
                noise_scheduler[0].config.num_train_timesteps,
            )
            encoder_hidden_states = text_encoder(
                input_ids,
                attention_mask=mask,
                params=params['text_encoder'],
                train=True,
                dropout_rng=dropout_rng,
            )[0]

            noisy_latents = noise_scheduler[0].add_noise(noise_scheduler_state, latents, noise, timesteps)

#             encoder_hidden_states 
#             save_(params['unet']['time_embedding']['linear_1']['kernel'],'k5.npy')
            
            unet_outputs = unet.apply({"params": unet_params},noisy_latents, timesteps, encoder_hidden_states, train=False)

            noise_pred = unet_outputs.sample 
            noise_pred , variance = noise_pred.split(2, axis=1)

            loss = (noise - noise_pred) ** 2
            loss = loss.mean()

            return loss

#         grad_fn = jax.value_and_grad(compute_loss)
        loss = compute_loss(params)
#         loss, grads = grad_fn(params)
#         unet_updates, new_unet_opt_state = optimizer.update(grads['unet'], unet_opt_state, params['unet'])
#         new_unet_params = optax.apply_updates(params['unet'], unet_updates)
        
#         text_updates, new_text_opt_state = optimizer.update(grads['text_encoder'], text_opt_state,params['text_encoder'])
#         save_(text_updates , 'text_updates')
#         new_text_params = optax.apply_updates(params['text_encoder'], text_updates)
        
        metrics = {"loss": loss}

        return unet_params, text_params, metrics, new_train_rng 
    def compute_loss(params,batch,rngs):
        # Convert images to latent space
#             latents = vae_outputs.latent_dist.sample(sample_rng)
        # (NHWC) -> (NCHW)
        latents = batch["pixel_values"]
#             latents = jnp.transpose(latents, (0, 3, 1, 2))
#             latents = latents * 0.18215

        # Sample noise that we'll add to the latents
        noise_rng, timestep_rng = jax.random.split(rngs)
        noise = jax.random.normal(noise_rng, latents.shape)
        # Sample a random timestep for each image
        bsz = latents.shape[0]
        timesteps = jax.random.randint(
            timestep_rng,
            (bsz,),
            0,
            noise_scheduler[0].config.num_train_timesteps,
        )
        encoder_hidden_states = text_encoder(
            batch["input_ids"],
            attention_mask=batch['attention_mask'],
            params=params,
            train=True,
            dropout_rng=rngs,
        )[0]

        noisy_latents = noise_scheduler[0].add_noise(noise_scheduler_state, latents, noise, timesteps)

        save_(unet_params['time_embedding']['linear_1']['kernel'],'k3_post.npy')
#         print("unet params post ---->" ,unet_params['time_embedding']['linear_1']['kernel'])
        unet_outputs = unet.apply({"params": unet_params},noisy_latents, timesteps, encoder_hidden_states, train=False)

        noise_pred = unet_outputs.sample 
        noise_pred , variance = noise_pred.split(2, axis=1)

        loss = (noise - noise_pred) ** 2
        loss = loss.mean()

        return loss
    # Create parallel version of the train step

    p_train_step = pjit(
        train_step,
        in_axis_resources=( param_spec,text_param_spec,P("dp",None),P("dp",None),P("dp",None),None ),
        out_axis_resources=( param_spec,text_param_spec, None, None),
        donate_argnums=(0, 1),
    )

    # Train!
    num_update_steps_per_epoch = math.ceil(len(train_dataloader))

    import glob
    from google.cloud import storage

    def upload_local_directory_to_gcs(local_path, bucket, gcs_path):
        #assert os.path.isdir(local_path)
        for local_file in glob.glob(local_path + '/**'):
            if not os.path.isfile(local_file):
               upload_local_directory_to_gcs(local_file, bucket, gcs_path + "/" + os.path.basename(local_file))
            else:
               remote_path = os.path.join(gcs_path, local_file[1 + len(local_path):])
               blob = bucket.blob(remote_path)
               blob.upload_from_filename(local_file)
               del blob

    # Scheduler and math around the number of training steps.
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel & distributed) = {total_train_batch_size}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = args.restart_from
#     @jax.jit
    def ema_update(rng, new_tensors, old_tensors, decay):
      # return (avg_params*(epoch_index+1)+params)/(epoch_index+2)  #
      step_size = (1 - decay**(args.ema_frequency))

      return jax.tree_util.tree_map(
          lambda new, old: step_size * new + (1.0 - step_size) * old,
          new_tensors, old_tensors)
    import time
    epochs = tqdm(range(args.num_train_epochs), desc="Epoch ... ", position=0)
#     avg = get_params_to_save(state.params)
#     text_avg = get_params_to_save(text_encoder_state.params)
#     avg = unet_params#jax.device_get(unet_params) #ema_update( rng, jax.device_get(unet_params) , avg, decay )
#     text_avg = text_params#jax.device_get(text_params) #ema_update(rng, jax.device_get(text_params) , text_avg, decay )
#     avg = jax.tree_util.tree_map(lambda x: jnp.copy(x), unet_params)
#     text_avg = jax.tree_util.tree_map(lambda x: jnp.copy(x), text_params)

    client = storage.Client()
    bucket = client.bucket(args.bucketname)
    
#     for ix , epoch in enumerate(epochs):k
#         # ======================== Training ================================
#     loss_jit = jax.jit(compute_loss)
#     grad_fun = jax.jit(jax.value_and_grad(loss_jit))
#     return
    with Mesh(mesh_devices, ("dp","mp")):
        for ix , epoch in enumerate(epochs):
            # ======================== Training ================================

            train_metrics = []

            steps_per_epoch = len(train_dataset) // total_train_batch_size
            train_step_progress_bar = tqdm(total=steps_per_epoch, desc="Training...", position=1, leave=False)

            for batch in train_dataloader:
    #                 batch = shard(batch)
                # batch = shard(batch)
    #             save_(unet_params['time_embedding']['linear_1']['kernel'],'k3_pre.npy')
    #             print("unet params pre ---->" ,unet_params['time_embedding']['linear_1']['kernel'])

#                 loss = loss_jit(text_params , batch , train_rngs )
#                 text_updates, text_opt_state = optimizer.update(grads, text_opt_state,text_params)

#                 text_params = optax.apply_updates(text_params, text_updates)
                bi = batch['input_ids']
                pixels = batch['pixel_values']
                mask = batch['attention_mask']
                unet_params,text_params, train_metric, train_rngs = p_train_step(unet_params,text_params, bi, pixels,mask,train_rngs)

    #             state, train_metric, train_rngs = p_train_step(state, text_encoder_params, vae_params, batch, train_rngs)
                # start = time.perf_counter()
#                 train_metrics.append({'loss':loss})

                train_step_progress_bar.update(1)

                if global_step % args.accumulation_frequency == 0 and global_step > args.restart_from and jax.process_index() == 0:
    #                     if args.ema_frequency > -1 and global_step % args.ema_frequency == 0:
    #                       it = global_step#//args.ema_frequency
    #                       decay = args.min_decay
    #                       decay = min(decay,(1 + it) / (10 + it))
    #                       rng, _ = jax.random.split(rng, 2)
    # #                       params = jax.device_get(unet_params)
    # # jax.device_get(unet_params)

    #                       avg = ema_update( rng, unet_params , avg, decay )
    #                       text_avg = ema_update(rng, text_params , text_avg, decay )

        #             if global_step % 512 == 0 and jax.process_index() == 0 and global_step > 0:
                    if global_step % args.save_frequency == 0:
                        print("saving -----------------------------------------------> " , global_step)
    #                         scheduler = FlaxDDIMScheduler(
    #                             beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", 
    #                             # clip_sample=False,
    #                             num_train_timesteps=1000,
    #                             prediction_type="v_prediction",
    #                             set_alpha_to_one=False,
    #                             steps_offset=1,
    #                             # skip_prk_steps=True,
    #                         )
                #         scheduler = FlaxPNDMScheduler(
                #             beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", skip_prk_steps=True
                #         )
                        scheduler_args = {}

                        scheduler_args["variance_type"] = 'fixed_small'

                        scheduler = FlaxDDIMScheduler.from_config(
                                noise_scheduler[0].config, **scheduler_args
                        )
    #                         params = jax.device_get(unet_params)

                        unet.save_pretrained(
                           args.output_dir+'/unet',params=jax.device_get(avg)
                        )
                        scheduler.save_pretrained(args.output_dir+'/scheduler')
    #                         del params
    #                         params2 = jax.device_get(text_params)

                        text_encoder.save_pretrained(
                            args.output_dir+'/text_encoder',params=jax.device_get(text_avg)
                        )
    #                         del params2

                        if args.ema_frequency > -1:
                            pass
    #                             pipeline.save_pretrained(
    #                                 args.output_dir,
    #                                 params={
    # #                                     "text_encoder": text_avg,
    # #                                     "vae": get_params_to_save(vae_params),
    #                                     "unet": unet_params,
    # #                                     "safety_checker": safety_checker.params,

    #                                 },
    #                             )
                        else:
                            pass
    #                             pipeline.save_pretrained(
    #                                 args.output_dir,
    #                                 params={
    # #                                     "text_encoder": get_params_to_save(text_encoder_state.params),l
    # #                                     "vae": get_params_to_save(vae_params),
    #                                       "unet": unet_params,
    # #                                     "safety_checker": safety_checker.params,

    #                                 },
    #                             )

        #                     blob = bucket.blob(args.output_dir+str(global_step))
                        try:
                            upload_local_directory_to_gcs(args.output_dir, bucket, args.bucketdir+str(global_step))
                            print("upload SUCCESS ===============================================>")
                        except:
                            print("upload fail =================>")

        #                     blob.upload_from_filename(args.output_dir+str(global_step))
        #                     del blob
        #                     jax.lib.xla_bridge.get_backend().defragment()


                global_step += 1
                if global_step >= args.max_train_steps:
                    break


    #             train_metric = jax_utils.unreplicate(train_metric)

            train_step_progress_bar.close()
            epochs.write(f"Epoch... ({epoch + 1}/{args.num_train_epochs} | Loss: {train_metric['loss']})")

        # Create the pipeline using using the trained modules and save it.
        if jax.process_index() == 0:
    #             scheduler = FlaxDDIMScheduler(
    #                 beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", 
    #                 # clip_sample=False,
    #                 num_train_timesteps=1000,
    #                 prediction_type="v_prediction",
    #                 set_alpha_to_one=False,
    #                 steps_offset=1,
    #                 # skip_prk_steps=True,
    #             )
            scheduler_args = {}

            scheduler_args["variance_type"] = 'fixed_small'

            scheduler = FlaxDDIMScheduler.from_config(
                    noise_scheduler[0].config, **scheduler_args
            )

    #             params = jax.device_get(unet_params)

            unet.save_pretrained(
                args.output_dir+'/unet',params=jax.device_get(avg)

            )
    #             del params
    #             params2 = jax.device_get(text_params)

            scheduler.save_pretrained(args.output_dir+'/scheduler')
            text_encoder.save_pretrained(
                args.output_dir+'/text_encoder',params=jax.device_get(text_avg)
            )
    #             del params2
    #         scheduler = FlaxPNDMScheduler(
    #             beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", skip_prk_steps=True
    #         )

            if args.ema_frequency > -1:
                pass
    #                 pipeline.save_pretrained(
    #                     args.output_dir,
    #                     params={
    # #                         "text_encoder": text_avg,
    #                         "unet": unet_params,

    #                     },
    #                 )
            else:
                pass
    #                 unet_params = jax.device_get(unet_params)

    #                 pipeline.save_pretrained(
    #                     args.output_dir,
    #                     params={
    # #                         "text_encoder": get_params_to_save(text_encoder_state.params),
    # #                         "vae": get_params_to_save(vae_params),
    #                           "unet": unet_params,
    # #                         "safety_checker": safety_checker.params,

    #                     },
    #                 )
            upload_local_directory_to_gcs(args.output_dir , bucket, args.bucketdir)


if __name__ == "__main__":
    main()
