import argparse
import logging
import math
import os
import random
from pathlib import Path
from typing import Optional

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

import chex

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
        ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = root_dir
        self.default_caption = default_caption
        self.return_paths = return_paths
        with open(root_dir+'metadata.jsonl', "rt") as f:
            lines = f.readlines()
            lines = [json.loads(x) for x in lines]
            # captions = {x["file_name"]: x["text"].strip("\n") for x in lines}
        self.captions = lines

        # Only used if there is no caption file
        # self.paths = []
        # if isinstance(image_transforms, ListConfig):
        #     image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
        image_transforms.extend([transforms.ToTensor(),
                                 transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        image_transforms = transforms.Compose(image_transforms)
        self.tform = transforms.Compose(
            [
        transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution) if True else transforms.RandomCrop(resolution),
        transforms.RandomHorizontalFlip() if True else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(token_dir, subfolder="tokenizer")
        self.negative_prompt = negative_prompt


    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        data = {}
#         print("fn",self.captions[index])
        filename = self.captions[index]['file_name']

        im = Image.open(self.root_dir+filename)
        im = self.process_im(im)
        data["image"] = im
        caption = self.negative_prompt + self.captions[index]['text']
        
        data["txt"] = self.tokenize_captions(caption)

        # if self.postprocess is not None:
        #     data = self.postprocess(data)

        return data
    
    def tokenize_captions(self,captions, is_train=True):
        inputs = self.tokenizer(captions, max_length=self.tokenizer.model_max_length, padding="do_not_pad", truncation=True)
        input_ids = inputs.input_ids
        return input_ids
    
    def process_im(self, im):
        im = im.convert("RGB")
        return self.tform(im)
    
logger = logging.getLogger(__name__)


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
    

    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
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
    return jax.tree_util.tree_map(lambda x: x[0], params)


def main():
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

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
#     if args.dataset_name is not None:
#         # Downloading and loading a dataset from the hub.
#         dataset = load_dataset(
#             args.dataset_name,
#             args.dataset_config_name,
#             cache_dir=args.cache_dir,
#         )
#     else:
#         data_files = {}
#         if args.train_data_dir is not None:
#             data_files["train"] = os.path.join(args.train_data_dir, "**")
#         dataset = load_dataset(
#             "imagefolder",
#             data_files=data_files,
#             cache_dir=args.cache_dir,
#         )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder
    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
#     column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
#     dataset_columns = dataset_name_mapping.get(args.dataset_name, None)
#     if args.image_column is None:
#         image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
#     else:
#         image_column = args.image_column
#         if image_column not in column_names:
#             raise ValueError(
#                 f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
#             )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
#     def tokenize_captions(examples, is_train=True):
#         captions = []
#         for caption in examples[caption_column]:
#             if isinstance(caption, str):
#                 captions.append(caption)
#             elif isinstance(caption, (list, np.ndarray)):
#                 # take a random caption if there are multiple
#                 captions.append(random.choice(caption) if is_train else caption[0])
#             else:
#                 raise ValueError(
#                     f"Caption column `{caption_column}` should contain either strings or lists of strings."
#                 )
#         inputs = tokenizer(captions, max_length=tokenizer.model_max_length, padding="do_not_pad", truncation=True)
#         input_ids = inputs.input_ids
#         return input_ids
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")

    dataset = FolderData(args.train_data_dir,args.pretrained_model_name_or_path,negative_prompt=args.negative_prompt)

    def tokenize_captions(captions, is_train=True):
#         captions = [].
        inputs = tokenizer(captions, max_length=tokenizer.model_max_length, padding="do_not_pad", truncation=True)
        input_ids = inputs.input_ids
        return input_ids

#     train_transforms = transforms.Compose(
#         [
#             transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
#             transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
#             transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
#             transforms.ToTensor(),
#             transforms.Normalize([0.5], [0.5]),
#         ]
#     )

#     def preprocess_train(examples):
#         images = [image.convert("RGB") for image in examples[image_column]]
#         examples["pixel_values"] = [train_transforms(image) for image in images]
#         examples["input_ids"] = tokenize_captions(examples)

#         return examples

    if jax.process_index() == 0:
        if args.max_train_samples is not None:
            pass#dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset = dataset#["train"]#.with_transform(preprocess_train)
    import random
    def collate_fn(examples):
        pixel_values = torch.stack([example["image"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = [example["txt"] for example in examples]

        padded_tokens = tokenizer.pad(
            {"input_ids": input_ids}, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt"
        )

        input_ids = tokenize_captions([args.negative_prompt for example in range(len(examples)) ])
        ids = [ids for ids in input_ids ]

        padded_tokens2 = tokenizer.pad(
            {"input_ids": ids}, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt"
        )
        unc = padded_tokens2.input_ids
        a = [0,1,2,3,4,5,6,7,8,9]
        random.shuffle(a)
        if a[0] == 0:

          batch = {
              "pixel_values": pixel_values,
              "input_ids": unc,
          }
        else:
            batch = {
              "pixel_values": pixel_values,
              "input_ids": padded_tokens.input_ids,
          }

        batch = {k: v.numpy() for k, v in batch.items()}

        return batch

    total_train_batch_size = args.train_batch_size * jax.local_device_count()
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=total_train_batch_size, drop_last=True
    )

    weight_dtype = jnp.float32
    if args.mixed_precision == "fp16":
        weight_dtype = jnp.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = jnp.bfloat16
    #shard(
    # Load models and create wrapper for stable diffusion
    text_encoder = FlaxCLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision='bf16', dtype=weight_dtype
    )
    
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

    vae, vae_params = FlaxAutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae",revision='bf16',dtype=weight_dtype
    )

    vae_param_dict = dict(flatdict.FlatDict(vae_params, delimiter='.'))
    for r in vae_param_dict.items():
      k , v = r[0], r[1]
      try:
        if v.dtype == jnp.float32:
          v2= v.astype(jnp.bfloat16)
          vae_param_dict[k] = v2
          del v
      except:
        print("f",k)
    vae_params = unflatten(vae_param_dict)
    del vae_param_dict
     
    unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet",  revision='bf16',dtype=weight_dtype
    )
    unet_param_dict = dict(flatdict.FlatDict(unet_params, delimiter='.'))
    for r in unet_param_dict.items():
      k , v = r[0], r[1]
      try:
        if v.dtype == jnp.float32:
          v2= v.astype(jnp.bfloat16)
          unet_param_dict[k] = v2
          del v
      except:
        print("f",k)
    unet_params = unflatten(unet_param_dict)
    del unet_param_dict

    # Optimization
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

    adamw = optax.adamw(
        learning_rate=scheduler,
        b1=args.adam_beta1,
        b2=args.adam_beta2,
        eps=args.adam_epsilon,
        weight_decay=args.adam_weight_decay,
    )
    
#     adamw2 = optax.MultiSteps(
#         adamw, 256
#     )


    optimizer_ = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        adamw,
    )
    optimizer_2 = optax.MultiSteps(
        optimizer_, args.accumulation_frequency
    )
    def flattened_traversal(fn):
      """Returns function that is called with `(path, param)` instead of pytree."""
      def mask(tree):
        flat = flax.traverse_util.flatten_dict(tree)
        return flax.traverse_util.unflatten_dict(
            {k: fn(k, v) for k, v in flat.items()})
      return mask
    label_fn = flattened_traversal(
        lambda path, _: 'adam' if any([check_str(i) for i in path]) else 'none')
    def check_str(s):
      if 'atte' in s:
        return True
      return False
    
    optimizer = optax.multi_transform(
      {'adam': optimizer_2, 'none': optax.set_to_zero()}, label_fn)

    state = train_state.TrainState.create(apply_fn=unet.__call__, params=unet_params, tx=optimizer)

    noise_scheduler = FlaxDDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
    )

    # Initialize our training
    rng = jax.random.PRNGKey(args.seed)
    train_rngs = jax.random.split(rng, jax.local_device_count())
    def train_step(state, text_encoder_params, vae_params, batch, train_rng):
        dropout_rng, sample_rng, new_train_rng = jax.random.split(train_rng, 3)

        def compute_loss(params):
            # Convert images to latent space
            vae_outputs = vae.apply(
                {"params": vae_params}, batch["pixel_values"], deterministic=True, method=vae.encode
            )
            latents = vae_outputs.latent_dist.sample(sample_rng)
            # (NHWC) -> (NCHW)
            latents = jnp.transpose(latents, (0, 3, 1, 2))
            latents = latents * 0.18215

            # Sample noise that we'll add to the latents
            noise_rng, timestep_rng = jax.random.split(sample_rng)
            noise = jax.random.normal(noise_rng, latents.shape)
            # Sample a random timestep for each image
            bsz = latents.shape[0]
            timesteps = jax.random.randint(
                timestep_rng,
                (bsz,),
                0,
                noise_scheduler.config.num_train_timesteps,
            )

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents, alpha, sigma = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            # print("batch", batch["input_ids"])
            encoder_hidden_states = text_encoder(
                batch["input_ids"],
                params=text_encoder_params,
                train=False,
            )[0]
            #unc = tokenizer([""]*len(batch['input_ids']), max_length=tokenizer.model_max_length, padding="do_not_pad", truncation=True)
            # print("unc",unc)

            # encoder_hidden_states_unc = text_encoder(
            #     batch['unc'],
            #     params=text_encoder_params,
            #     train=False,
            # )[0]

            # Predict the noise residual and compute loss
            unet_outputs = unet.apply({"params": params}, noisy_latents, timesteps, encoder_hidden_states, train=True)
            # unet_outputs_unc = unet.apply({"params": params}, noisy_latents, timesteps, encoder_hidden_states_unc, train=True)

            noise_pred = unet_outputs.sample
            # noise_pred_unc = unet_outputs_unc.sample
            # noise_pred_use = noise_pred_unc + 3*( noise_pred - noise_pred_unc )
            v = alpha*noise - sigma * latents
            loss = (v - noise_pred) ** 2
            loss = loss.mean()

            return loss

        grad_fn = jax.value_and_grad(compute_loss)
        loss, grad = grad_fn(state.params)
        grad = jax.lax.pmean(grad, "batch")

        new_state = state.apply_gradients(grads=grad)
        metrics = {"loss": loss}
        metrics = jax.lax.pmean(metrics, axis_name="batch")

        return new_state, metrics, new_train_rng 

    # Create parallel version of the train step
    p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0,))

    # Replicate the train state on each device
    state = jax_utils.replicate(state)
    # avg_state = jax_utils.replicate(avg_state)

    text_encoder_params = jax_utils.replicate(text_encoder.params)
    vae_params = jax_utils.replicate(vae_params)

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
    #@jax.jit
    def ema_update(params, avg_params, it):
      # return (avg_params*(epoch_index+1)+params)/(epoch_index+2)  #
#       step_ = 1/(epoch_index+2)
      decay = 0.9999
      decay = min(decay,(1 + it) / (10 + it))
      step = 1 - decay
      return optax.incremental_update(params, avg_params, step_size=step)
    import time
    epochs = tqdm(range(args.num_train_epochs), desc="Epoch ... ", position=0)
    avg = get_params_to_save(state.params)
    client = storage.Client()
    bucket = client.bucket(args.bucketname)
    
        
    for ix , epoch in enumerate(epochs):
        # ======================== Training ================================

        train_metrics = []

        steps_per_epoch = len(train_dataset) // total_train_batch_size
        train_step_progress_bar = tqdm(total=steps_per_epoch, desc="Training...", position=1, leave=False)
        # train
        for batch in train_dataloader:
            batch = shard(batch)
            # batch = shard(batch)

            state, train_metric, train_rngs = p_train_step(state, text_encoder_params, vae_params, batch, train_rngs)
            # start = time.perf_counter()
            train_metrics.append(train_metric)

            train_step_progress_bar.update(1)

            if global_step % args.accumulation_frequency == 0 and global_step > args.restart_from and jax.process_index() == 0:
                if global_step % args.ema_frequency == 0:
                  avg = ema_update( get_params_to_save(state.params) , avg, global_step//args.accumulation_frequency )

#             if global_step % 512 == 0 and jax.process_index() == 0 and global_step > 0:
                if global_step % args.save_frequency == 0:
                    scheduler = FlaxDDIMScheduler(
                        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", 
                        # clip_sample=False,
                        num_train_timesteps=1000,
                        prediction_type="v_prediction",
                        set_alpha_to_one=False,
                        steps_offset=1,
                        # skip_prk_steps=True,
                    )
            #         scheduler = FlaxPNDMScheduler(
            #             beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", skip_prk_steps=True
            #         )

                    safety_checker = FlaxStableDiffusionSafetyChecker.from_pretrained(
                        "CompVis/stable-diffusion-safety-checker", from_pt=True
                    )
                    pipeline = FlaxStableDiffusionPipeline(
                        text_encoder=text_encoder,
                        vae=vae,
                        unet=unet,
                        tokenizer=tokenizer,
                        scheduler=scheduler,
                        safety_checker=safety_checker,
                        feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
                    )

                    pipeline.save_pretrained(
                        args.output_dir+str(global_step),
                        params={
                            "text_encoder": get_params_to_save(text_encoder_params),
                            "vae": get_params_to_save(vae_params),
                            "unet": avg,
                            "safety_checker": safety_checker.params,
                        },
                    )
#                     blob = bucket.blob(args.output_dir+str(global_step))
                    upload_local_directory_to_gcs(args.output_dir+str(global_step), bucket, args.bucketdir+str(global_step))

#                     blob.upload_from_filename(args.output_dir+str(global_step))
#                     del blob
                    del pipeline
                    del safety_checker
#                     jax.lib.xla_bridge.get_backend().defragment()


            global_step += 1
            if global_step >= args.max_train_steps:
                break


        train_metric = jax_utils.unreplicate(train_metric)

        train_step_progress_bar.close()
        epochs.write(f"Epoch... ({epoch + 1}/{args.num_train_epochs} | Loss: {train_metric['loss']})")

    # Create the pipeline using using the trained modules and save it.
    if jax.process_index() == 0:
        scheduler = FlaxDDIMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", 
            # clip_sample=False,
            num_train_timesteps=1000,
            prediction_type="v_prediction",
            set_alpha_to_one=False,
            steps_offset=1,
            # skip_prk_steps=True,
        )
#         scheduler = FlaxPNDMScheduler(
#             beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", skip_prk_steps=True
#         )

        safety_checker = FlaxStableDiffusionSafetyChecker.from_pretrained(
            "CompVis/stable-diffusion-safety-checker", from_pt=True
        )
        pipeline = FlaxStableDiffusionPipeline(
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            tokenizer=tokenizer,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
        )

        pipeline.save_pretrained(
            args.output_dir,
            params={
                "text_encoder": get_params_to_save(text_encoder_params),
                "vae": get_params_to_save(vae_params),
                "unet": get_params_to_save(state.params),
                "safety_checker": safety_checker.params,
            },
        )

        if args.push_to_hub:
            repo.push_to_hub(commit_message="End of training", blocking=False, auto_lfs_prune=True)


if __name__ == "__main__":
    main()
