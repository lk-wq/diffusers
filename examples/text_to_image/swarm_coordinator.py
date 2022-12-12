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
        rs = restart_from % len(lines)
        self.captions = lines[rs:] + lines[:rs]

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
        caption = self.captions[index]['text']
        
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
    
    parser.add_argument("--file_list", type=str, default="", help="file list")
    parser.add_argument("--coordination_interval", type=float, default=2, help="Hours between coordination rounds")
    parser.add_argument("--local_path", type=str, default="", help="where to save local updates")
    

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

    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")

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
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(args.bucketname)

    def download_remote_directory_to_local(local_path, bucket, files_list, last_update_time):
    #assert os.path.isdir(local_path)
      print("file list -------------------------------------------------->", files_list)
      update_list = []
      for ix , remote_path in enumerate(files_list):
#        remote_path = os.path.join(gcs_path, local_file[1 + len(local_path):])
        print("remote path is --->",remote_path)
        print("local path is ---->", local_path)
        blobs = bucket.list_blobs(prefix=remote_path) 
        for blob in blobs:
            filename = blob.name.replace('/', '_')
            last_modified = blob.updated
            # We only need the unet and also we only want to use the local update if the last modification
            # to the local update happened after the last global update
            print("filenames", last_modified > last_update_time)
            if 'unet' in filename and last_modified > last_update_time:
                import os
                try:
                    dir_ = local_path+str(ix)+'/unet'
                    os.mkdir(local_path+str(ix))
                    os.mkdir(dir_)
                except:
                    pass
                print("file name and dir_",filename,dir_)
                if 'config' in filename:
                    blob.download_to_filename(dir_+'/'+ 'config.json' )  # Download
                else:
                    blob.download_to_filename(dir_+'/'+ 'diffusion_flax_model.msgpack' )  # Download
                update_list.append(local_path+str(ix))                
#             blob.download_to_filename(local_path+str(ix))
        return list(set(update_list))
    
    import glob
    from google.cloud import storage
    from datetime import datetime, timezone, timedelta
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

    fl = args.file_list.split(',')
    print("args.file_list --------------------------->",args.file_list)
    print("??????????????????? fl", fl)
    start = datetime.now( timezone.utc )-timedelta(days=4)
    interval = timedelta(hours=args.coordination_interval)
    while True:
      print("beginning")
      if datetime.now( timezone.utc ) - start > interval:
        import random
        update_list = download_remote_directory_to_local( args.local_path, bucket, fl, start )
        start = datetime.now(timezone.utc)
        print("update_list - - - - - - - - - - - - - - - - - - - - - - - - - -------------------------> ",update_list)

        count = 0
        for ix , i in enumerate(update_list):
#           try: 

          unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
              i, subfolder="unet",  revision='bf16',dtype=weight_dtype
          )
          
#           except:
#               print("load fail ------------------------------>",i) 
#               continue
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
        if count > 0:
          unet_params = unflatten(unet_param_dict)
          step = 1-(count/(count+1))
          unet_params_avg = optax.incremental_update(unet_params, unet_params_avg, step_size=step)
        else:
          unet_params_avg = unflatten(unet_param_dict)
        count += 1
#     text_encoder_params = jax_utils.replicate(text_encoder.params)
#     vae_params = jax_utils.replicate(vae_params)

        # Train!
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
                    "text_encoder": text_encoder.params,
                    "vae": vae_params,
                    "unet": unet_params_avg,
                    "safety_checker": safety_checker.params,
                },
          )
          print("completed")
          break


if __name__ == "__main__":
    main()
