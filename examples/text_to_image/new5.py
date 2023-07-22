import argparse
import logging
import math
import os
import random
from pathlib import Path
from jax.experimental import mesh_utils
from jax.sharding import NamedSharding
from jax.experimental.pjit import pjit
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
import torch.utils.checkpoint
import transformers
from datasets import load_dataset
from flax import jax_utils
from flax.training import train_state
from flax.training.common_utils import shard
from huggingface_hub import create_repo, upload_folder
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPTokenizer, FlaxCLIPTextModel, set_seed

from partitions import partition_shape
from jax.sharding import Mesh

from diffusers import (
    FlaxAutoencoderKL,
    FlaxDDPMScheduler,
    FlaxPNDMScheduler,
    FlaxStableDiffusionPipeline,
    FlaxUNet2DConditionModel,
)
from diffusers.pipelines.stable_diffusion import FlaxStableDiffusionSafetyChecker
from diffusers.utils import check_min_version


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

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
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that Datasets can understand."
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
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
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
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
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
        "--model_parallel",
        action="store_true",
        default=False,
        help="Use model parallel training, where layers are sharded across multiple devices",
    )
    parser.add_argument(
        "--from_pt",
        action="store_true",
        default=False,
        help="Convert pytorch models to jax/flax",
    )
    parser.add_argument("--accumulation_frequency", type=int, default=1, help="How frequently to save")


    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    return args


dataset_name_mapping = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}


def get_params_to_save(params):
    return jax.device_get(jax.tree_util.tree_map(lambda x: x[0], params))


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
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files["train"] = os.path.join(args.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    dataset_columns = dataset_name_mapping.get(args.dataset_name, None)
    if args.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: '{', '.join(column_names)}'"
            )

    if args.caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{args.caption_column}' needs to be one of: '{', '.join(column_names)}' "
            )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(captions, max_length=tokenizer.model_max_length, padding="do_not_pad", truncation=True)
        input_ids = inputs.input_ids
        return input_ids

    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)

        return examples

    if args.max_train_samples is not None:
        dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
    train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = [example["input_ids"] for example in examples]

        padded_tokens = tokenizer.pad(
            {"input_ids": input_ids}, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt"
        )
        batch = {
            "pixel_values": pixel_values,
            "input_ids": padded_tokens.input_ids,
        }
        batch = {k: v.numpy() for k, v in batch.items()}

        return batch
    if args.model_parallel:
        total_train_batch_size = args.train_batch_size
    else:
        total_train_batch_size = args.train_batch_size * jax.local_device_count()
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=total_train_batch_size, drop_last=True
    )

    weight_dtype = jnp.float32
    if args.mixed_precision == "fp16":
        weight_dtype = jnp.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = jnp.bfloat16

    # Load models and create wrapper for stable diffusion
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, revision=args.revision, subfolder="tokenizer"
    )
    text_encoder = FlaxCLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, revision=args.revision, subfolder="text_encoder", dtype=weight_dtype, from_pt=args.from_pt
    )
    vae, vae_params = FlaxAutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, revision=args.revision, subfolder="vae", dtype=weight_dtype, from_pt=args.from_pt
    )
    unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, revision=args.revision, subfolder="unet", dtype=weight_dtype, from_pt=args.from_pt
    )
    unet_params = jax.tree_util.tree_map(lambda x: np.asarray(x), unet_params)
    vae_params = jax.tree_util.tree_map(lambda x: np.asarray(x), vae_params)

    # Optimization
    if args.scale_lr:
        args.learning_rate = args.learning_rate * total_train_batch_size

    constant_scheduler = optax.constant_schedule(args.learning_rate)

    adamw = optax.adamw(
        learning_rate=constant_scheduler,
        b1=args.adam_beta1,
        b2=args.adam_beta2,
        eps=args.adam_epsilon,
        weight_decay=args.adam_weight_decay,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        adamw,
    )
    if args.accumulation_frequency > 1:
        optimizer = optax.MultiSteps(
            optimizer, args.accumulation_frequency
        )
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
            if 'atte' not in s:# or 'up_blocks_3' in s or 'norm' in s or 'bias' in s or 'emb' in s.lower() or 'conv_in' in s or 'conv_out' in s:
                return True
            print('fail',path)
          return False
        optimizer = optax.multi_transform(
          {'adam': optimizer, 'none': optax.set_to_zero()}, label_fn )
    import gc 
    
    gc.collect()

    def partition_shape(shape):
      # for i in shape:
      #   if 6 in shape:
      #       if len(shape) == 1:
      #           return P(None)
      #       if len(shape) == 4:
      #           return P(None,None,None,None)
      if len(shape) == 1:
        if shape[0] % 4 == 0:
          return P("dp")
        elif shape[0] % 2 == 0:
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
        
      print("fail",shape)
      return P()
    from jax.sharding import PartitionSpec as P 
    from jax.sharding import NamedSharding
    mesh = Mesh(mesh_devices , axis_names=('dp','mp'))

    if not args.model_parallel:
        unet_params = jax.tree_util.tree_map(lambda x: np.asarray(x).astype(weight_dtype), unet_params)
        vae_params = jax.tree_util.tree_map(lambda x: np.asarray(x).astype(weight_dtype), vae_params)
        text_encoder_params = jax.tree_util.tree_map(lambda x: np.asarray(x).astype(weight_dtype), text_encoder.params)

    if args.model_parallel:
        from jax.sharding import NamedSharding
        unet_params = jax.tree_util.tree_map(lambda x: np.asarray(x), unet_params)
        vae_params = jax.tree_util.tree_map(lambda x: np.asarray(x), vae_params)
        text_params = text_encoder.params
        # del text_encoder.params
        e = jax.tree_util.tree_map(lambda x: None, text_params)

        text_params = jax.tree_util.tree_map(lambda x: np.asarray(x), text_params)
        setattr(text_encoder,'params',text_params)

        # print(text_encoder)
        # mesh_devices = mesh_utils.create_device_mesh((4, 2))

        # mesh = Mesh(mesh_devices , axis_names=('dp','mp'))
        text_param_spec = jax.tree_util.tree_map(lambda x: partition_shape(x.shape) , text_params )
        unet_param_spec = jax.tree_util.tree_map(lambda x: partition_shape(x.shape) , unet_params )
        vae_param_spec = jax.tree_util.tree_map(lambda x: partition_shape(x.shape) , vae_params )
    
        text_params = jax.tree_util.tree_map(lambda x: jax.device_put(x ,NamedSharding(mesh , partition_shape(x.shape)) ).astype(weight_dtype), text_params)
        vae_params = jax.tree_util.tree_map(lambda x: jax.device_put(x ,NamedSharding(mesh , partition_shape(x.shape)) ).astype(weight_dtype), vae_params)
        
        unet_params = jax.tree_util.tree_map(lambda x: jax.device_put(x ,NamedSharding(mesh , partition_shape(x.shape)) ).astype(weight_dtype), unet_params)
        
        # del text_encoder
        opt_state = optimizer.init(unet_params)
        # print('os',opt_state)
        # return
        unet_opt_state_spec = jax.tree_util.tree_map(lambda x : partition_shape(x.shape), opt_state )
    import gc 
    
    gc.collect()

    if not args.model_parallel:
        state = train_state.TrainState.create(apply_fn=unet.__call__, params=unet_params, tx=optimizer)

    noise_scheduler = FlaxDDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
    )
    noise_scheduler_state = noise_scheduler.create_state()

    # Initialize our training
    rng = jax.random.PRNGKey(args.seed)
    if args.model_parallel:
        train_rngs = rng
    else:
        train_rngs = jax.random.split(rng, jax.local_device_count())

    # Create parallel version of the train step
    if args.model_parallel:
        # from jax.sharding import PartitionSpec as P 
        # from jax.sharding import NamedSharding

        def train_step(unet_params, opt_state, text_encoder_params, vae_params, batchi,batchp, train_rng):
                
            dropout_rng, sample_rng, new_train_rng = jax.random.split(train_rng, 3)
            def compute_loss(params):
                # Convert images to latent space
                vae_outputs = vae.apply(
                    {"params": vae_params}, batchp, deterministic=True, method=vae.encode
                )
                latents = vae_outputs.latent_dist.sample(sample_rng)
                # (NHWC) -> (NCHW)
                latents = jnp.transpose(latents, (0, 3, 1, 2))
                latents = latents * vae.config.scaling_factor
    
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
                noisy_latents = noise_scheduler.add_noise(noise_scheduler_state, latents, noise, timesteps)
    
                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(
                    batchi,
                    params=text_encoder_params,
                    train=False,
                )[0]
    
                # Predict the noise residual and compute loss
                model_pred = unet.apply(
                    {"params": params}, noisy_latents, timesteps, encoder_hidden_states, train=True
                ).sample
        
                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(noise_scheduler_state, latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = (target - model_pred) ** 2
                loss = loss.mean()
    
                return loss
    
            grad_fn = jax.value_and_grad(compute_loss)
            loss, grads = grad_fn(unet_params)
            unet_updates, new_unet_opt_state = optimizer.update(grads, opt_state, unet_params)
            new_unet_params = optax.apply_updates(unet_params, unet_updates)
                        
            metrics = {"loss": loss}
    
            return new_unet_params, new_unet_opt_state, metrics, new_train_rng 

        p_train_step = pjit(
            train_step,
            in_axis_resources=( unet_param_spec,unet_opt_state_spec,text_param_spec,vae_param_spec,P("dp",None),P('dp',None,None,'mp'),None ),
            out_axis_resources=( unet_param_spec,unet_opt_state_spec,None, None),
            donate_argnums=(0, 1),
        )
    
    else:

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
                latents = latents * vae.config.scaling_factor
    
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
                noisy_latents = noise_scheduler.add_noise(noise_scheduler_state, latents, noise, timesteps)
    
                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(
                    batch["input_ids"],
                    params=text_encoder_params,
                    train=False,
                )[0]
    
                # Predict the noise residual and compute loss
                model_pred = unet.apply(
                    {"params": params}, noisy_latents, timesteps, encoder_hidden_states, train=True
                ).sample
        
                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(noise_scheduler_state, latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
    
                loss = (target - model_pred) ** 2
                loss = loss.mean()
    
                return loss
    
            grad_fn = jax.value_and_grad(compute_loss)
            loss, grad = grad_fn(state.params)

            grad = jax.lax.pmean(grad, "batch")
    
            new_state = state.apply_gradients(grads=grad)
    
            metrics = {"loss": loss}
            metrics = jax.lax.pmean(metrics, axis_name="batch")

            return new_state, metrics, new_train_rng
    
        p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0,))
    
        # Replicate the train state on each device
        state = jax_utils.replicate(state)
        
        text_encoder_params = jax_utils.replicate(text_encoder_params)
        vae_params = jax_utils.replicate(vae_params)

    # Train!
    num_update_steps_per_epoch = math.ceil(len(train_dataloader))

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

    global_step = 0

    epochs = tqdm(range(args.num_train_epochs), desc="Epoch ... ", position=0)
    if args.model_parallel:
        context = mesh
    else:
        from contextlib import nullcontext
        context = nullcontext()
    with context:

        for epoch in epochs:
            # ======================== Training ================================
    
            train_metrics = []
    
            steps_per_epoch = len(train_dataset) // total_train_batch_size
            train_step_progress_bar = tqdm(total=steps_per_epoch, desc="Training...", position=1, leave=False)
            # train
            for batch in train_dataloader:
                if args.model_parallel:
                    bi = batch['input_ids']
                    bp = batch['pixel_values']
                    unet_params, opt_state, train_metric, train_rngs = p_train_step(unet_params, opt_state, text_params, vae_params, bi,bp, train_rngs)
                else:
                    batch = shard(batch)
                    state, train_metric, train_rngs = p_train_step(state, text_encoder_params, vae_params, batch, train_rngs)
    
                train_metrics.append(train_metric)
    
                train_step_progress_bar.update(1)
    
                global_step += 1
                if global_step >= args.max_train_steps:
                    break
    
            if not args.model_parallel:
                train_metric = jax_utils.unreplicate(train_metric)
    
            train_step_progress_bar.close()
            epochs.write(f"Epoch... ({epoch + 1}/{args.num_train_epochs} | Loss: {train_metric['loss']})")

    # Create the pipeline using using the trained modules and save it.
    if jax.process_index() == 0:
        if args.model_parallel:
            scheduler = FlaxPNDMScheduler(
                beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", skip_prk_steps=True
            )
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
                feature_extractor=CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32"),
            )

            pipeline.save_pretrained(
                args.output_dir,
                params={
                    "text_encoder": jax.device_get(text_params),
                    "vae": jax.device_get(vae_params),
                    "unet": jax.device_get(unet_params),
                    "safety_checker": safety_checker.params,
                },
            )


        else:
            scheduler = FlaxPNDMScheduler(
                beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", skip_prk_steps=True
            )
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
                feature_extractor=CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32"),
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
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )


if __name__ == "__main__":
    main()