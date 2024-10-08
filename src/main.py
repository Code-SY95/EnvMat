import argparse
import datetime
import glob
import os
import sys
import time
from copy import deepcopy
from functools import partial

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from omegaconf import OmegaConf
from packaging import version
from PIL import Image
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import (Callback, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.utilities.distributed import rank_zero_only
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.utils import save_image

# import wandb
from ldm.data.material_utils import *  
from ldm.util import instantiate_from_config

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import torch.multiprocessing as mp

MULTINODE_HACKS = True

conditions = [
    # ["sketch"], # Sy
    # ["palette"],
    ["image_embed"],
    ["text"],
]


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p", "--project", help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    return np.random.seed(np.random.get_state()[1][0] + worker_id)


class RandomSampler(Sampler):
    def __init__(self, data_source, num_samples=None):
        self.data_source = data_source
        self._num_samples = num_samples

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(self.num_samples)
            )

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        return iter(torch.randperm(n, dtype=torch.int64)[: self.num_samples].tolist())

    def __len__(self):
        return self.num_samples


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        train=None,
        validation=None,
        test=None,
        predict=None,
        wrap=False,
        num_workers=None,
        shuffle_test_loader=False,
        use_worker_init_fn=False,
        shuffle_val_dataloader=False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        # self.device = device # Sy:
        
        if train is not None:
            self.dataset_configs["train"] = train
            # Sy: self.train_dataloader is a function which will be executed within trainer.fit(model, data). Where data is a dataModule = self.
            # Sy: In other words, within the fit method data.train_dataloader will be executed and then a DataLoader will be created using the MatFuseDataset object.
            self.train_dataloader = partial( # Sy: Evaluate function self._train_dataloader with no arguments : self it self contains the datasets["train"]. 
                self._train_dataloader, 
            ) # Sy: self._train_dataloader returns DataLoader(
                                                    #     self.datasets["train"],
                                                    #     batch_size=self.batch_size,
                                                    #     num_workers=self.num_workers,
                                                    #     worker_init_fn=init_fn,
                                                    #     sampler=sampler,
                                                    #     persistent_workers=self.num_workers > 0,
                                                    # )
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(
                self._val_dataloader, shuffle=shuffle_val_dataloader
            )
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(
                self._test_dataloader, shuffle=shuffle_test_loader
            )
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self): # Sy: Networkmodule and Datamodule(self) are passed to trainer.fit() 
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg) # Sy: This calls the __init__() of MatFuseDataset class. And it creates MatFuseDataset.materials() which contains the name of types and their folders.

    def setup(self, stage=None): # Sy: setup create dataset. In our case we create the dataset of 'train data'&'validation data'.
        self.datasets = dict( # Sy: self.datasets dict contains the root folder of the dataset.
            (k, instantiate_from_config(self.dataset_configs[k])) # Sy: instantiate_from_config(self.dataset_configs[k]) is refers to MatFuseDataset class object.
            for k in self.dataset_configs
        )
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        
        subsample_size = min(50000, len(self.datasets["train"]))

        sampler = RandomSampler(self.datasets["train"], subsample_size)
        if self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        return DataLoader(
            self.datasets["train"], # Sy: self.datasets["train"] refers to object of MatFuseDataset. The object has the root folder.
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=init_fn,
            sampler=sampler,
            persistent_workers=self.num_workers > 0,
        )

    def _val_dataloader(self, shuffle=False):
        if self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(
            self.datasets["validation"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=init_fn,
            shuffle=shuffle,
        )

    def _test_dataloader(self, shuffle=False):
        if self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        return DataLoader(
            self.datasets["test"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=init_fn,
            shuffle=shuffle,
        )

    def _predict_dataloader(self, shuffle=False):
        if self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(
            self.datasets["predict"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=init_fn,
        )


class SetupCallback(Callback):
    def __init__(
        self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config, debug
    ):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config
        self.debug = debug

    def on_keyboard_interrupt(self, trainer, pl_module):
        if not self.debug and trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if (
                    "metrics_over_trainsteps_checkpoint"
                    in self.lightning_config["callbacks"]
                ):
                    os.makedirs(
                        os.path.join(self.ckptdir, "trainstep_checkpoints"),
                        exist_ok=True,
                    )
            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            if MULTINODE_HACKS:
                import time

                time.sleep(5)
            OmegaConf.save(
                self.config,
                os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)),
            )

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(
                OmegaConf.create({"lightning": self.lightning_config}),
                os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)),
            )

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not MULTINODE_HACKS and not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass


class ImageLogger(Callback):
    def __init__(
        self,
        batch_frequency,
        max_images,
        clamp=True,
        increase_log_steps=True,
        rescale=True,
        disabled=False,
        log_on_batch_idx=True,
        log_first_step=False,
        log_images_kwargs=None,
    ):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            # pl.loggers.TestTubeLogger: self._testtube,
            # pl.loggers.WandbLogger: self._wandb,
        }
        self.log_steps = [2**n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        # Sy
        # wandb.init(project="matfuse", entity="matereal-diffuser")
        # wandb.init(project="matGen", entity="sogang_swatchon")

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid, global_step=pl_module.global_step
            )

    @rank_zero_only
    def _wandb(self, pl_module, images, batch_idx, split, outputs):
        # conditions = images.pop("conditions")
        # wandb.log(
        #     {"conditions": wandb.Image(conditions), "epoch": pl_module.current_epoch}
        # )
        # Sy: images is the dict of autoencoder's log. So it has two keys which are 'inputs', 'reconstructions'. There is no 'conditions' key. 
        wandb.init(project="matGen", entity="sogang_swatchon")
        wandb.log(
            {"AE loss" : outputs[0], "DISC loss" : outputs[1], "epoch": pl_module.current_epoch}
        )

        for k in images:
            if images[k].shape[1] == 12:
                maps = unpack_maps(((images[k] + 1) / 2).clamp(0, 1))
                plot = make_plot_maps(maps)
                image_chunks = torch.chunk(plot, len(plot) // 4, dim=0)
                grids = [
                    torchvision.utils.make_grid(plots, nrow=2, padding=20)
                    for plots in image_chunks
                ]
                wnb_images = [wandb.Image(grid) for grid in grids]
            else:
                image_chunks = torch.chunk(images[k], len(images[k]) // 4, dim=0)
                grids = [
                    torchvision.utils.make_grid(plots, nrow=2, padding=20)
                    for plots in image_chunks
                ]
                # grid = torchvision.utils.make_grid(images[k], nrow=2, padding=20)
                wnb_images = []
                for grid in grids:
                    if self.rescale:
                        grid -= (grid == 0).float()
                        grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                    grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                    grid = grid.numpy()
                    plot = (grid * 255).astype(np.uint8)
                    wnb_images.append(wandb.Image(plot))

            tag = f"images/{k}"
            wandb.log({tag: wnb_images, "epoch": pl_module.current_epoch})

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images: # Sy: images is the dict of autoencoder's log. And k in the keys of log dict.
            # if k in ["diffusion_row", "mask", "progressive_row"]:
            #     continue
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k, global_step, current_epoch, batch_idx
            )
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)

            if images[k].shape[1] == 12:  
                maps = unpack_maps(((images[k] + 1) / 2).clamp(0, 1))
                maps = make_plot_maps(maps)
                save_image(maps, path, nrow=4, padding=20)
            ### Sy: Add.
            elif images[k].shape[1] == 15:  
                maps = unpack_maps_and_env(((images[k] + 1) / 2).clamp(0, 1))
                maps = make_plot_maps_and_env(maps)
                save_image(maps, path, nrow=4, padding=20)
            else:
                grid = torchvision.utils.make_grid(images[k], nrow=4) # Sy: grid에 render_image가 반영되지 않는 이유가 뭘까? env_recon은 되는데
                if self.rescale:
                    grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w    
                grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1) # Sy: grid = [260, 1034, 15]
                grid = grid.numpy()
                grid = (grid * 255).astype(np.uint8)
                Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, outputs, split="train", cond=None): # Sy: I add outputs
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (
            self.check_frequency(check_idx)
            and hasattr(pl_module, "log_images")  # batch_idx % self.batch_freq == 0
            and callable(pl_module.log_images)
            and batch_idx > 5
            and self.max_images > 0
        ):
            logger = type(pl_module.logger) # Sy: <class 'pytorch_lightning.loggers.test_tube.TestTubeLogger'>

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images( 
                    batch, split=split, cond=cond, **self.log_images_kwargs
                ) # Sy: It return (autoencoder.log_images -> log dict.) or (ddpm.log_images)
            
            for k in images: # Sy: k is the keys of images dict.
                N = min(images[k].shape[0], self.max_images) # Sy: N is batch size.
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1.0, 1.0) # Sy: Last k will be 'reconstructions'
            
            self.log_local(
                pl_module.logger.save_dir,
                split,
                images, # Sy: {... "image_embed": non-zero tensors(4,3,256,256) }
                pl_module.global_step,
                pl_module.current_epoch,
                batch_idx,
            )
            logger_log_images = self.logger_log_images.get(
                logger, lambda *args, **kwargs: None
            )

            logger_log_images(pl_module, images, pl_module.global_step, split)
  
            # self._wandb(pl_module, images, pl_module.global_step, split, outputs) # Sy: I add outputs

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
            check_idx > 0 or self.log_first_step
        ):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                print(e)
                pass
            return True
        return False

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx # Sy: outputs = {'loss' : aeloss}, {'loss' : discloss}
    ):
        # if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
        if batch_idx == 10: # Sy:batch_idx에 log를 저장하기 위한 장치
            for cond in conditions: # Sy: conditions = [['image_embed'], ['text']]
                tmp_batch = deepcopy(batch)
                tmp_batch["image_embed"] = batch["image_embed"] # Sy: tmp_batch['image_embed'] = 0.으로만 구성된 tensor라서 batch['image_embed']를 할당.
                self.log_img(pl_module, tmp_batch, batch_idx, outputs, cond=cond, split="train") # Sy: I add outputs

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if not self.disabled and pl_module.global_step > 0:
            for cond in conditions:
                tmp_batch = deepcopy(batch) # Sy: 여기서 batch를 deepcopy하면서 'image_embed'의 value 값이 전부 0.인 텐서로 변경됨. 나머지 key의 value는 동일.
                tmp_batch["image_embed"] = batch["image_embed"] # Sy: tmp_batch['image_embed'] = 0.으로만 구성된 tensor라서 batch['image_embed']를 할당.
                self.log_img(pl_module, tmp_batch, batch_idx, outputs, cond=cond, split="val") # Sy: I add outputs
            # self.log_img(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, "calibrate_grad_norm"):
            if (
                pl_module.calibrate_grad_norm and batch_idx % 25 == 0
            ) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.root_gpu)
        torch.cuda.synchronize(trainer.root_gpu)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.synchronize(trainer.root_gpu)
        max_memory = torch.cuda.max_memory_allocated(trainer.root_gpu) / 2**20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass


class SingleImageLogger(Callback):
    """does not save as grid but as single images"""

    def __init__(
        self,
        batch_frequency,
        max_images,
        clamp=True,
        increase_log_steps=True,
        rescale=True,
        disabled=False,
        log_on_batch_idx=True,
        log_first_step=False,
        log_images_kwargs=None,
        log_always=False,
    ):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.TestTubeLogger: self._testtube,
        }
        self.log_steps = [2**n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.log_always = log_always

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid, global_step=pl_module.global_step
            )

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        os.makedirs(root, exist_ok=True)
        for k in images:
            subroot = os.path.join(root, k)
            os.makedirs(subroot, exist_ok=True)
            base_count = len(glob.glob(os.path.join(subroot, "*.png")))
            for img in images[k]:
                if self.rescale:
                    img = (img + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                img = img.transpose(0, 1).transpose(1, 2).squeeze(-1)
                img = img.numpy()
                img = (img * 255).astype(np.uint8)
                filename = "{}_gs-{:06}_e-{:06}_b-{:06}_{:08}.png".format(
                    k, global_step, current_epoch, batch_idx, base_count
                )
                path = os.path.join(subroot, filename)
                Image.fromarray(img).save(path)
                base_count += 1

    def log_img(self, pl_module, batch, batch_idx, split="train", save_dir=None):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (
            self.check_frequency(check_idx)
            and hasattr(pl_module, "log_images")  # batch_idx % self.batch_freq == 0
            and callable(pl_module.log_images)
            and self.max_images > 0
        ) or self.log_always:
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(
                    batch, split=split, **self.log_images_kwargs
                )

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1.0, 1.0)

            self.log_local(
                pl_module.logger.save_dir if save_dir is None else save_dir,
                split,
                images,
                pl_module.global_step,
                pl_module.current_epoch,
                batch_idx,
            )

            logger_log_images = self.logger_log_images.get(
                logger, lambda *args, **kwargs: None
            )
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
            check_idx > 0 or self.log_first_step
        ):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                print(e)
            return True
        return False
##### Sy: 
# def freeze_except_decoder(model):
#     for name, param in model.named_parameters():
#         if "first_stage_model_pbr.decoder" not in name:
#             param.requires_grad = False
#         else:
#             # print(name)
#             param.requires_grad = True



if __name__ == "__main__":
    # custom parser to specify config files, train, test and debug mode,
    # postfix, resume.
    # `--key value` arguments are interpreted as arguments to the trainer.
    # `nested.key=value` arguments are interpreted as config parameters.
    # configs are merged from left-to-right followed by command line parameters.

    # model:
    #   base_learning_rate: float
    #   target: path to lightning module
    #   params:
    #       key: value
    # data:
    #   target: main.DataModuleFromConfig
    #   params:
    #      batch_size: int
    #      wrap: bool
    #      train:
    #          target: path to train dataset
    #          params:
    #              key: value
    #      validation:
    #          target: path to validation dataset
    #          params:
    #              key: value
    #      test:
    #          target: path to test dataset
    #          params:
    #              key: value
    # lightning: (optional, has sane defaults and can be specified on cmdline)
    #   trainer:
    #       additional arguments to trainer
    #   logger:
    #       logger to instantiate
    #   modelcheckpoint:
    #       modelcheckpoint to instantiate
    #   callbacks:
    #       callback1:
    #           target: importpath
    #           params:
    #               key: value

    mp.set_start_method('spawn', force=True) # Sy: multiprocessing을 위한 수정
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())
    sys.path.append(os.getcwd()+"/src/ldm") # Sy: Append path
    print(sys.path)

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        # opt.ckpt_path = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    # try:
    # init and save configs
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    lightning_config = config.pop("lightning", OmegaConf.create())
    # merge trainer cli with config
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    # default to ddp
    trainer_config["accelerator"] = "ddp" # MK : 
    # trainer_config["strategy"] = "ddp"
    trainer_config["replace_sampler_ddp"] = False
    for k in nondefault_trainer_args(opt):
        trainer_config[k] = getattr(opt, k)
    if not "gpus" in trainer_config:
        del trainer_config["accelerator"]
        cpu = True
    else:
        gpuinfo = trainer_config["gpus"]
        print(f"Running on GPUs {gpuinfo}")
        cpu = False
    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    if opt.gpus:
        device = torch.device("cuda", index=int(opt.gpus.split(",")[0]))
    
    # model
    model = instantiate_from_config(config.model).to(device)
    
    ##### Sy: To fine tuning #####
    # # Initialize model from checkpoint
    # model.init_from_ckpt(opt.resume)

    # # Freeze all parameters except the decoder
    # freeze_except_decoder(model)
    
    # # Ensure there are parameters to update
    # if not any(param.requires_grad for param in model.parameters()):
    #     raise RuntimeError("No parameters require gradients. Check the freezing logic.")

    # # Synchronize EMA model with main model
    # model.model_ema.copy_to(model)

    # # Use DistributedDataParallel if needed
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])

    # # Set the new learning rate for fine-tuning
    # model.learning_rate = 1e-4  # or another appropriate value
    
    # if opt.resume_from_checkpoint:
    #     checkpoint = torch.load(opt.resume_from_checkpoint, map_location=device)
    #     model.load_state_dict(checkpoint['state_dict'])

    # # Freeze all parameters
    # for param in model.parameters():
    #     param.requires_grad = False

    # # Unfreeze only the decoder parameters of the first_stage_model
    # for param in model.first_stage_model_pbr.decoder.parameters():
    #     param.requires_grad = True
        
    # for param in model.first_stage_model_env.decoder.parameters():
    #     param.requires_grad = True
        
    # # Initialize the optimizer only with parameters that require gradients
    # finetune_lr = 0.0001
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=finetune_lr)


    # trainer and callbacks
    trainer_kwargs = dict()

    # default logger configs
    default_logger_cfgs = {
        "wandb": {
            "target": "pytorch_lightning.loggers.WandbLogger",
            "params": {
                "name": nowname,
                "entity": "matereal-diffuser",
                "project": "matfuse",
                "save_dir": logdir,
                # "offline": opt.debug,
                "id": nowname,
            },
        },
        "testtube": {
            "target": "pytorch_lightning.loggers.TestTubeLogger",
            "params": {
                "name": "testtube",
                "save_dir": logdir,
            },
        },
    }
    default_logger_cfg = default_logger_cfgs["testtube"]
    # default_logger_cfg = default_logger_cfgs["wandb"]
    if "logger" in lightning_config:
        logger_cfg = lightning_config.logger
    else:
        logger_cfg = OmegaConf.create()
    logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
    trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

    # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
    # specify which metric is used to determine best models
    default_modelckpt_cfg = {
        "target": "pytorch_lightning.callbacks.ModelCheckpoint",
        "params": {
            "dirpath": ckptdir,
            "filename": "last.ckpt", # Sy: 
            "verbose": True,
            "save_last": True,
        },
    }
    if hasattr(model, "monitor"):
        print(f"Monitoring {model.monitor} as checkpoint metric.")
        default_modelckpt_cfg["params"]["monitor"] = model.monitor
        default_modelckpt_cfg["params"]["save_top_k"] = 3

    if "modelcheckpoint" in lightning_config:
        modelckpt_cfg = lightning_config.modelcheckpoint
    else:
        modelckpt_cfg = OmegaConf.create()
    modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
    print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")
    if version.parse(pl.__version__) < version.parse("1.4.0"):
        trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

    # add callback which sets up log directory
    default_callbacks_cfg = {
        "setup_callback": {
            "target": "main.SetupCallback",
            "params": {
                "resume": opt.resume,
                "now": now,
                "logdir": logdir,
                "ckptdir": ckptdir,
                "cfgdir": cfgdir,
                "config": config,
                "lightning_config": lightning_config,
                "debug": opt.debug,
            },
        },
        "image_logger": {
            "target": "main.ImageLogger",
            "params": {"batch_frequency": 10, "max_images": 4, "clamp": True},
        },
        "learning_rate_logger": {
            "target": "main.LearningRateMonitor",
            "params": {
                "logging_interval": "step",
                # "log_momentum": True
            },
        },
        "cuda_callback": {"target": "main.CUDACallback"},
    }
    if version.parse(pl.__version__) >= version.parse("1.4.0"):
        default_callbacks_cfg.update({"checkpoint_callback": modelckpt_cfg})

    if opt.debug:
        setattr(trainer_opt, "limit_train_batches", 100)
        setattr(trainer_opt, "limit_val_batches", 10)
        lightning_config.callbacks.image_logger.params["batch_frequency"] = 5

    if "callbacks" in lightning_config:
        callbacks_cfg = lightning_config.callbacks
    else:
        callbacks_cfg = OmegaConf.create()

    if "metrics_over_trainsteps_checkpoint" in callbacks_cfg:
        print(
            "Caution: Saving checkpoints every n train steps without deleting. This might require some free space."
        )
        default_metrics_over_trainsteps_ckpt_dict = {
            "metrics_over_trainsteps_checkpoint": {
                "target": "pytorch_lightning.callbacks.ModelCheckpoint",
                "params": {
                    "dirpath": os.path.join(ckptdir, "trainstep_checkpoints"),
                    "filename": "{epoch:06}-{step:09}",
                    "verbose": True,
                    "save_top_k": -1,
                    # "every_n_train_steps": 10000,
                    "every_n_train_steps": 5000, # Sy:
                    "save_weights_only": True,
                },
            }
        }
        default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
    if "ignore_keys_callback" in callbacks_cfg and hasattr(
        trainer_opt, "resume_from_checkpoint"
    ):
        callbacks_cfg.ignore_keys_callback.params["ckpt_path"] = (
            trainer_opt.resume_from_checkpoint
        )
    elif "ignore_keys_callback" in callbacks_cfg:
        del callbacks_cfg["ignore_keys_callback"]

    trainer_kwargs["callbacks"] = [
        instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg
    ]
    if not "plugins" in trainer_kwargs:
        trainer_kwargs["plugins"] = list()
    if not lightning_config.get("find_unused_parameters", True):
        from pytorch_lightning.plugins import DDPPlugin

        trainer_kwargs["plugins"].append(DDPPlugin(find_unused_parameters=False))
    if MULTINODE_HACKS:
        # disable resume from hpc ckpts
        # NOTE below only works in later versions
        # from pytorch_lightning.plugins.environments import SLURMEnvironment
        # trainer_kwargs["plugins"].append(SLURMEnvironment(auto_requeue=False))
        # hence we monkey patch things
        from pytorch_lightning.trainer.connectors.checkpoint_connector import \
            CheckpointConnector

        setattr(CheckpointConnector, "hpc_resume_path", None)

    trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
    trainer.logdir = logdir  ###

    # data
    data = instantiate_from_config(config.data)
    # data = instantiate_from_config_data(config.data, device) # Sy: data is DataModule
    # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    # calling these ourselves should not be necessary but it is.
    # lightning still takes care of proper multiprocessing though
    # data.prepare_data() # Sy: It create the dataset from the provided folder path.
    data.prepare_data() # Sy: 
    data.setup() # Sy: In this code the setup is do the same thing as prepare_data
    print("#### Data #####")
    try:
        for k in data.datasets: # Sy: data.datasets = {'train': <data.matfuse.MatFuseDataset object at 0x7f5e34a5feb0>, 'validation': <data.matfuse.MatFuseDataset object at 0x7f5e34a5ded0>}
            print(
                f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}"
            )
    except:
        print("datasets not yet initialized.")

    # configure learning rate
    bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate

    if not cpu:
        ngpu = len(lightning_config.trainer.gpus.strip(",").split(","))
    else:
        ngpu = 1
    if "accumulate_grad_batches" in lightning_config.trainer:
        accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
    else:
        accumulate_grad_batches = 1
    print(f"accumulate_grad_batches = {accumulate_grad_batches}")
    lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
    if opt.scale_lr:
        model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        print(
            "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr
            )
        )
    else:
        model.learning_rate = base_lr
        print("++++ NOT USING LR SCALING ++++")
        print(f"Setting learning rate to {model.learning_rate:.2e}")

    # allow checkpointing via USR1
    def melk(*args, **kwargs):
        # run all checkpoint hooks
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def divein(*args, **kwargs):
        if trainer.global_rank == 0:
            import pudb

            pudb.set_trace()

    import signal

    signal.signal(signal.SIGUSR1, melk)
    signal.signal(signal.SIGUSR2, divein)

    if opt.gpus: # Sy: opt.gpus == "0,1"
        model = model.to(torch.device(f"cuda:{opt.gpus.split(',')[0]}")) # Sy: opt.gpus.split(',')[0] == "0"

    trainer.fit(model, data) # Sy: Here, the data refers to DataModule which include DataLoader(train&validation).