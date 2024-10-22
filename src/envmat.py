import os
import torch
import numpy as np
from PIL import Image
import argparse
import random
from pathlib import Path
import glob
import shutil

from torchvision.transforms import ToTensor, ToPILImage
from ldm.models.diffusion.ddim import DDIMSampler
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from contextlib import contextmanager, nullcontext

import cv2
import einops
import tqdm

import torch
import torch.nn.functional as F
from ldm.data.material_utils import *
from ldm.util import load_model_from_config, visualize_palette
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from torchvision.transforms.functional import center_crop, to_tensor


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

parser = argparse.ArgumentParser(description="MatFuse")
parser.add_argument("--ckpt", type=str, help="Path to the MatFuse model")
parser.add_argument("--config", type=str, help="Path to the MatFuse config")
args = parser.parse_args()


model_config = args.config
model_ckpt = args.ckpt

config = OmegaConf.load(model_config)

model = load_model_from_config(config, model_ckpt)
device = torch.device('cuda:0')
model = model.to(device)
model.eval()



def map_transform_func(x, load_size=256):
    x = TF.resize(x, load_size)
    x = TF.center_crop(x, load_size)
    x = TF.to_tensor(x)
    x = TF.normalize(x, 0.5, 0.5)
    return x

def process_image(image_path: str, img_size: int, device: str):
    # if image is None:
    #     return torch.zeros(3, img_size, img_size, device=device)
    image = Image.open(image_path).convert("RGB")
    image = image.resize((img_size, img_size))
    return map_transform_func(image, img_size).to(device)


# Function to postprocess and save generated images
def postprocess_and_save_images(images, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    component_names = ["basecolor", "normal", "roughness", "metallic"]
    
    for i, img in enumerate(images):
        img_pil = ToPILImage()(img)
        img_pil.save(os.path.join(output_dir, f"generated_pbr_{i+1}.png"))
        
        # Assuming the input image is a 2x2 grid of the components
        w, h = img_pil.size
        half_w, half_h = w // 2, h // 2
        
        # Crop the 2x2 grid into four separate images
        basecolor = img_pil.crop((0, 0, half_w, half_h))
        normal = img_pil.crop((0, half_h, half_w, h))
        roughness = img_pil.crop((half_w, 0, w, half_h))
        metallic = img_pil.crop((half_w, half_h, w, h))
        
        components = [basecolor, normal, roughness, metallic]
        
        # Save each component with the respective name
        for component, name in zip(components, component_names):
            component.save(os.path.join(output_dir, f"{name}_{i+1}.png"))

def postprocess_and_save_images_env(images, output_dir):
    for i, img in enumerate(images):
        img = (img + 1.0) / 2.0
        img = torch.clamp(img, min=0.0, max=1.0)
        img_pil = ToPILImage()(img)
        img_pil.save(os.path.join(output_dir, f"generated_env_{i+1}.png"))
        # img = img.transpose(0, 1).transpose(1, 2)
        # img = img.numpy()
        # img = (img * 255).astype(np.uint8)
        # Image.fromarray(img).save(os.path.join(output_dir, f"generated_env_{i+1}.png"))

# Main function to generate images

@torch.no_grad()
def generate(
    control,
    render_img_path,
    output_dir,
    image_resolution,
    num_samples,
    ddim_steps,
    ddim_eta,
    ucg_scale,
    seed=-1,
    x=None,
    mask=None,
    use_ddim=True,
    use_ema_scope=True,
):
    ema_scope = model.ema_scope if use_ema_scope else nullcontext

    # if seed == -1:
    #     seed = random.randint(0, 65535)
    seed_everything(seed)

    latent_shape = (3, image_resolution // 8, image_resolution // 8)

    unconditional_guidance_label = {
        k: torch.zeros_like(v) for k, v in control.items() if "text" not in k
    }
    unconditional_guidance_label["text"] = [""] * num_samples

    cond = model.get_learned_conditioning(control) # Sy: control is {'image_embed': [1, 3, h, w], 'text': , ... }. cond is {'c_crossattn': [1, 2, 512]}.
    map_samples = torch.tensor([], device=model.device)

    # Basic sampling
    samples, z_denoise_row = model.sample_log(
        cond=cond,
        batch_size=num_samples,
        ddim=use_ddim,
        ddim_steps=ddim_steps,
        eta=ddim_eta,
        x0=x,
        mask=mask,
        image_size=latent_shape[-1],
    )
    samples = F.pad(samples, (7, 7, 7, 7), mode="circular")
    samples_pbr = samples[:, :12, :, :]
    samples_env = samples[:, 12:, :, :]
    x_samples_pbr = model.decode_first_stage(samples_pbr)
    x_samples_env = model.decode_first_stage(samples_env)
    x_samples_pbr = center_crop(x_samples_pbr, (image_resolution, image_resolution))
    x_samples_env = center_crop(x_samples_env, (image_resolution, image_resolution))
    map_samples_pbr = torch.cat([map_samples, x_samples_pbr], dim=0)
    map_samples_env = torch.cat([map_samples, x_samples_env], dim=0)

    # Sampling with EMA
    with ema_scope("Sampling"):
        samples, z_denoise_row = model.sample_log(
            cond=cond,
            batch_size=num_samples,
            ddim=use_ddim,
            ddim_steps=ddim_steps,
            eta=ddim_eta,
            x0=x,
            mask=mask,
            image_size=latent_shape[-1],
        )
    samples = F.pad(samples, (7, 7, 7, 7), mode="circular")
    samples_pbr = samples[:, :12, :, :]
    samples_env = samples[:, 12:, :, :]
    x_samples_ema_pbr = model.decode_first_stage(samples_pbr)
    x_samples_ema_env = model.decode_first_stage(samples_env)
    x_samples_ema_pbr = center_crop(x_samples_ema_pbr, (image_resolution, image_resolution))
    x_samples_ema_env = center_crop(x_samples_ema_env, (image_resolution, image_resolution))
    map_samples_pbr = torch.cat([map_samples_pbr, x_samples_ema_pbr], dim=0)
    map_samples_env = torch.cat([map_samples_env, x_samples_ema_env], dim=0)

    # Sampling with classifier-free guidance
    if ucg_scale > 1.0:
        uc = model.get_unconditional_conditioning(
            num_samples, unconditional_guidance_label
        )
        with ema_scope("Sampling with classifier-free guidance"):
            samples_cfg, _ = model.sample_log(
                cond=cond,
                batch_size=num_samples,
                ddim=use_ddim,
                ddim_steps=ddim_steps,
                eta=ddim_eta,
                unconditional_guidance_scale=ucg_scale,
                unconditional_conditioning=uc,
                x0=x,
                mask=mask,
                image_size=latent_shape[-1],
                reduce_memory=True,
            )
            samples = F.pad(samples, (7, 7, 7, 7), mode="circular")
            x_samples_cfg = model.decode_first_stage(samples_cfg)
            x_samples_cfg = center_crop(
                x_samples_cfg, (image_resolution, image_resolution)
            )
            map_samples = torch.cat([map_samples, x_samples_cfg], dim=0)

    pbrmaps = unpack_maps(map_samples_pbr)
    envmap = map_samples_env.cpu()
    pbrmaps = make_plot_maps(pbrmaps)

    pbrmaps = (
        (einops.rearrange(pbrmaps, "b c h w -> b h w c") * 127.5 + 127.5)
        .cpu()
        .numpy()
        .clip(0, 255)
        .astype(np.uint8)
    )

    pbrmaps = [m for m in pbrmaps]

    results_pbr = [*pbrmaps]
    postprocess_and_save_images(results_pbr, output_dir)
    postprocess_and_save_images_env(envmap, output_dir)
    # Copy the input image to the output directory
    shutil.copy(render_img_path, os.path.join(output_dir, os.path.basename(render_img_path)))
    
    torch.cuda.empty_cache()
    return results_pbr
    
@torch.no_grad()
def run_generation(
    render_img_path,
    output_dir,
    prompt="",
    num_samples=1,
    image_resolution=256,
    ddim_steps=50,
    seed=-1, # Sy: Use -1 as default and then assign a random value if -1
    ddim_eta=0.0,
    ucg_scale=1.0,
    use_ema_scope=True,
    use_ddim=True,
):
    control = {}
   
    control["image_embed"] = process_image(render_img_path, image_resolution, model.device) # Sy: render_emb is PIL render image. And  control["image_embed"] shape is [1, 3, h, w].
    control["text"] = prompt
    control["image_embed"] = torch.stack(
        [control["image_embed"] for _ in range(num_samples)], dim=0
    ) # Sy: control["image_embed"] stacked by [b, 3, h, w] shape.
    control["text"] = [prompt] * num_samples

     # Assign a random seed if seed is -1
    if seed == -1:
        seed = random.randint(0, 65535)
    
    return generate(
        control=control,
        render_img_path=render_img_path,
        output_dir=output_dir,
        image_resolution=image_resolution,
        num_samples=num_samples,
        ddim_steps=ddim_steps,
        ddim_eta=ddim_eta,
        ucg_scale=ucg_scale,
        seed=seed,
        use_ddim=use_ddim,
        use_ema_scope=use_ema_scope,
    )


if __name__ == "__main__":
    input_root = Path("/mnt/1TB/MatGen/data/test")
    output_dir_root = Path("/mnt/1TB/MatGen/output")
    
    folder_paths = []
    for root, categories, files in os.walk(input_root):
        if 'envmap_val_256' in categories:
            categories.remove('envmap_val_256')
        for category in categories:
            path = os.path.join(root, category)
            for _, mats, _ in os.walk(path):
                for mat in mats:
                    j = 0
                    for cr in Path(os.path.join(path, mat)).glob("*/*/render.png"):
                        if j > 15:
                            continue
                        folder_paths.append(cr)
                        j += 1
    print(len(folder_paths))
    
    i = 0
    # j = 0
    # for x in input_root.glob("*/*/*/*/render.png"):
    for x in folder_paths:
        pbr_name = x.parent.parent.parent.stem
        rot_crop = x.parent.parent.stem
        env_name = x.parent.stem[7:]
        render_input = x
        output_dir = output_dir_root/(pbr_name + "-" + rot_crop + "-" + env_name)
        
        if (output_dir.exists() and (output_dir/"render.png").exists()):
            # j += 1
            continue
        
        run_generation(render_input, output_dir)
        print(i)
        i += 1
    # print(f"Finish {i} times. Already exist {j}. Total {i + j}.")
    print(f"Finish {i} times.")

    # render_img_path = "/mnt/1TB/MatGen/data/test/Ceramic/acg_tiles_009/rot_000_crop_000/render_AG8A6425-1d62954fe4_0/render.png"
