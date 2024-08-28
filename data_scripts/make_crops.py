import argparse
import math
import threading
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm


def rotate_normal_map(normal_map, angle_deg):
    angle_rad = angle_deg * (torch.pi / 180.0)

    normal_map  = normal_map * 2.0 - 1.0 # Convert to [-1, 1]
    normal_map = normal_map.unsqueeze(0) # Add batch dimension

    # Rotate the Vectors
    rotation_matrix = torch.tensor([[math.cos(angle_rad), -math.sin(angle_rad), 0],
                                    [math.sin(angle_rad), math.cos(angle_rad), 0],
                                    [0, 0, 1]], device=normal_map.device)

    # Reshape for batch matrix multiplication
    reshaped_normal_map = normal_map.view(1, 3, -1)  # Reshape to [1, 3, H*W]
    rotation_matrix = rotation_matrix.view(1, 3, 3)  # Add batch dimension

    # Rotate the vectors
    rotated_vectors = torch.bmm(rotation_matrix, reshaped_normal_map)
    rotated_vectors = rotated_vectors.view(1, 3, normal_map.size(2), normal_map.size(3))

    rotated_vectors = rotated_vectors / 2.0 + 0.5 # Convert back to [0, 1]

    return rotated_vectors[0]

def process_map(map, mat_dest):
    map_name = map.stem
    img = Image.open(map)
    img = TF.to_tensor(img).cuda()
    img = TF.resize(img, (4096, 4096), antialias=True)

    img = img.repeat(1, 3, 3)
    img = TF.center_crop(img, (5793, 5793))

    for rot_angle in range(0, 360, 45):
        crop_i = 0

        if "normal" in map_name:
            # rot_img = rotate_normal_map(img, axis='z', angle_deg=rot_angle) # Sy: TypeError: rotate_normal_map() got an unexpected keyword argument 'axis' => So I changed the code as below.
            rot_img = rotate_normal_map(img, angle_deg=rot_angle)
            rot_img = TF.rotate(rot_img, rot_angle)
        else:
            rot_img = TF.rotate(img, rot_angle)

        rot_img = TF.center_crop(rot_img, (4096, 4096))
        
        # for crop_res in [4096, 2048, 1024]:
        for crop_res in [4096, 2048]: # Sy: 1024는 제외하고 랜더링
            # split into crops
            crops = rot_img.unfold(1, crop_res, crop_res).unfold(2, crop_res, crop_res)
            crops = crops.permute(1, 2, 0, 3, 4)
            crops = crops.reshape(-1, crops.size(2), crop_res, crop_res)

            for crop in crops:
                crop_dir = mat_dest / f"rot_{rot_angle:03d}_crop_{crop_i:03d}"
                crop_dir.mkdir(parents=True, exist_ok=True)

                # crop = TF.resize(crop, (1024, 1024), antialias=True)
                crop = TF.resize(crop, (256, 256), antialias=True) # Sy: Resize croped map to 256, 256
                
                if map_name in ["height", "displacement"]:
                    crop = crop.permute(1, 2, 0).cpu().numpy()
                    crop = crop.astype(np.uint16)
                    crop = Image.fromarray(crop[..., 0])
                    crop.save(crop_dir / f"{map_name}.png")
                else:
                    TF.to_pil_image(crop).save(crop_dir / f"{map_name}.png")

                crop_i += 1

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Make dataset crops.")
    parser.add_argument("--source_dir", required=True, help="Directory where the original 4K maps are stored.")
    parser.add_argument("--dest_dir", required=True , help="Destination directory to store the 1K crops.")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    dest_dir = Path(args.dest_dir)

    # Find all materials in the source directory
    for file in tqdm([x for x in source_dir.glob("*/*/*/basecolor.png")]):
        mat_dir = file.parent

        name = mat_dir.stem # sy: material name
        category = mat_dir.parent.stem # sy: folder category (Blends, ..)
        split = mat_dir.parent.parent.stem # sy: train/test

        mat_dest = dest_dir / split / category / name # sy: PosixPath
        mat_dest.mkdir(parents=True, exist_ok=True)

        thread = []
        for map in mat_dir.glob("*.png"): # sy: map is the name of pbr map(: basecolor, normal, diffuse...)
            # Sy: I can process the name of mpa like {map.name} is looks like 'normal.png'
            if (map.name == 'diffuse.png' or map.name == 'normal.png' or map.name == 'roughness.png' or map.name == 'specular.png' or map.name == 'basecolor.png' or map.name == 'metallic.png'): 
                t = threading.Thread(target=process_map, args=(map, mat_dest))
                t.start()

                thread.append(t)

        for t in thread:
            t.join()