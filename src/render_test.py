from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torchvision.transforms.functional import pil_to_tensor
from ldm.modules.losses.rendering import GGXRenderer  # src/ldm/modules/losses/rendering.py

ren = GGXRenderer()

data_root = Path('/home/sogang/oh/MatGen/Croped/train/Concrete/acg_asphalt_005/rot_000_crop_000')
# print(data_root)
map_list = [x.name for x in data_root.glob("*.png")] # 'diffuse.png', 'normal.png', 'height.png', 'roughness.png', 'specular.png'

diffuse = pil_to_tensor(Image.open(data_root._str + '/' + map_list[0]))
normal = pil_to_tensor(Image.open(data_root._str + '/' + map_list[1]))
roughness = pil_to_tensor(Image.open(data_root._str + '/' + map_list[3]).convert("RGB"))
# roughness = pil_to_tensor(Image.open(data_root._str + '/' + map_list[3]))
specular = pil_to_tensor(Image.open(data_root._str + '/' + map_list[4]))

svbrdf = torch.cat((diffuse, normal, roughness, specular),dim=0)
svbrdf = torch.permute(svbrdf, (1,2,0))

