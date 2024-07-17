from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from einops import rearrange
from PIL import Image

from pathlib import Path
import glob
import os
from PIL import Image
import numpy as np
import shutil
import gc

'''This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.'''
import drjit as dr

import mitsuba as mi
mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')
from mitsuba.scalar_rgb import Transform4f as T

from abc import ABC, abstractmethod
from functools import partial
import yaml
from torch.nn import functional as F
from torchvision import torch

import matplotlib.pyplot as plt

import numpy as np
import random
import torchvision.transforms.functional as TF
from tqdm import tqdm

import math
import time
from  torch.cuda.amp import autocast
import threading 
import torch.multiprocessing as mp

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# Initialize available devices
num_devices = torch.cuda.device_count()
devices = [torch.device(f'cuda:{i}') for i in range(num_devices)]

__OPERATOR__ = {}

def spherical_to_cartesian(r, theta, phi):
    x = r * math.sin(math.pi * (theta/180)) * math.sin(math.pi * (phi/180))
    y = -r * math.cos(math.pi * (theta/180))
    z = r * math.sin(math.pi * (theta/180)) * math.cos(math.pi * (phi/180))
    return x, y, z # MK : 실제 구면좌표계 to 직각좌표계(y축이 up, -z축이 forward) -- mitsuba3
                   # MK : theta - 수직방향, phi - 수평방향 


def load_sensor(r, theta, phi):

    x, y, z = spherical_to_cartesian(r, theta, phi) 

    origin = [x, y, z]

    if theta == 0:
        up_vector = [1, 0, 0]  # Adjusted 'up' vector
    else:
        up_vector = [0, 1, 0]  # Default 'up' vector # MK : mitusba default is [0,1,0]

    sensor = T.look_at(origin=origin, target=[0, 0, 0], up=up_vector) # MK : sensor is 4*4 view matrix, and 4*4 view matrix is defined by r, phi, theta, 
    # MK : so we set r, phi, theta as trainable parameter

    return sensor # MK : refer from https://mitsuba.readthedocs.io/en/latest/src/rendering/multi_view_rendering.html

def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls
    return wrapper


def get_operator(name: str, **kwargs):
    if 'raytracing' in name:
        name='raytracing' 
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

    @abstractmethod
    def transpose(self, data, **kwargs):
        # calculate A^T * X
        pass
    
    def ortho_project(self, data, **kwargs):
        # calculate (I - A^T * A)X
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)

    def project(self, data, measurement, **kwargs):
        # calculate (I - A^T * A)Y - AX
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)


@register_operator(name='noise')
class DenoiseOperator(LinearOperator):
    def __init__(self, device):
        self.device = device
    
    def forward(self, data):
        return data

    def transpose(self, data):
        return data
    
    def ortho_project(self, data):
        return data

    def project(self, data):
        return data

class NonLinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass

    def project(self, data, measurement, **kwargs):
        return data + measurement - self.forward(data) 

@register_operator(name='raytracing')
class RaytracingOperator(NonLinearOperator):
    def __init__(self,
                scene_name,
                ldr,
                scene_path,
                illumi_gamma,
                illumi_scale,
                illumi_normalize,
                texture_res,
                device):
        self.device = device
        self.ldr=ldr
        
        self.scene_name=scene_name
        self.scene=mi.load_file(scene_path)
        self.params = mi.traverse(self.scene)

        
        self.texture_res=texture_res
        # self.lock = threading.Lock()
        self.rgb_images = []                                

        
        self.gamma=illumi_gamma
        self.scale=illumi_scale
        self.normal=illumi_normalize

    @dr.wrap_ad(source='torch', target='drjit')
    def render_multiview(self, envmap, basecolor, normal, metallic, roughness, r=1.5, theta=0, phi=0, spp=256):

        r = np.array(r)
        theta = np.array(theta)
        phi = np.array(phi)

        sensor = load_sensor(r, theta, phi)
        
        self.params['PerspectiveCamera.to_world']=sensor

        # sensor = sensor

        # MK : load envmap
        self.params['EnvironmentMapEmitter.data']=envmap # MK : params is scene discription parameters, and it is used to render the given scene in mi.render funcion

        # MK : test
        self.params['OBJMesh.bsdf.nested_bsdf.base_color.data']=basecolor
        self.params['OBJMesh.bsdf.normalmap.data']=normal
        self.params['OBJMesh.bsdf.nested_bsdf.metallic.data']=metallic
        self.params['OBJMesh.bsdf.nested_bsdf.roughness.data']=roughness
        
        self.params.update()

        basecolor_param = self.params['OBJMesh.bsdf.nested_bsdf.base_color.data']
    
        
        rendered_img = mi.render(self.scene, self.params, spp=spp)
        
        return rendered_img
    
    ############# forward method ###################       
    def forward(self, data,spp=16, **kwargs):
        data=(data+1.)/2
        data=data.clamp(0,1)
        data=self.scale*torch.pow(data/self.normal, self.gamma)
        
        envmap=torch.ones([256,257,3],device=self.device)
        envmap[:,:,:3]*=1e-8

        envmap[:,:256,:3]=data.squeeze().permute(1,2,0)
        envmap[:,256,:3]=envmap[:,255,:3]
            
        rendered_img=self.render_envmap(envmap,spp=spp) 
        rendered_img=rendered_img.permute(2,0,1).unsqueeze(0)
        
    def forward_multiview(self, data_envmap, data_pbrmap, spp=16, **kwargs): # Sy: data_pbrmap, data_envmap are already sent in each device.

        basecolor = data_pbrmap[:, :, :3]
        normal = data_pbrmap[:, :, 3:6]
        roughness = data_pbrmap[:, :, 6:9]
        metallic = data_pbrmap[:, :, 9:]
        
        # TF.to_pil_image(basecolor.permute(2,0,1)).save("/home/sogang/mnt/db_2/oh/MatGen/sample/basecolor.png")
        # TF.to_pil_image(normal.permute(2,0,1)).save("/home/sogang/mnt/db_2/oh/MatGen/sample/normal.png")
        # TF.to_pil_image(roughness.permute(2,0,1)).save("/home/sogang/mnt/db_2/oh/MatGen/sample/roughness.png")
        # TF.to_pil_image(metallic.permute(2,0,1)).save("/home/sogang/mnt/db_2/oh/MatGen/sample/metallic.png")
        # TF.to_pil_image(data_envmap.permute(2,0,1)).save("/home/sogang/mnt/db_2/oh/MatGen/sample/envmap.png")
        
        data_envmap_scale=(data_envmap+1.)/2
        data_envmap_scale=data_envmap.clamp(0,1)
        # data_envmap_scale=self.scale*torch.pow(data_envmap_scale/self.normal, self.gamma)

        # envmap=torch.ones([256,257,3],device=self.device)
        # envmap[:,:,:3]*=1e-8
        # if envmap.shape[-1] == 256:
        #     envmap[:,:256,:3] = data_envmap_scale.permute(1,2,0)
        # else:
        #     envmap[:,:256,:3] = data_envmap_scale
        # envmap[:,256,:3]=envmap[:,255,:3]
        
        envmap=torch.ones([256,257,3],device=self.device)
        envmap[:,:,:3]*=1e-8
        if envmap.shape[-1] == 256:
            envmap[:,:256,:3] = data_envmap_scale.permute(1,2,0)
        else:
            envmap[:,:256,:3] = data_envmap_scale
        envmap[:,256,:3]=envmap[:,255,:3]


        pbrmap = data_pbrmap
        # pbrmap = (data_pbrmap+1.)/2
        # pbrmap = data_pbrmap.clamp(0,1)
        if pbrmap.shape[-1] == 256:
            pbrmap = pbrmap.permute(1,2,0) # Sy: remove squeeze
        
        basecolor = pbrmap[:, :, :3]
        normal = pbrmap[:, :, 3:6]
        roughness = pbrmap[:, :, 6:9]
        metallic = pbrmap[:, :, 9:]
        
        # TF.to_pil_image(basecolor.permute(2,0,1)).save("/home/sogang/mnt/db_2/oh/MatGen/sample/basecolor2.png")
        # TF.to_pil_image(normal.permute(2,0,1)).save("/home/sogang/mnt/db_2/oh/MatGen/sample/normal2.png")
        # TF.to_pil_image(roughness.permute(2,0,1)).save("/home/sogang/mnt/db_2/oh/MatGen/sample/roughness2.png")
        # TF.to_pil_image(metallic.permute(2,0,1)).save("/home/sogang/mnt/db_2/oh/MatGen/sample/metallic2.png")
        # TF.to_pil_image(envmap.permute(2,0,1)).save("/home/sogang/mnt/db_2/oh/MatGen/sample/envmap2.png")
        envmap = torch.roll(envmap, (128, 128), (0, 1))
        rendered_img_ = self.render_multiview(envmap, basecolor, normal, roughness, metallic, spp=spp) # MK : add campos
        
        rendered_img = rendered_img_.permute(2,0,1) # Sy:
        # rendered_img = rendered_img_ ** (1.0 / 2.2)
        
        return rendered_img



class Rendering():
    # Sy: get_obj_from_str(config["target"])(**config.get("params", dict()))
    def __init__(self, data_root, device=torch.device("cuda:1"), size=256, input_pbr_names=["basecolor_2", "normal_2", "roughness_2", "metallic_2"]): # Sy: Why just diffuse only? (torch.device("cuda:0") or torch.device("cuda:1") or torch.device("cuda:2"))
        # Sy: prams = {'data_root': 'sample_materials', 'size': 256, 'input_pbr_names': ['diffuse', 'normal', 'roughness', 'specular'], 'render': 'data/maps/renders'}
        self.data_root = Path(data_root) # Sy: data_root is yaml file data_root

        self.input_pbr_names = input_pbr_names

        self.materials = [
            {"name": x.parent.stem, "folder": x.parent}
            for x in self.data_root.glob("*/basecolor_2.png") # Sy: for example x is '/sample_materials/asphalt_floor/diffuse.png'
        ]

        self.pbr_length = len(self.materials) # Sy: self.materials is refers to the set of names of materials and their folder names defined in yaml file's data_root. In the published code it is 5.
            
        # Sy:
        self.envs = Path("/mnt/1TB/MatGen/env/AG8A5986-25bed8d33f_0.png")

        self.size = size

        self._prng = np.random.RandomState()
        self.device = device

    def __len__(self):
        return self.pbr_length

    def render_preprocess(self, src, name):
        if not src.exists():
            image = self.make_placeholder_map(name)
            image = TF.to_tensor(image)
        else:
            image = Image.open(src).convert("RGB")
            image = TF.resize(image, self.size, antialias=True)
            image = TF.to_tensor(image) # Sy: [0,1]
        
        # image = image * 2.0 - 1.0
        # image = image.clamp(-1.0, 1.0)
            
        image = rearrange(image, "c h w -> h w c")
        return image
    
    def render_env(self, folder, pbrmap, envmap, env_name, device):
        ##### Sy: When loading data, render the selected envmap and pbr map and use the load 
        save_dir = folder/("render")
        # if not os.path.exists(save_dir/"render.png"):
        os.makedirs(save_dir, exist_ok=True)
        torch.cuda.set_device(device)
        render_fn = RaytracingOperator(scene_name="plane",
                                    ldr=True,
                                    scene_path="/mnt/1TB/MatGen/pbrmap/plane_scene.xml",
                                    illumi_gamma=2.4,
                                    illumi_scale=1,
                                    illumi_normalize=0.5,
                                    texture_res=256,
                                    device=device) # Sy: render_fn object has the information about the gpu device to render on.
        
        # pbr = packed[:, :, :12]
        
        render_tensor = render_fn.forward_multiview(envmap.to(self.device), pbrmap.to(self.device)) # Sy: render_tensor is range [0,1]. It should be changed to [-1,1].
        render_image = TF.to_pil_image(render_tensor)
        # print(save_dir)
        render_image.save(save_dir/"render.png")

        torch.cuda.empty_cache()
        gc.collect()
        
        return save_dir

    def render(self, number): # Sy: number = rendering 할 env map 갯수
        for i in tqdm(range(self.pbr_length)):
            # load material
            material = self.materials[i]
            folder = material["folder"]
            

            pbrmap_list = []
            # load maps
            for curr_map in self.input_pbr_names:
                src = folder / (curr_map + ".png")
                pbrmap_list.append(self.render_preprocess(src, curr_map)) # Sy: This tensor range is [0,1]. It uses for input of Mitsuba3 render.
            pbrmap = torch.cat(pbrmap_list, dim=2) # Sy: shape = h, w, c
                
            ##### Sy: load env maps. 1개 pbr map당 number개의 다른 render image 생성
            envmap = self.envs
            
            
            # Sy: This code for rendering to each GPU in parallely. 
            env_folder1 = envmap.parent # Sy: env_chunk[0] is the first env_map dict in env_chunk.
            env_name1 = envmap.stem
            # env_folder2 = env_chunk[1]["folder"] # Sy: env_chunk[1] is the second env_map dict in env_chunk.
            # env_name2 = env_chunk[1]["name"]
            # env_folder3 = env_chunk[2]["folder"] 
            # env_name3 = env_chunk[2]["name"]
            
            env_src1 = env_folder1/(env_name1 + ".png")
            # env_src2 = env_folder2/(env_name2 + ".png")
            # env_src3 = env_folder3/(env_name3 + ".png")
            
            envmap_img1 = self.render_preprocess(env_src1, env_name1)
            # envmap_img2 = self.render_preprocess(env_src2, env_name2)
            # envmap_img3 = self.render_preprocess(env_src3, env_name3)
            
            # dir1, dir2, dir3 = self.render_env(folder, pbrmap, envmap_img1, env_name1, devices[0]), \
            #                     self.render_env(folder, pbrmap, envmap_img2, env_name2, devices[1]), \
            #                     self.render_env(folder, pbrmap, envmap_img3, env_name3, devices[2]) # Sy: 
            dir1 = self.render_env(folder, pbrmap, envmap_img1, env_name1, self.device)
            common = dir1.parent
            # d1, d2, d3 = dir1.stem, dir2.stem, dir3.stem
            d1 = dir1.stem
            
            # print(f"{common}, {d1}, {d2}, {d3}")
            print(f"{common}, {d1}")
        
            torch.cuda.empty_cache()
            gc.collect()
            
            
            
            
if __name__ == "__main__":
    # mp.set_start_method('spawn')
    # root = Path('/mnt/1TB/MatGen/output_matfuse')
    # root = Path('/mnt/1TB/MatGen/output_envmat')
    root = Path('/mnt/1TB/MatGen/output')
    # root = Path('/mnt/1TB/MatGen/data/test')
    
    render = Rendering(data_root=root)
    
    num_of_env_map = 1
    
    render.render(num_of_env_map)