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

# from util.resizer import Resizer
# from util.img_utils import fft2_m,imread

import numpy as np
import random
from PIL import Image 
import torchvision.transforms.functional as TF


import math
import time
from  torch.cuda.amp import autocast

import os
import glob

### Sy:
import gc
import threading 

# =================
# Operation classes
# =================

__OPERATOR__ = {}

def spherical_to_cartesian(r, theta, phi):
    x = r * math.sin(math.pi * (theta/180)) * math.sin(math.pi * (phi/180))
    y = -r * math.cos(math.pi * (theta/180))
    z = r * math.sin(math.pi * (theta/180)) * math.cos(math.pi * (phi/180))
    return x, y, z # MK : 실제 구면좌표계 to 직각좌표계(y축이 up, -z축이 forward) -- mitsuba3
                   # MK : theta - 수직방향, phi - 수평방향 


def load_sensor(r, theta, phi):
    # Apply two rotations to convert from spherical coordinates to world 3D coordinates.
    # origin = T.rotate([0, 0, 1], phi).rotate([0, 1, 0], theta) @ mi.ScalarPoint3f([0, 0, r])
    # MK : @ -- 앞쪽이 메트릭스, 뒤쪽이 백터 --> 새로운 백터
    # MK : [0, 0, 1]을 기준으로 phi만큼 회전, [0, 1, 0]을 중심으로 theta만틈 회전 --> 이게 잘 작동하는지 확인필요(unit test)
    # MK : T.rotate + .rotate + @ --> 이렇게 한 줄로 작성하는것은 위험하다. 3줄로 분리해서 작성

    # # Dr.Jit 변수를 numpy 배열로 변환
    # r_numpy = r.numpy()
    # phi_numpy = phi.numpy()
    # theta_numpy = theta.numpy()

    # # numpy 배열을 파이썬의 float 타입으로 변환
    # r = float(r_numpy)
    # phi = float(phi_numpy)
    # theta = float(theta_numpy)

    x, y, z = spherical_to_cartesian(r, theta, phi) 

    origin = [x, y, z]


    # Adjust the 'up' vector when theta is 0 to avoid rendering issues.
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
                #  image_path,
                #  n_images,
                ldr,
                scene_path,
                # camera_path,
                illumi_gamma,
                illumi_scale,
                illumi_normalize,
                texture_res,
                device):
        self.device = device
        self.ldr=ldr
        
        self.scene_name=scene_name
        # self.image_path=image_path
        self.scene=mi.load_file(scene_path)
        # self.cam_scene=mi.load_file(camera_path)
        self.params = mi.traverse(self.scene)
        # self.cam_params = mi.traverse(self.cam_scene)
        
        self.texture_res=texture_res
        # self.lock = threading.Lock()
        # self.n_images=n_images
        self.rgb_images = []                                
        # for i in range(self.n_images):
        #     if self.ldr:  
        #         rgb_img = imread('{}/{}.png'.format(self.image_path,i),gamma=1)[:,:,:3]
        #     else:
        #         rgb_img = imread('{}/{}.exr'.format(self.image_path,i))[:,:,:3]
            #  self.rgb_images.append(rgb_img.to(self.device))
        

 
        
        # self.basecolor=0.01*torch.ones([self.texture_res,self.texture_res,3],device=torch.device('cuda')) 
        # self.metallic=self.params['OBJMesh.bsdf.nested_bsdf.metallic.data'].torch().clone()[:,:,0:1]
        # self.roughness=self.params['OBJMesh.bsdf.nested_bsdf.roughness.data'].torch().clone()[:,:,0:1]
        
        # self.basecolor.requires_grad=True
        # self.metallic.requires_grad=True
        # self.roughness.requires_grad=True
        
        # self.optimizer = torch.optim.Adam([
        #                                 {'params': [self.basecolor]},
        #                                 {'params': [self.roughness], 'lr': 1.5e-2},
        #                                 {'params': [self.metallic], 'lr': 1e-2},
                                      
        #             ], lr=2e-2)
        
        
        # self.loss_fn = torch.nn.MSELoss()
        
        self.gamma=illumi_gamma
        self.scale=illumi_scale
        self.normal=illumi_normalize
    
    
    
     
    def reset_optimizer(self):
        
        self.basecolor=0.01*torch.ones([self.texture_res,self.texture_res,3],device=torch.device('cuda')) 
        self.metallic=self.metallic.detach().clone()
        self.roughness=self.roughness.detach().clone()
        
        self.basecolor.data=torch.nan_to_num(self.basecolor.data.clamp(1e-8,1-1e-8))
        self.metallic.data=torch.nan_to_num(self.metallic.data.clamp(1e-8,1-1e-8))
        self.roughness.data=torch.nan_to_num(self.roughness.data.clamp(1e-8,1-1e-8))
        
        self.basecolor.requires_grad=True
        self.metallic.requires_grad=True
        self.roughness.requires_grad=True
        
        self.optimizer = torch.optim.Adam([
                                        {'params': [self.basecolor]},
                                        {'params': [self.roughness], 'lr': 1.5e-2},
                                        {'params': [self.metallic], 'lr': 1e-2},
                                      
                    ], lr=2e-2)#
        
        
############# rendering method ###################        
    @dr.wrap_ad(source='torch', target='drjit')
    def render_envmap(self,envmap,spp=256):
        
        self.params['PerspectiveCamera.to_world']=self.cam_params['PerspectiveCamera.to_world']
        
        self.params['EnvironmentMapEmitter.data']=envmap  
        self.params.update()
        rendered_img=mi.render(self.scene, self.params, spp=spp)

        return rendered_img
    
    @dr.wrap_ad(source='torch', target='drjit')
    def render_test(self,envmap, i, spp=256):

        cam = i
        if cam==0:
            self.params['PerspectiveCamera.to_world']=self.cam_params['PerspectiveCamera.to_world']
        else:
            self.params['PerspectiveCamera.to_world']=self.cam_params['PerspectiveCamera_{}.to_world'.format(cam)]
        
        
        
        self.params['EnvironmentMapEmitter.data']=envmap
        self.params.update()
        # MK : 단순히 카메라와 envmap만 update
        
        rendered_gt=self.rgb_images[cam].cuda()
        
        rendered_img=mi.render(self.scene, self.params, spp=spp)

        return rendered_gt, rendered_img
    
    @dr.wrap_ad(source='torch', target='drjit')
    def render_multiview_with_material(self,envmap,basecolor,roughness,metallic,spp=256): 
        
        
        cam=random.randint(0,self.n_images-1)
 
        if cam==0:
            self.params['PerspectiveCamera.to_world']=self.cam_params['PerspectiveCamera.to_world']
        else:
            self.params['PerspectiveCamera.to_world']=self.cam_params['PerspectiveCamera_{}.to_world'.format(cam)]
        
      
        
        self.params['EnvironmentMapEmitter.data']=envmap # MK : 노이즈가 추가가 안된 envmap
        self.params['OBJMesh.bsdf.base_color.data']=basecolor
        self.params['OBJMesh.bsdf.roughness.data']=roughness
        self.params['OBJMesh.bsdf.metallic.data']=metallic

        self.params.update() 
        

        
        rendered_gt=self.rgb_images[cam].cuda()
        
        
        rendered_img=mi.render(self.scene, self.params, spp=spp)

       

        return rendered_gt,rendered_img
    
    ########## Sy: 이거 추천. ###########
    @dr.wrap_ad(source='torch', target='drjit')
    def render_multiview(self, envmap, basecolor, normal, metallic, roughness, r=1.5, theta=0, phi=0, spp=256):
        # MK : cam=random.randint(0,self.n_images-1)
        # cam=0 # MK : random selec the camera
        # if cam==0: # MK : is default camera
        # self.params['PerspectiveCamera.to_world']=self.cam_params['PerspectiveCamera.to_world']
        # sensor=load_sensor(campos[0], campos[1], campos[2])

        # else:
        #     self.params['PerspectiveCamera.to_world']=self.cam_params['PerspectiveCamera_{}.to_world'.format(cam)] # MK : it sets the current camera to the selected camera
        # r = campos[0][0]
        # theta = campos[0][1]
        # phi = campos[0][2]

        r = np.array(r)
        theta = np.array(theta)
        phi = np.array(phi)

        sensor = load_sensor(r, theta, phi)
        # camera = mi.traverse(sensor)

        # MK : self.params is dic
        
        self.params['PerspectiveCamera.to_world']=sensor

        # sensor = sensor

        # MK : load envmap
        self.params['EnvironmentMapEmitter.data']=envmap # MK : params is scene discription parameters, and it is used to render the given scene in mi.render funcion
        # self.params['emitter.data'] = mi.TensorXf(mi.Bitmap(envmap).convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32))

        # MK : test
        self.params['OBJMesh.bsdf.nested_bsdf.base_color.data']=basecolor
        self.params['OBJMesh.bsdf.normalmap.data']=normal
        self.params['OBJMesh.bsdf.nested_bsdf.metallic.data']=metallic
        self.params['OBJMesh.bsdf.nested_bsdf.roughness.data']=roughness
        
        # MK : load pbrmap
        # self.params['OBJMesh.bsdf.basecolor.data']=pbrmap[:,:3,:,:]
        # self.params['OBJMesh.bsdf.normal.data']=pbrmap[:,3:6,:,:]
        # self.params['OBJMesh.bsdf.metallic.data']=pbrmap[:,6:7,:,:]
        # self.params['OBJMesh.bsdf.roughness.data']=pbrmap[:,7:8,:,:]
        self.params.update()
        # time.sleep(1)
        # rendered_gt=self.rgb_img.cuda() # Sy:
        # MK : rgb_images[cam].cuda() # MK : because we have only one input image, rendered_gt is always the same
        
        basecolor_param = self.params['OBJMesh.bsdf.nested_bsdf.base_color.data']
        
        rendered_img=mi.render(self.scene, self.params, spp=spp) # MK : render the image from the given scene using the scene discription parameter, spp refers to number of renderings to create its pixel.
        # MK : 실제로 multiview이미지로 실험해보고, 결과 비교self.scene, 
       

        # return rendered_gt, rendered_img, basecolor_param # Sy: rendered_gt = ground truth render image, basecolor_param = basecolor map
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
        
        
        
        return rendered_img
    
    def forward_gt(self,spp=16, **kwargs):

        cam=random.randint(0,self.n_images-1)
        rendered_img=self.rgb_images[cam].cuda()
            
        return rendered_img.permute(2,0,1).unsqueeze(0)
    

    
    def forward_multiview(self, data_envmap, data_pbrmap, spp=16, **kwargs): # MK : data = sample is envmap
        ### Sy: The input pbrmap & envmap is correct.
        # basecolor = data_pbrmap[:, :, :3]
        # normal = data_pbrmap[:, :, 3:6]
        # roughness = data_pbrmap[:, :, 6:9]
        # metallic = data_pbrmap[:, :, 9:]
        
        # TF.to_pil_image(basecolor.permute(2,0,1)).save("/home/sogang/mnt/db_2/oh/MatGen/sample/basecolor.png")
        # TF.to_pil_image(normal.permute(2,0,1)).save("/home/sogang/mnt/db_2/oh/MatGen/sample/normal.png")
        # TF.to_pil_image(roughness.permute(2,0,1)).save("/home/sogang/mnt/db_2/oh/MatGen/sample/roughness.png")
        # TF.to_pil_image(metallic.permute(2,0,1)).save("/home/sogang/mnt/db_2/oh/MatGen/sample/metallic.png")
        # TF.to_pil_image(data_envmap.permute(2,0,1)).save("/home/sogang/mnt/db_2/oh/MatGen/sample/envmap.png")
        
        data_envmap_scale=(data_envmap+1.)/2
        data_envmap_scale=data_envmap.clamp(0,1)
        # data_envmap_scale=self.scale*torch.pow(data_envmap_scale/self.normal, self.gamma)
        
        
        envmap=torch.ones([256,257,3],device=self.device)
        envmap[:,:,:3]*=1e-8

        # envmap[:,:256,:3]=data_envmap_scale.squeeze().permute(1,2,0)
        if envmap.shape[-1] == 256:
            envmap[:,:256,:3] = data_envmap_scale.permute(1,2,0)
        else:
            envmap[:,:256,:3] = data_envmap_scale.permute(1,2,0)
        envmap[:,256,:3]=envmap[:,255,:3]
        # envmap = (envmap+1.)/2
        # envmap = envmap.squeeze().permute(1,2,0)
        # envmap = (envmap + 1.)/2
        # envmap = envmap.clamp(0,1)
        # envmap = self.scale*torch.pow(envmap/self.normal, self.gamma)
        # envmap_scale = torch.ones([256,257,3],device=self.device)
        # envmap_scale[:,:,:3]*=1e-8
        # envmap_scale[:,:256,:3]=envmap.squeeze().permute(1,2,0)


        # envmap = envmap.squeeze().permute(1,2,0)

        data_pbrmap_scale = (data_pbrmap+1.)/2
        pbrmap = data_pbrmap_scale.clamp(0,1)
        # pbrmap = pbrmap.squeeze().permute(1,2,0)
        if pbrmap.shape[-1] == 256:
            pbrmap = pbrmap.permute(1,2,0) # Sy: remove squeeze
        else:
            pbrmap = pbrmap.permute(1,2,0)
        # data_pbrmap_scale = self.scale*torch.pow(data_pbrmap_scale/self.normal, self.gamma)

        # pbrmap=torch.ones([256,257,8],device=self.device)
        # pbrmap[:,:,:8]*=1e-8

        # pbrmap[:,:256,:8] = data_pbrmap_scale.squeeze().permute(1,2,0)
        # pbrmap[:,256,:8]=pbrmap[:,255,:8]

        # pbrmap = pbrmap.squeeze().permute(1,2,0).clamp(0,1)
        pbrmap = pbrmap.clamp(0,1)
        
        basecolor = pbrmap[:, :, :3]
        normal = pbrmap[:, :, 3:6]
        roughness = pbrmap[:, :, 6:9]
        metallic = pbrmap[:, :, 9:]
        
        # TF.to_pil_image(basecolor.permute(2,0,1)).save("/home/sogang/mnt/db_2/oh/MatGen/sample/basecolor2.png")
        # TF.to_pil_image(normal.permute(2,0,1)).save("/home/sogang/mnt/db_2/oh/MatGen/sample/normal2.png")
        # TF.to_pil_image(roughness.permute(2,0,1)).save("/home/sogang/mnt/db_2/oh/MatGen/sample/roughness2.png")
        # TF.to_pil_image(metallic.permute(2,0,1)).save("/home/sogang/mnt/db_2/oh/MatGen/sample/metallic2.png")
        # TF.to_pil_image(envmap.permute(2,0,1)).save("/home/sogang/mnt/db_2/oh/MatGen/sample/envmap2.png")

        # campos = campos[0]
        # r = campos[0]
        # theta = campos[1]
        # phi = campos[2]
        # sensor=load_sensor(r, theta, phi)
        # time.sleep(1)
        rendered_img_ = self.render_multiview(envmap, basecolor, normal, roughness, metallic, spp=spp) # MK : add campos
        
        # rendered_img = np.array(rendered_img)
        # rendered_img = torch.from_numpy(rendered_img).cuda()
        # rendered_img.requires_grad = True
        
        # rendered_gt=rendered_gt.permute(2,0,1).unsqueeze(0)
        # rendered_img_ = rendered_img_.permute(2,0,1).unsqueeze(0)
        rendered_img = rendered_img_.permute(2,0,1) # Sy:
        # rendered_img = rendered_img_ ** (1.0 / 2.2) # Sy: Add Gamma.
        
        # image_array = rendered_gt.squeeze().permute(1,2,0)
        # image_array = image_array.detach()
        # image_array = image_array.cpu().numpy()

        # image_array_ = rendered_img.squeeze().permute(1,2,0).clamp(0,1)
        # image_array_ = image_array_.detach()
        # image_array_ = image_array_.cpu().numpy()

        # image_array__ = basecolor.detach().cpu().numpy()
        # image_array___ = bs.detach().cpu().nuimsave("/home/sogang/mnt/db_2/woduck/PSLD_fix/stable-diffusion/scripts/results/rendered_image/rendered_gt_image.png", image_imsave("/home/sogang/mnt/db_2/woduck/PSLD_fix/stable-diffusion/scripts/results/rendered_image/rendered_img_image.png", image_aimsave("/home/sogang/mnt/db_2/woduck/PSLD_fix/stable-diffusion/scripts/results/rendered_image/basecolor.png", image_arimsave("/home/sogang/mnt/db_2/woduck/PSLD_fix/stable-diffusion/scripts/results/rendered_image/bs.png", image_array___)

        return rendered_img
    
    ### Sy: back-up code
    def forward_multiview_latent(self, data_envmap, data_pbrmap, spp=16, **kwargs): # MK : data = sample is envmap
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
        data_envmap_scale=data_envmap_scale.clamp(0,1)
        # data_envmap_scale=self.scale*torch.pow(data_envmap_scale/self.normal, self.gamma)
        
        
        envmap=torch.ones([256,257,3],device=self.device)
        envmap[:,:,:3]*=1e-8

        # envmap[:,:256,:3]=data_envmap_scale.squeeze().permute(1,2,0)
        envmap[:,:256,:3] = data_envmap_scale.permute(1,2,0)
        envmap[:,256,:3]=envmap[:,255,:3]
        # envmap = (envmap+1.)/2
        # envmap = envmap.squeeze().permute(1,2,0)
        # envmap = (envmap + 1.)/2
        # envmap = envmap.clamp(0,1)
        # envmap = self.scale*torch.pow(envmap/self.normal, self.gamma)
        # envmap_scale = torch.ones([256,257,3],device=self.device)
        # envmap_scale[:,:,:3]*=1e-8
        # envmap_scale[:,:256,:3]=envmap.squeeze().permute(1,2,0)


        # envmap = envmap.squeeze().permute(1,2,0)

        data_pbrmap_scale = (data_pbrmap+1.)/2
        pbrmap = data_pbrmap_scale.clamp(0,1)
        # pbrmap = pbrmap.squeeze().permute(1,2,0)
        if pbrmap.shape[2] == 256:
            pbrmap = pbrmap.permute(1,2,0) # Sy: remove squeeze
        # data_pbrmap_scale = self.scale*torch.pow(data_pbrmap_scale/self.normal, self.gamma)

        # pbrmap=torch.ones([256,257,8],device=self.device)
        # pbrmap[:,:,:8]*=1e-8

        # pbrmap[:,:256,:8] = data_pbrmap_scale.squeeze().permute(1,2,0)
        # pbrmap[:,256,:8]=pbrmap[:,255,:8]

        # pbrmap = pbrmap.squeeze().permute(1,2,0).clamp(0,1)
        
        basecolor = pbrmap[:, :, :3]
        normal = pbrmap[:, :, 3:6]
        roughness = pbrmap[:, :, 6:9]
        metallic = pbrmap[:, :, 9:]
        
        TF.to_pil_image(basecolor.permute(2,0,1)).save("/home/sogang/mnt/db_2/oh/MatGen/sample/basecolor2.png")
        TF.to_pil_image(normal.permute(2,0,1)).save("/home/sogang/mnt/db_2/oh/MatGen/sample/normal2.png")
        TF.to_pil_image(roughness.permute(2,0,1)).save("/home/sogang/mnt/db_2/oh/MatGen/sample/roughness2.png")
        TF.to_pil_image(metallic.permute(2,0,1)).save("/home/sogang/mnt/db_2/oh/MatGen/sample/metallic2.png")
        TF.to_pil_image(envmap.permute(2,0,1)).save("/home/sogang/mnt/db_2/oh/MatGen/sample/envmap2.png")

        # campos = campos[0]
        # r = campos[0]
        # theta = campos[1]
        # phi = campos[2]
        # sensor=load_sensor(r, theta, phi)
            
        rendered_img_ = self.render_multiview(envmap, basecolor, normal, roughness, metallic, spp=spp) # MK : add campos
        
        # rendered_img = np.array(rendered_img)
        # rendered_img = torch.from_numpy(rendered_img).cuda()
        # rendered_img.requires_grad = True
        
        # rendered_gt=rendered_gt.permute(2,0,1).unsqueeze(0)
        # rendered_img_ = rendered_img_.permute(2,0,1).unsqueeze(0)
        rendered_img_ = rendered_img_.permute(2,0,1) # Sy:
        rendered_img = rendered_img_ ** (1.0 / 2.2) # Sy: Add Gamma.
        
        # image_array = rendered_gt.squeeze().permute(1,2,0)
        # image_array = image_array.detach()
        # image_array = image_array.cpu().numpy()

        # image_array_ = rendered_img.squeeze().permute(1,2,0).clamp(0,1)
        # image_array_ = image_array_.detach()
        # image_array_ = image_array_.cpu().numpy()

        # image_array__ = basecolor.detach().cpu().numpy()
        # image_array___ = bs.detach().cpu().nuimsave("/home/sogang/mnt/db_2/woduck/PSLD_fix/stable-diffusion/scripts/results/rendered_image/rendered_gt_image.png", image_imsave("/home/sogang/mnt/db_2/woduck/PSLD_fix/stable-diffusion/scripts/results/rendered_image/rendered_img_image.png", image_aimsave("/home/sogang/mnt/db_2/woduck/PSLD_fix/stable-diffusion/scripts/results/rendered_image/basecolor.png", image_arimsave("/home/sogang/mnt/db_2/woduck/PSLD_fix/stable-diffusion/scripts/results/rendered_image/bs.png", image_array___)

        return rendered_img
    
    def forward_test(self, data,spp=16, **kwargs):
        data_scale=(data+1.)/2
        data_scale=data_scale.clamp(0,1)
        data_scale=self.scale*torch.pow(data_scale/self.normal, self.gamma)
        
        
        envmap=torch.ones([256,257,3],device=self.device)
        envmap[:,:,:3]*=1e-8

        envmap[:,:256,:3]=data_scale.squeeze().permute(1,2,0)
        envmap[:,256,:3]=envmap[:,255,:3]

        mse_loss_list = []
        norm_list = []
        rendered_gt_list = []
        rendered_img_list = []

        for i in range(60):
            
            rendered_gt,rendered_img = self.render_test(envmap, i ,spp=spp)

            rendered_gt=rendered_gt.permute(2,0,1).unsqueeze(0)
            rendered_img=rendered_img.permute(2,0,1).unsqueeze(0)

            difference = rendered_gt - rendered_img # MK : difference shape [1, 3, 512, 512]        512*512 = 262144
        
            norm = torch.linalg.norm(difference) # MK : norm -- L2 loss 

            # MSE 손실 계산
            mse_loss = F.mse_loss(rendered_gt, rendered_img) # MK : MSE, n = num_pixels, [1, 3, 512, 512] 

            rendered_gt_list.append(rendered_gt)
            rendered_img_list.append(rendered_img)

            norm_list.append(norm)
            mse_loss_list.append(mse_loss)
        

        return rendered_gt_list,rendered_img_list, norm_list, mse_loss_list
        
        # image_array = rendered_gt.clamp(0,1)
        # image_array = image_array.detach()
        # image_array = image_array.cpu().numpy()
        # # MK : .squeeze().permute(1,2,0)


        # image_array_ = rendered_img.clamp(0,1)
        # image_array_ = image_array_.detach()
        # image_array_ = image_array_.cpu().numpy()

        # plt.imsave("/home/sogang/woduck/DPI/re/rendered_image/rendered_gt_image.png", image_array)
        # plt.imsave("/home/sogang/woduck/DPI/re/rendered_image/rendered_img_image.png", image_array_)
        
        
        
        
        
        
        
        
        # return rendered_gt_list,rendered_img_list, norm_list, mse_loss_list
   
    def update_material(self, data,spp=32,t=0., **kwargs):
        with autocast():
            data_scale=(data+1.)/2
            data_scale=data_scale.clamp(0,1)
            data_scale=self.scale*torch.pow(data_scale/self.normal, self.gamma)
            
            
            
            
            envmap=torch.ones([256,257,3],device=self.device)
            envmap[:,:,:3]*=1e-8
            envmap[:,:256,:3]=data_scale.squeeze().permute(1,2,0).detach()
            envmap[:,256,:3]=envmap[:,255,:3]
            
          
            
            rendered_gt,rendered_img=self.render_multiview_with_material(envmap,self.basecolor,self.roughness,self.metallic,spp=spp)
            
            loss = self.loss_fn(rendered_img, rendered_gt)    # MK : self.basecolor - 위에서 정의 
            
        
        
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward() 
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.basecolor.data=torch.nan_to_num(self.basecolor.data.clamp(min=1e-8))
            self.metallic.data=torch.nan_to_num(self.metallic.data.clamp(1e-8,1-1e-8))
            self.roughness.data=torch.nan_to_num(self.roughness.data.clamp(min=1e-8,max=1.),nan=1.0)
            if int(t*1000)%1.5==0:
                self.basecolor.data=self.basecolor.data/self.basecolor.data.max()
                self.roughness.data=self.roughness.data/self.roughness.data.max()
           
        del(loss,rendered_img,envmap)
    

        
# =============
# Noise classes
# =============


__NOISE__ = {}

def register_noise(name: str):
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already defined!")
        __NOISE__[name] = cls
        return cls
    return wrapper

def get_noise(name: str, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser

class Noise(ABC):
    def __call__(self, data):
        return self.forward(data)
    
    @abstractmethod
    def forward(self, data):
        pass

@register_noise(name='clean')
class Clean(Noise):
    def forward(self, data):
        return data

@register_noise(name='gaussian')
class GaussianNoise(Noise):
    def __init__(self, sigma):
        self.sigma = sigma
    
    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma


@register_noise(name='poisson')
class PoissonNoise(Noise):
    def __init__(self, rate):
        self.rate = rate

    def forward(self, data):
        '''
        Follow skimage.util.random_noise.
        '''

        # TODO: set one version of poisson
       
        # version 3 (stack-overflow)
        import numpy as np
        data = (data + 1.0) / 2.0
        data = data.clamp(0, 1)
        device = data.device
        data = data.detach().cpu()
        data = torch.from_numpy(np.random.poisson(data * 255.0 * self.rate) / 255.0 / self.rate)
        data = data * 2.0 - 1.0
        data = data.clamp(-1, 1)
        return data.to(device)

        # version 2 (skimage)
        # if data.min() < 0:
        #     low_clip = -1
        # else:
        #     low_clip = 0

    
        # # Determine unique values in iamge & calculate the next power of two
        # vals = torch.Tensor([len(torch.unique(data))])
        # vals = 2 ** torch.ceil(torch.log2(vals))
        # vals = vals.to(data.device)

        # if low_clip == -1:
        #     old_max = data.max()
        #     data = (data + 1.0) / (old_max + 1.0)

        # data = torch.poisson(data * vals) / float(vals)

        # if low_clip == -1:
        #     data = data * (old_max + 1.0) - 1.0
       
        # return data.clamp(low_clip, 1.0)