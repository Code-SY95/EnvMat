import torch
import torch.nn as nn
from torch.nn.functional import mse_loss as l2_loss

from ldm.modules.losses.rendering import GGXRenderer
from ldm.util import instantiate_from_config


class MaterialLoss(nn.Module):

    def __init__(self, imgloss):
        super().__init__()
        self.imgloss = instantiate_from_config(imgloss)

        self.renderer = GGXRenderer()

        self.discriminator = (
            self.imgloss.discriminator
        )  # needed for AutoencoderKL model

    def forward(self, codebook_loss, inputs, reconstructions, *args, **kwargs): # Sy: inputs = x, reconstructions = xrec
        # Split inputs into 3-channel images for loss (LPIPS), aggregate computed losses
        loss_out = None
        log_out = {}

        for i_c in range(0, 12, 3): # Sy: i_c는 각 map별로 짜르기 위한 index. 이 반복문을 통해 각 map별 loss를 계산하고 log_out에 각 map별 loss를 합산
            inputs_c = inputs[:, i_c : i_c + 3, :, :]
            reconstructions_c = reconstructions[:, i_c : i_c + 3, :, :]

            loss_c, log_c = self.imgloss(
                codebook_loss, inputs_c, reconstructions_c, *args, **kwargs
            )

            if loss_out is None:
                loss_out = loss_c
                log_out = log_c 
            else:
                loss_out += loss_c # Sy: log_out에 각 map별 loss를 합산
                for k in log_c.keys():
                    log_out[k] += log_c[k] # Sy: log_out에 각 map별 loss를 합산

        # Compute render loss (only when optimizing the reconstruction loss: optimizer_idx == 0)
        if args[0] == 0:
            inputs = inputs * 0.5 + 0.5
            reconstructions = reconstructions.clamp(-1, 1).clone()
            reconstructions = reconstructions * 0.5 + 0.5

            # # convert diffuse and specular from sRGB to linear
            inputs[:, :3] = inputs[:, :3] ** 2.2
            # inputs[:, 9:12] = inputs[:, 9:12] ** 2.2 # Sy: Change to specular -> metallic. Gray scale image, so no need to multiply by gamma.

            rec_diff, rec_norm, rec_rough, rec_spec = reconstructions.chunk(4, dim=1) # Sy: [b, 12, 256, 256] tensor를 4개의 map 순서(d, n, r, s)대로 나눔
            rec_diff = rec_diff**2.2
            # rec_spec = rec_spec**2.2 # Sy: Change to specular -> metallic. Gray scale image, so no need to multiply by gamma.
            reconstructions = torch.cat(
                [rec_diff, rec_norm, rec_rough, rec_spec], dim=1
            )

            # convert to [-1, 1]
            inputs = inputs.permute(0, 2, 3, 1) * 2 - 1
            reconstructions = reconstructions.permute(0, 2, 3, 1) * 2 - 1

            # compute renderings # Sy: Using randomly selected view & light position.
            # rend_diff_in, rend_diff_rec = self.renderer.generateDiffuseRendering( 
            #     1, 9, inputs, reconstructions
            # )
            # rend_spec_in, rend_spec_rec = self.renderer.generateDiffuseRendering( 
            #     1, 9, inputs, reconstructions
            # ) # Sy : Why didn't I call generateSpecularRendering?
            rend_in, rend_rec = self.renderer.generateRendering( # Sy: surfaceArray=mesh
                1, 9, inputs, reconstructions
            )

            # compute loss. # Sy: Rather than using an already rendered image, they put x and xrec into the Render eq and counted the difference.
            # rec_loss_diff = l2_loss(
            #     rend_diff_in.contiguous(), rend_diff_rec.contiguous()
            # )
            # rec_loss_spec = l2_loss(
            #     rend_spec_in.contiguous(), rend_spec_rec.contiguous()
            # )
            # rec_loss = (rec_loss_diff + rec_loss_spec) / 2
            rec_loss = l2_loss(rend_in.contiguous(), rend_rec.contiguous())

            loss_out += rec_loss # Sy: 위에서 다 합해진 loss_out에 render loss 더함
            log_out[f'{kwargs["split"]}/total_loss'] += rec_loss.detach().mean() # Sy: kwargs["split"] = 'train' or 'val'
            log_out[f'{kwargs["split"]}/rend_loss'] = rec_loss.detach().mean()

        return loss_out, log_out
