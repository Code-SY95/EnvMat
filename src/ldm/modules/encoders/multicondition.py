import torch
import torch.nn as nn

from ldm.util import instantiate_from_config


class ZeroEmbedder(nn.Module):
    """
    Dummy embedder that returns a zero tensor, to exclude conditions without breaking the code
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.zeros_like(x)


class PaletteEncoder(nn.Module):
    def __init__(self, in_ch, hid_ch, out_ch, n_colors=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_ch, hid_ch),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(hid_ch * n_colors, out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class MultiConditionEncoder(nn.Module):
    def __init__(
        self,
        image_embed_config=None,
        text_embed_config=None,
        binary_encoder_config=None,
        palette_proj_config=None,
    ):
        super().__init__()
        print(f"Conditional model: Multiconditional")
        self.image_embed = (
            instantiate_from_config(image_embed_config)
            if image_embed_config
            else ZeroEmbedder()
        )
        self.text_embed = (
            instantiate_from_config(text_embed_config)
            if text_embed_config
            else ZeroEmbedder()
        )
        # Sy: Remove
        # self.sketch_encoder = (
        #     instantiate_from_config(binary_encoder_config)
        #     if binary_encoder_config
        #     else ZeroEmbedder()
        # )
        # self.palette_encoder = (
        #     instantiate_from_config(palette_proj_config)
        #     if palette_proj_config
        #     else ZeroEmbedder()
        # )

        self.encoders = {
            "image_embed": self.image_embed,
            "text": self.text_embed,
            # "sketch": self.sketch_encoder,
            # "palette": self.palette_encoder, # Sy: Remove
        }

        self.keys = [
            k
            for k in self.encoders.keys()
            if not isinstance(self.encoders[k], ZeroEmbedder)
        ]

    def parameters(self, trainable_only=False):
        for param in self._get_params(trainable_only):
            yield param

    def _get_params(self, trainable_only=False):
        # Sy: Remove sketch & palette
        # params = (
        #     list(self.sketch_encoder.parameters())
        #     + list(self.palette_encoder.parameters())
        # )
        if not trainable_only:
            # params += list(self.text_embed.parameters())
            params = list(self.text_embed.parameters()) # Sy: 위에서 params 정의한걸 지웠으니 여기서 다시 수정.
            params += list(self.image_embed.parameters())
        return params

    def forward(self, x):
        with torch.no_grad():
            image_embed = self.image_embed(x["image_embed"]) # Sy: image_embed = [b, 1, 512] 
            text_embed = self.text_embed(x["text"]).unsqueeze(1) # Sy: text_embed = [b, 1, 512] # Sy: 여기서 torch.Size 쭈루룩
        # sketch = self.sketch_encoder(x["sketch"]) # Sy: Remove sketch & palette
        # palette = self.palette_encoder(x["palette"]).unsqueeze(1)

        # c_local = sketch
        # c_global = torch.cat([image_embed, text_embed, palette], dim=1) # Sy: Remove sketch & palette condition
        c_global = torch.cat([image_embed, text_embed], dim=1) # Sy: c_global = [b, 2, 512]

        # return {"c_crossattn": c_global, "c_concat": c_local}
        return {"c_crossattn": c_global} # Sy: global condition(text, image) of model
