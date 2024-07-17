from torchmetrics.multimodal import CLIPImageQualityAssessment
import torch

_ = torch.manual_seed(42)

imgs = torch.randint(255, (2, 3, 256, 256)).float()

metric = CLIPImageQualityAssessment(prompts=(("high quality.", "low quality."),))
# metric = CLIPImageQualityAssessment(prompts=("quality",))

result = metric(imgs)
print(result)