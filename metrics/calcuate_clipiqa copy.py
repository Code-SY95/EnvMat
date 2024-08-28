import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def calculate_clip_iqa(image_paths, descriptions):
    images = [Image.open(image_path) for image_path in image_paths]
    inputs = processor(text=descriptions, images=images, return_tensors="pt", padding=True)
    
    # Calculate features
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract image and text embeddings
    image_embeddings = outputs.image_embeds
    text_embeddings = outputs.text_embeds
    
    # Normalize the embeddings
    image_embeddings = image_embeddings / image_embeddings.norm(p=2, dim=-1, keepdim=True)
    text_embeddings = text_embeddings / text_embeddings.norm(p=2, dim=-1, keepdim=True)
    
    # Compute cosine similarity between image and text embeddings
    similarity = torch.matmul(image_embeddings, text_embeddings.T)
    
    # Calculate IQA score (diagonal elements of the similarity matrix)
    iqa_scores = similarity.diag().cpu().numpy()
    
    return iqa_scores

# Example usage
image_paths = ['path_to_image1.jpg', 'path_to_image2.jpg']
descriptions = ["description of image 1", "description of image 2"]

clip_iqa_scores = calculate_clip_iqa(image_paths, descriptions)
for i, score in enumerate(clip_iqa_scores):
    print(f'CLIP-IQA score for image {i+1}: {score}')
