import torchvision.transforms.functional as TF
from datasets import load_dataset
from torch.utils.data import DataLoader

# image processing function
def process_img(x):
    x = TF.resize(x, (1024, 1024))
    x = TF.to_tensor(x)
    return x

# item processing function
def process_batch(examples):
    examples["basecolor"] = [process_img(x) for x in examples["basecolor"]]
    return examples

# load the dataset in streaming mode
ds = load_dataset(
    "gvecchio/MatSynth", 
    streaming = True,
)

# remove unwanted columns
ds = ds.remove_columns(["diffuse", "specular", "displacement", "opacity", "blend_mask"])
# or keep only specified columns
ds = ds.select_columns(["metadata", "basecolor"])

# shuffle data
ds = ds.shuffle(buffer_size=100)

# filter data matching a specific criteria, e.g.: only CC0 materials
ds = ds.filter(lambda x: x["metadata"]["license"] == "CC0")
# filter out data from Deschaintre et al. 2018
ds = ds.filter(lambda x: x["metadata"]["source"] != "deschaintre_2020")

# Set up processing
ds = ds.map(process_batch, batched=True, batch_size=8)

# set format for usage in torch
ds = ds.with_format("torch")

# iterate over the dataset
for x in ds:
    print(x)
    
    data_loader = DataLoader(ds[x], batch_size=1)
    for batch in data_loader:
        print(batch.item())
