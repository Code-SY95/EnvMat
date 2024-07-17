from pathlib import Path 
import shutil

root = Path('/mnt/1TB/MatGen/output')

# Ensure the destination directory exists
destination_dir = Path("/mnt/1TB/MatGen/output/render_images")
destination_dir.mkdir(parents=True, exist_ok=True)

i = 0
for render in root.glob("*/*/render.png"):
    # Generate a unique filename to avoid overwriting
    destination_path = destination_dir / f"render_{i}.png"
    shutil.copy(render, destination_path)
    i += 1

print(f"Finish {i}")