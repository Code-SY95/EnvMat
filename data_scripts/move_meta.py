import os
import shutil
from pathlib import Path
from tqdm import tqdm

source_dir = Path('/home/sogang/mnt/db_2/oh/MatSynth/pbr256/train/')
dest_dir = Path('/home/sogang/mnt/db_2/oh/EnvMat/data/train/')

for file in tqdm([x for x in source_dir.glob("*/*/metadata.json")]):
    mat_dir = file.parent
    mat = mat_dir.stem # material
    cat = file.parent.parent.stem # category

    src_file = os.path.join(mat_dir, 'metadata.json')
    dest_file = os.path.join(dest_dir/cat/mat, 'metadata.json') 
    shutil.copy(src_file, dest_file)
