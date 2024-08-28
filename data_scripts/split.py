import os
import shutil
from pathlib import Path
from tqdm import tqdm
import random

source_dir = Path('/home/sogang/mnt/db_2/oh/EnvMat/data/train')
dest_dir = Path('/home/sogang/mnt/db_2/oh/EnvMat/data/validation')

cate_list = ['Ceramic', 'Concrete', 'Fabric', 'Ground', 'Leather', 'Marble', 'Metal', 'Misc', 'Plaster', 'Plastic', 'Stone', 'Terracotta', 'Wood']

# print(len([x for x in source_dir.glob('Ceramic/*')]))

i = 0

for cate in cate_list:
    # src = source_dir/cate
    # if cate == 'Metal':
    #     move = 8
    # elif cate == 'Stone':
    #     move = 8
    # elif cate == 'Wood':
    #     move = 9
    # else:
    #     print(f'The number of materials of {cate} is {len([x for x in (dest_dir/cate).glob("*")])}')
    #     continue
    
    for folder in random.sample([x for x in src.glob("*")], move):
    # for folder in [x for x in src.glob("*/*/basecolor.png")]:
        # mat_dir = folder.parent.parent
        mat = folder.stem # material
        # cat = folder.parent.parent.parent.stem # category

        # src_file = os.path.join(mat_dir, 'metadata.json')
        dest = dest_dir/cate/mat
        os.makedirs(os.path.dirname(dest), exist_ok=True) 
        shutil.move(folder, dest)
        i += 1
    print(f'The number of materials of {cate} is {len([x for x in (dest_dir/cate).glob("*")])}')
print(i)