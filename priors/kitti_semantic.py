import os
from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
from mmseg.core.evaluation import get_palette
from mmcv.runner import load_checkpoint
from mmseg.core import get_classes
import numpy as np
from tqdm import tqdm

config_file = '_configs/segformer/segformer_mit-b5_8x1_1024x1024_160k_cityscapes.py'
checkpoint_file = 'ckpts/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth'
palette = 'cityscapes'

model = init_segmentor(config_file, checkpoint=None, device='cuda:0')
checkpoint = load_checkpoint(model, checkpoint_file, map_location='cpu')
if 'CLASSES' in checkpoint.get('meta', {}):
    model.CLASSES = checkpoint['meta']['CLASSES']
else:
    model.CLASSES = get_classes(palette)
print("model.CLASSES: ", model.CLASSES)

root_dir = '/home/hongchix/main/root/datasets/KITTI-360/'
seq_id = 0
dir_seq = '2013_05_28_drive_{:0>4d}_sync'.format(seq_id)
start = 7606
end = 7800
dir_rgb_0 = os.path.join(root_dir, 'data_2d_raw', dir_seq, 'image_00', 'data_rect')
dir_rgb_1 = os.path.join(root_dir, 'data_2d_raw', dir_seq, 'image_01', 'data_rect')
dir_sem_0 = os.path.join(root_dir, 'data_2d_semantics_sup/train', dir_seq, 'image_00/semantic')
dir_sem_1 = os.path.join(root_dir, 'data_2d_semantics_sup/train', dir_seq, 'image_01/semantic')

os.makedirs(dir_sem_0, exist_ok=True)
for file in tqdm(sorted(os.listdir(dir_rgb_0))):
    if file[-4:] == '.png' and int(file[:-4]) >= start and int(file[:-4]) <= end:
        img = os.path.join(dir_rgb_0, file) # or img = mmcv.imread(img), which will only load it once
        result = inference_segmentor(model, img)
        model.save_pgm(result, os.path.join(dir_sem_0, file.replace('.png', '.pgm'))) # save as .pgm file

os.makedirs(dir_sem_1, exist_ok=True)
for file in tqdm(sorted(os.listdir(dir_rgb_1))):
    if file[-4:] == '.png' and int(file[:-4]) >= start and int(file[:-4]) <= end:
        img = os.path.join(dir_rgb_1, file) # or img = mmcv.imread(img), which will only load it once
        result = inference_segmentor(model, img)
        model.save_pgm(result, os.path.join(dir_sem_1, file.replace('.png', '.pgm'))) # save as .pgm file