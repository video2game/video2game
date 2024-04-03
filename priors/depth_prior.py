import torch
import torchvision.transforms.functional as F
from torchvision import transforms

import PIL
from PIL import Image
import numpy as np

import argparse
import os
from tqdm import tqdm
from modules.midas.dpt_depth import DPTDepthModel
import cv2

parser = argparse.ArgumentParser()

parser.add_argument('--source_dir', help="path to rgb image", required=True)
parser.add_argument('--output_dir', help="path to where output should be stored", required=True)
parser.add_argument('--vis_dir', help="path to where output visualization should be stored", default=None)
parser.add_argument('--ckpt_dir', help="path to where omnidata dpt models are stored", required=True)

args = parser.parse_args()

trans_topil = transforms.ToPILImage()
map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device('cpu')
print(f"map_location: {'cuda' if torch.cuda.is_available() else 'cpu'}")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

pretrained_weights_path = os.path.join(args.ckpt_dir, 'omnidata_dpt_depth_v2.ckpt')  # 'omnidata_dpt_depth_v1.ckpt'
model = DPTDepthModel(backbone='vitb_rn50_384') # DPT Hybrid
checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
if 'state_dict' in checkpoint:
    state_dict = {}
    for k, v in checkpoint['state_dict'].items():
        state_dict[k[6:]] = v
else:
    state_dict = checkpoint
model.load_state_dict(state_dict)
model.to(device)


def save_outputs(img_path, save_path, vis_path=None):
    with torch.no_grad():

        def process_patch(img_slice):
            img_patch = Image.fromarray(img_slice)
            _h, _w = img_slice.shape[0], img_slice.shape[1]
            if max(_h, _w) > 2048:
                ratio = max(_h, _w) / 2048
                _h = int(_h / ratio)
                _w = int(_w / ratio)
            _h_r = int((_h // 64 + 1) * 64) if _h % 64 != 0 else _h
            _w_r = int((_w // 64 + 1) * 64) if _w % 64 != 0 else _w
            trans_totensor = transforms.Compose([transforms.Resize((_h_r, _w_r), interpolation=PIL.Image.BILINEAR),
                                                 transforms.CenterCrop((_h_r, _w_r)),
                                                 transforms.ToTensor()])

            img_tensor = trans_totensor(img_patch)[:3].unsqueeze(0).to(device)
            if img_tensor.shape[1] == 1:
                img_tensor = img_tensor.repeat_interleave(3, 1)
            img_tensor = img_tensor / 255.
            output = model(img_tensor).clamp(min=0, max=1)
            output = F.resize(output, (img_slice.shape[0], img_slice.shape[1]))
            return output

        img = np.array(Image.open(img_path))
        h, w, c = np.array(img).shape

        output = process_patch(img[:, :])

        output = output.clamp(0, 1).reshape(h, w)
        output = np.array(output.cpu(), dtype=np.float32)
        output = np.clip(output, 0, 1)
        np.save(save_path, output)

        # visualize
        if vis_path is not None:
            output = output * 10
            output = np.clip(output, 0, 1)
            output = output * 255
            output = output.astype(np.uint8)
            output = cv2.applyColorMap(output, cv2.COLORMAP_PLASMA)
            cv2.imwrite(vis_path, output)


names = [png_name for png_name in sorted(os.listdir(args.source_dir)) if os.path.splitext(png_name)[1] in ['.JPG', '.png', '.jpg', '.jpeg']]
source_paths = [os.path.join(args.source_dir, name) for name in names]
os.makedirs(args.output_dir, exist_ok=True)
output_paths = [os.path.join(args.output_dir, os.path.splitext(name)[0]+'.npy') for name in names]
if args.vis_dir is not None:
    os.makedirs(args.vis_dir, exist_ok=True)
    vis_paths = [os.path.join(args.vis_dir, name) for name in names]
n_path = len(output_paths)

for i in tqdm(range(n_path)):
    if args.vis_dir is not None:
        save_outputs(source_paths[i], output_paths[i], vis_paths[i])
    else:
        save_outputs(source_paths[i], output_paths[i])