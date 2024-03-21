import numpy as np
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

parser = argparse.ArgumentParser()

parser.add_argument('--source_dir', help="path to rgb image", required=True)
parser.add_argument('--output_dir', help="path to where output image should be stored", required=True)
parser.add_argument('--ckpt_dir', help="path to where omnidata dpt models are stored", required=True)
parser.add_argument('--slice', help="patch number per axis", default=1, type=int)

args = parser.parse_args()

trans_topil = transforms.ToPILImage()
map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device('cpu')
print(f"map_location: {'cuda' if torch.cuda.is_available() else 'cpu'}")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

pretrained_weights_path = os.path.join(args.ckpt_dir, 'omnidata_dpt_normal_v2.ckpt')
model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=3) # DPT Hybrid
checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
if 'state_dict' in checkpoint:
    state_dict = {}
    for k, v in checkpoint['state_dict'].items():
        state_dict[k[6:]] = v
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)
model.to(device)

def save_outputs(img_path, save_path):
    with torch.no_grad():

        def process_patch(img_slice):
            img_patch = Image.fromarray(img_slice)
            _h, _w = img_slice.shape[0], img_slice.shape[1]
            if max(_h, _w) > 1024:
                ratio = max(_h, _w) / 1024
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
        if args.slice > 1:
            hs = np.linspace(0, int(0.4 * h), args.slice).astype(np.int32)
            he = np.linspace(int(0.6 * h), h, args.slice).astype(np.int32)
            ws = np.linspace(0, int(0.4 * w), args.slice).astype(np.int32)
            we = np.linspace(int(0.6 * w), w, args.slice).astype(np.int32)

            weights = torch.ones((h, w)).cuda() * 1e-6
            normals = torch.zeros((h, w, 3)).cuda()

            for _hs, _he in zip(hs, he):
                for _ws, _we in zip(ws, we):
                    slice_output = process_patch(img[_hs:_he, _ws:_we]).squeeze(0).permute(1, 2, 0)
                    normals[_hs:_he, _ws:_we] += slice_output
                    weights[_hs:_he, _ws:_we] += 1

            normals = normals / weights.unsqueeze(-1)
            output = normals.permute(2, 0, 1).unsqueeze(0)
        else:
            output = process_patch(img[:, :])

        trans_topil(output.squeeze(0)).save(save_path)


names = [png_name for png_name in sorted(os.listdir(args.source_dir)) if png_name[-4:] in ['.JPG', '.png', '.jpg', 'jpeg']]
source_paths = [os.path.join(args.source_dir, name) for name in names]
os.makedirs(args.output_dir, exist_ok=True)
output_paths = [os.path.join(args.output_dir, name) for name in names]
n_path = len(output_paths)

for i in tqdm(range(n_path)):
    save_outputs(source_paths[i], output_paths[i])