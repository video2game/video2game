import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def extract_model_state_dict(ckpt_path, model_name='model', prefixes_to_ignore=[]):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    checkpoint_ = {}
    if 'state_dict' in checkpoint: # if it's a pytorch-lightning checkpoint
        checkpoint = checkpoint['state_dict']
    for k, v in checkpoint.items():
        if not k.startswith(model_name):
            continue
        k = k[len(model_name)+1:]
        for prefix in prefixes_to_ignore:
            if k.startswith(prefix):
                break
        else:
            checkpoint_[k] = v
    return checkpoint_


def load_ckpt(model, ckpt_path, model_name='model', prefixes_to_ignore=[]):
    if not ckpt_path: return
    model_dict = model.state_dict()
    checkpoint_ = extract_model_state_dict(ckpt_path, model_name, prefixes_to_ignore)
    print("checkpoint_.keys(): ", checkpoint_.keys())
    model_dict.update(checkpoint_)
    keys = model.load_state_dict(model_dict)
    print("loading: ", keys)


def slim_ckpt(ckpt_path, save_poses=False):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    # pop unused parameters
    keys_to_pop = ['directions', 'model.density_grid', 'model.grid_coords']
    if not save_poses: keys_to_pop += ['poses']
    for k in ckpt['state_dict']:
        if k.startswith('val_lpips'):
            keys_to_pop += [k]
    for k in keys_to_pop:
        ckpt['state_dict'].pop(k, None)
    return ckpt['state_dict']

def box_filter(image, r):
    '''
    Input
        image: (h, w)
        r: constant
    Return
        image: (h, w)
    '''
    image = image[None][None] #(1, 1, h, w)
    device = image.device
    filters = torch.ones(1, 1, 2*r+1, 2*r+1, device=device) / ((2*r+1)**2)
    image_pad = F.pad(image, (r, r, r, r), mode='reflect')
    image_out = F.conv2d(image_pad, filters)
    image_out = image_out[0, 0] #(h, w)
    return image_out

def guided_filter(image_p, image_i, r, eps=0.1):
    '''
    Input:
        image_p: input (h, w)
        image_i: guided (h, w)
        r: radius of filter window
        eps: regularization weight, higher->smooth
    '''
    mean_p  = box_filter(image_p, r)
    mean_i  = box_filter(image_i, r)
    corr_ip = box_filter(image_i*image_p, r)
    corr_ii = box_filter(image_i*image_i, r)

    var_i = corr_ii - mean_i * mean_i 
    cov_ip = corr_ip - mean_i * mean_p 
    a = cov_ip / (var_i + eps**2)
    b = mean_p - a * mean_i
    
    mean_a = box_filter(a, r)
    mean_b = box_filter(b, r)

    image_out = mean_a * image_i + mean_b
    return image_out

def save_image(image, path):
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image)
    image.save(path)

def test():
    from matplotlib import pyplot as plt
    def show(img):
        img = img.numpy()
        plt.axis('off')
        plt.imshow(img)
        plt.show()
        plt.close()

    depth = np.load('results/tnt/playground/depth_raw.npy')
    depth = torch.tensor(depth[0])/16
    # show(depth)
    # depth_0 = guided_filter(depth, depth, r=5, eps=0.01)
    # show(depth_0)
    depth_1 = guided_filter(depth, depth, r=2, eps=0.01)
    show(depth_1)

if __name__ == '__main__':
    test()