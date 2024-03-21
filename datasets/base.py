from torch.utils.data import Dataset
import numpy as np
import torch

class BaseDataset(Dataset):
    """
    Define length and sampling method
    """
    def __init__(self, root_dir, split='train', downsample=1.0):
        self.root_dir = root_dir
        self.split = split
        self.downsample = downsample
        # reset mark for ray sampling strategy 'epoch_like'
        self.epoch_like_sampling_reset_mark = False
        # epoch length of ray sampling strategy 'single_image'
        self.epoch_length_baking = 100
        
    def read_intrinsics(self):
        raise NotImplementedError

    def epoch_like_sampling_reset(self):
        img_num = self.img_num = len(self.poses)
        pix_num = self.pix_num = self.img_wh[0]*self.img_wh[1]
        print("epoch_like_sampling_reset: ", "img_num: ", img_num, " pix_num: ", pix_num)
        total_samples = img_num * pix_num
        self.sample_iter = 0
        self.sample_total_iter = total_samples // self.batch_size + 1
        print("self.sample_total_iter: ", self.sample_total_iter)
        self.sample_seq = np.arange(self.sample_total_iter * self.batch_size)
        np.random.shuffle(self.sample_seq)
        self.epoch_like_sampling_reset_mark = True
        
    def epoch_like_sampling(self):
        if not self.epoch_like_sampling_reset_mark:
            self.epoch_like_sampling_reset()
            
        sample_start = self.sample_iter * self.batch_size
        sample_end = (self.sample_iter + 1) * self.batch_size
        samples = self.sample_seq[sample_start:sample_end]
        img_idx = (samples // self.pix_num) % self.img_num
        pix_idx = samples % self.pix_num
        self.sample_iter += 1
        if self.sample_iter == self.sample_total_iter:
            self.epoch_like_sampling_reset()
        return img_idx, pix_idx
        
    def __len__(self):
        if self.split.startswith('train'):
            if self.ray_sampling_strategy == 'single_image':
                return self.epoch_length_baking
            return 1000
        return len(self.poses)

    def __getitem__(self, idx):
        """
            ray_sampling_strategy:
                1. NeRF training
                - all_images: randomly sample from all images
                - same_image: randomly sample from only one image
                - epoch_like: randomly sample but iterate over all pixels in one epoch
                2. NeRF baking
                - single_image: sample one image once 
        """
        if self.split.startswith('train'):
            # if True:
            if self.ray_sampling_strategy == 'single_image':
                indices = np.random.choice(self.poses.shape[0], self.batch_size, replace=False)
                samples = []
                for idx in indices:
                    index = idx
                    H = self.img_wh[1]
                    W = self.img_wh[0]
                    rays = self.rays[index]
                    
                    if not hasattr(self, 'masks') or self.masks is None:
                        masks = None
                    else:
                        masks = torch.logical_not(self.masks[index].reshape(H*W, 1))
                    sample = {
                        'H': self.img_wh[1], 
                        'W': self.img_wh[0],
                        'rays': rays,
                        'index': index,
                        'mask': masks
                    }
                    samples.append(sample)
                return samples
             
            if (not hasattr(self, 'masks')) or self.masks is None :
                # training pose is retrieved in train.py
                if self.ray_sampling_strategy == 'all_images': # randomly select images
                    img_idxs = np.random.choice(len(self.poses), self.batch_size)
                    # randomly select pixels
                    pix_idxs = np.random.choice(self.img_wh[0]*self.img_wh[1], self.batch_size)
                elif self.ray_sampling_strategy == 'same_image': # randomly select ONE image
                    img_idxs = np.random.choice(len(self.poses), 1)[0]
                    img_idxs = (np.ones(self.batch_size) * img_idxs).astype(np.int64)
                    # randomly select pixels
                    pix_idxs = np.random.choice(self.img_wh[0]*self.img_wh[1], self.batch_size)
                elif self.ray_sampling_strategy == "epoch_like":
                    img_idxs, pix_idxs = self.epoch_like_sampling()
                
                        
                rays = self.rays[img_idxs, pix_idxs]
                
            else:
                
                assert self.ray_sampling_strategy == 'all_images'
                n_samples = 0
                rays = None
                with torch.no_grad():
                    while n_samples < self.batch_size:
                        _img_idxs = np.random.choice(len(self.poses), self.batch_size)
                        _pix_idxs = np.random.choice(self.img_wh[0]*self.img_wh[1], self.batch_size)
                        mask = torch.logical_not(self.masks[_img_idxs, _pix_idxs])
                        _rays = self.rays[_img_idxs, _pix_idxs][mask] # true means maintain
                        _img_idxs = _img_idxs[mask]
                        _pix_idxs = _pix_idxs[mask]
                        n_samples_before = n_samples
                        n_samples += _rays.shape[0]
                        if n_samples > self.batch_size:
                            _rays = _rays[:self.batch_size-n_samples_before]
                            _img_idxs = _img_idxs[:self.batch_size-n_samples_before]
                            _pix_idxs = _pix_idxs[:self.batch_size-n_samples_before]
                        
                        if rays is None:
                            rays = _rays
                            img_idxs = _img_idxs
                            pix_idxs = _pix_idxs
                        else:
                            rays = torch.cat((rays, _rays), dim=0)
                            img_idxs = np.concatenate((img_idxs, _img_idxs), axis=0)
                            pix_idxs = np.concatenate((pix_idxs, _pix_idxs), axis=0)
                    assert rays.shape[0] == self.batch_size
                    assert img_idxs.shape[0] == self.batch_size
                    assert pix_idxs.shape[0] == self.batch_size
              
            if hasattr(self, 'img_wh'):
                w, h = self.img_wh
                u = pix_idxs // w
                v = pix_idxs % w
                uv = np.stack([u, v], axis=-1)
            sample = {'img_idxs': img_idxs, 'pix_idxs': pix_idxs, 'uv': uv,
                    'rgb': rays[:, :3]}
            if hasattr(self, 'labels'):
                labels = self.labels[img_idxs, pix_idxs]
                sample['label'] = labels
            if hasattr(self, 'depths'):
                depth = self.depths[img_idxs, pix_idxs]
                sample['depth'] = depth
            if hasattr(self, 'normals'):
                normal = self.normals[img_idxs, pix_idxs]
                sample['normal'] = normal
            if self.rays.shape[-1] == 4: # HDR-NeRF data
                sample['exposure'] = rays[:, 3:]
            if hasattr(self, 'lidarp'):
                lidarp = self.lidarp[img_idxs, pix_idxs]
                sample['lidarp'] = lidarp
                lidarp_mask = self.lidarp_mask[img_idxs, pix_idxs]
                sample['lidarp_mask'] = lidarp_mask
            if hasattr(self, 'depth_dp'):
                depth_dp = self.depth_dp[img_idxs, pix_idxs]
                sample['depth_dp'] = depth_dp

        else:
            sample = {'pose': self.poses[idx], 'img_idxs': idx}
            if len(self.rays)>0: # if ground truth available
                rays = self.rays[idx]
                sample['rgb'] = rays[:, :3]
                if hasattr(self, 'labels'):
                    labels = self.labels[idx]
                    sample['label'] = labels

        return sample
    
    def get_image_with_idx(self, idx):
        sample = {'pose': self.poses[idx], 'img_idxs': idx}
        
        return sample
        