from .colmap import ColmapDataset
from .kitti360 import KittiDataset
from .scannetpp import ScannetDataset
from .scannetpp_dslr import ScannetDSLRDataset
from .nerfstudio import NerfstudioDataset
from .vr_nerf import VRNeRFDataset

dataset_dict = {
                'colmap': ColmapDataset,
                'kitti': KittiDataset,
                'scannetpp': ScannetDataset,
                'scannetpp_dslr': ScannetDSLRDataset,
                'nerfstudio': NerfstudioDataset,
                'vr_nerf': VRNeRFDataset,
}
