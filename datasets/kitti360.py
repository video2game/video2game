import cv2
from PIL import Image
try:
    from .ray_utils import *
except:
    from ray_utils import *
try:
    from .base import BaseDataset
except:
    from base import BaseDataset
import os 
import xml.etree.ElementTree as ET
import pickle5 as pickle
import imageio
from collections import namedtuple
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'kittiId'     , # An integer ID that is associated with this label for KITTI-360
                    # NOT FOR RELEASING

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'ignoreInInst', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations of instance segmentation or not

    'color'       , # The color of this label
    ] )
labels = [
    #       name                     id    kittiId,    trainId   category            catId     hasInstances   ignoreInEval   ignoreInInst   color
    Label(  'unlabeled'            ,  0 ,       -1 ,       255 , 'void'            , 0       , False        , True         , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,       -1 ,       255 , 'void'            , 0       , False        , True         , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,       -1 ,       255 , 'void'            , 0       , False        , True         , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,       -1 ,       255 , 'void'            , 0       , False        , True         , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,       -1 ,       255 , 'void'            , 0       , False        , True         , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,       -1 ,       255 , 'void'            , 0       , False        , True         , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,       -1 ,       255 , 'void'            , 0       , False        , True         , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        1 ,         0 , 'flat'            , 1       , False        , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        3 ,         1 , 'flat'            , 1       , False        , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,        2 ,       255 , 'flat'            , 1       , False        , True         , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,        10,       255 , 'flat'            , 1       , False        , True         , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        11,         2 , 'construction'    , 2       , True         , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        7 ,         3 , 'construction'    , 2       , False        , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        8 ,         4 , 'construction'    , 2       , False        , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,        30,       255 , 'construction'    , 2       , False        , True         , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,        31,       255 , 'construction'    , 2       , False        , True         , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,        32,       255 , 'construction'    , 2       , False        , True         , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        21,         5 , 'object'          , 3       , True         , False        , True         , (153,153,153) ),
    Label(  'polegroup'            , 18 ,       -1 ,       255 , 'object'          , 3       , False        , True         , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        23,         6 , 'object'          , 3       , True         , False        , True         , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        24,         7 , 'object'          , 3       , True         , False        , True         , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        5 ,         8 , 'nature'          , 4       , False        , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        4 ,         9 , 'nature'          , 4       , False        , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,        9 ,        10 , 'sky'             , 5       , False        , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,        19,        11 , 'human'           , 6       , True         , False        , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,        20,        12 , 'human'           , 6       , True         , False        , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,        13,        13 , 'vehicle'         , 7       , True         , False        , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,        14,        14 , 'vehicle'         , 7       , True         , False        , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,        34,        15 , 'vehicle'         , 7       , True         , False        , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,        16,       255 , 'vehicle'         , 7       , True         , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,        15,       255 , 'vehicle'         , 7       , True         , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,        33,        16 , 'vehicle'         , 7       , True         , False        , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,        17,        17 , 'vehicle'         , 7       , True         , False        , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,        18,        18 , 'vehicle'         , 7       , True         , False        , False        , (119, 11, 32) ),
    Label(  'garage'               , 34 ,        12,         2 , 'construction'    , 2       , True         , True         , True         , ( 64,128,128) ),
    Label(  'gate'                 , 35 ,        6 ,         4 , 'construction'    , 2       , False        , True         , True         , (190,153,153) ),
    Label(  'stop'                 , 36 ,        29,       255 , 'construction'    , 2       , True         , True         , True         , (150,120, 90) ),
    Label(  'smallpole'            , 37 ,        22,         5 , 'object'          , 3       , True         , True         , True         , (153,153,153) ),
    Label(  'lamp'                 , 38 ,        25,       255 , 'object'          , 3       , True         , True         , True         , (0,   64, 64) ),
    Label(  'trash bin'            , 39 ,        26,       255 , 'object'          , 3       , True         , True         , True         , (0,  128,192) ),
    Label(  'vending machine'      , 40 ,        27,       255 , 'object'          , 3       , True         , True         , True         , (128, 64,  0) ),
    Label(  'box'                  , 41 ,        28,       255 , 'object'          , 3       , True         , True         , True         , (64,  64,128) ),
    Label(  'unknown construction' , 42 ,        35,       255 , 'void'            , 0       , False        , True         , True         , (102,  0,  0) ),
    Label(  'unknown vehicle'      , 43 ,        36,       255 , 'void'            , 0       , False        , True         , True         , ( 51,  0, 51) ),
    Label(  'unknown object'       , 44 ,        37,       255 , 'void'            , 0       , False        , True         , True         , ( 32, 32, 32) ),
    Label(  'license plate'        , -1 ,        -1,        -1 , 'vehicle'         , 7       , False        , True         , True         , (  0,  0,142) ),
]

#--------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
#--------------------------------------------------------------------------------

# Please refer to the main method below for example usages!

# name to label object
name2label      = { label.name    : label for label in labels           }
# id to label object
id2label        = { label.id      : label for label in labels           }
# trainId to label object
trainId2label   = { label.trainId : label for label in reversed(labels) }
# KITTI-360 ID to cityscapes ID
kittiId2label   = { label.kittiId : label for label in labels           }

class KITTI360Bbox3D(object):
    # Constructor
    def __init__(self):
        # the polygon as list of points
        self.vertices  = []
        self.faces  = []


        # projected vertices
        self.vertices_proj = None
        self.meshes = []

        # name
        self.name = ''
        self.category = ''

    def __str__(self): 
        return self.name

                
    def parseOpencvMatrix(self, node):
        rows = int(node.find('rows').text)
        cols = int(node.find('cols').text)
        data = node.find('data').text.split(' ')
    
        mat = []
        for d in data:
            d = d.replace('\n', '')
            if len(d)<1:
                continue
            mat.append(float(d))
        mat = np.reshape(mat, [rows, cols])
        return mat

    def parseVertices(self, child):
        transform = self.parseOpencvMatrix(child.find('transform'))
        R = transform[:3,:3]
        T = transform[:3,3]
        vertices = self.parseOpencvMatrix(child.find('vertices'))
        faces = self.parseOpencvMatrix(child.find('faces'))
        self.R = R.copy()
        self.T = T.copy()
        self.original_vertices = vertices.copy()
        vertices = np.matmul(R, vertices.transpose()).transpose() + T
        self.vertices = vertices
        self.faces = faces
        self.R = R
        self.T = T
        
        self.x_min = np.min(vertices[:, 0])
        self.y_min = np.min(vertices[:, 1])
        self.z_min = np.min(vertices[:, 2])
        
        self.x_max = np.max(vertices[:, 0])
        self.y_max = np.max(vertices[:, 1])
        self.z_max = np.max(vertices[:, 2])

    def parseBbox(self, child):
        # semanticIdKITTI = int(child.find('semanticId').text)
        # self.semanticId = kittiId2label[semanticIdKITTI].id
        # self.instanceId = int(child.find('instanceId').text)
        # self.name = kittiId2label[semanticIdKITTI].name
        self.parseVertices(child)
        
    def parseStuff(self, child):
        classmap = {'driveway': 'parking', 'ground': 'terrain', 'unknownGround': 'ground', 
                    'railtrack': 'rail track', 'bigPole': 'pole', 'unknownObject': 'unknown object',
                    'smallPole': 'smallpole', 'trafficSign': 'traffic sign', 'trashbin': 'trash bin',
                    'guardrail': 'guard rail', 'trafficLight': 'traffic light', 'pedestrian': 'person',
                    'vendingmachine': 'vending machine', 'unknownConstruction': 'unknown construction',
                    'unknownVehicle': 'unknown vehicle'}
        label = child.find('label').text 
        if label in classmap.keys():
            label = classmap[label]
        self.name = label
        for _label in labels:
            if self.name == _label.name:
                self.category = _label.category
        self.timestamp = int(child.find('timestamp').text)
        self.parseVertices(child)
        
    def overlap_bbox(self, scene_bbox_glb_x_max, scene_bbox_glb_x_min,
                            scene_bbox_glb_y_max, scene_bbox_glb_y_min,
                            scene_bbox_glb_z_max, scene_bbox_glb_z_min):
        x_overlap = not(self.x_min > scene_bbox_glb_x_max or self.x_max < scene_bbox_glb_x_min)
        y_overlap = not(self.y_min > scene_bbox_glb_y_max or self.y_max < scene_bbox_glb_y_min)
        z_overlap = not(self.z_min > scene_bbox_glb_z_max or self.z_max < scene_bbox_glb_z_min)

        return x_overlap and y_overlap and z_overlap
    
    def in_bbox(self, vertices):
        x_in = np.logical_and(vertices[:, 0] < self.x_max, vertices[:, 0] > self.x_min)
        y_in = np.logical_and(vertices[:, 1] < self.y_max, vertices[:, 1] > self.y_min)
        z_in = np.logical_and(vertices[:, 2] < self.z_max, vertices[:, 2] > self.z_min)
        _in = np.logical_and(x_in, y_in)
        _in = np.logical_and(_in, z_in)
        
        return _in
   
        
def _load_3d_bboxes(bbox_path, seq):
    bboxes = []

    seq = '2013_05_28_drive_{:0>4d}_sync'.format(seq)
    with open(os.path.join(bbox_path, f"{seq}.xml"), "rb") as f:
        tree = ET.parse(f)
    root = tree.getroot()

    num_bbox = 0

    for child in root:
        if child.find('transform') is None:
            continue
        
        obj = KITTI360Bbox3D()
       
        if child.find("semanticId") is not None:
            obj.parseBbox(child)
        else:
            obj.parseStuff(child)
        if obj.timestamp == -1:
            if obj.category == "vehicle":
                bboxes.append(obj)
                num_bbox +=1
        

    return bboxes
    
class KittiDataset(BaseDataset):
    def __init__(self, root_dir, split, downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)
        # path and initialization
        self.root_dir = root_dir
        self.split = split
        seq_id = kwargs.get('seq_id', 0)
        dir_seq = '2013_05_28_drive_{:0>4d}_sync'.format(seq_id)
        
        # images
        dir_rgb_0 = os.path.join(root_dir, 'data_2d_raw', dir_seq, 'image_00', 'data_rect')
        dir_rgb_1 = os.path.join(root_dir, 'data_2d_raw', dir_seq, 'image_01', 'data_rect')
        
        # semantics of kitti360 labels
        dir_sem_0 = os.path.join(root_dir, 'data_2d_semantics/train', dir_seq, 'image_00/semantic')
        dir_sem_1 = os.path.join(root_dir, 'data_2d_semantics/train', dir_seq, 'image_01/semantic')
        
        # semantics generated by semantic segmentation models using pretrained models in cityscapes dataset
        dir_sem_0_sup = os.path.join(root_dir, 'data_2d_semantics_sup/train', dir_seq, 'image_00/semantic')
        dir_sem_1_sup = os.path.join(root_dir, 'data_2d_semantics_sup/train', dir_seq, 'image_01/semantic')
        
        # normal prior
        dir_normal_0 = os.path.join(root_dir, 'normal', dir_seq, 'image_00')
        dir_normal_1 = os.path.join(root_dir, 'normal', dir_seq, 'image_01')
        
        # depth prior
        dir_depth_0 = os.path.join(root_dir, 'depth', dir_seq, 'image_00')
        dir_depth_1 = os.path.join(root_dir, 'depth', dir_seq, 'image_01')
        
        # selective mask
        dir_mask_0 = os.path.join(root_dir, 'mask', dir_seq, 'image_01')
        dir_mask_1 = os.path.join(root_dir, 'mask', dir_seq, 'image_01')
        
        # lidar prior if necessary
        dir_lidarp_0 = os.path.join(root_dir, 'lidar_proxy', dir_seq, 'image_00')
        dir_lidarp_1 = os.path.join(root_dir, 'lidar_proxy', dir_seq, 'image_01')
        
        # depth prior provided by deepruner
        dir_depth_dp = os.path.join(root_dir, 'depth_dp', dir_seq)
        
        dir_calib = os.path.join(root_dir, 'calibration')
        dir_poses = os.path.join(root_dir, 'data_poses', dir_seq)
        
        # Intrinsics
        intrinsic_path = os.path.join(dir_calib, 'perspective.txt')
        K_00 = parse_calib_file(intrinsic_path, 'P_rect_00').reshape(3, 4)
        K_00[:2] *= downsample
        img_size = parse_calib_file(intrinsic_path, 'S_rect_00')
        self.K = K_00[:, :-1]
        w, h = int(img_size[0]), int(img_size[1])
        self.img_wh = (w, h)
        self.directions, self.grid = get_ray_directions(h, w, self.K, anti_aliasing_factor=kwargs.get('anti_aliasing_factor', 1.0), return_uv=True)

        # Extrinsics
        
        # support two kinds of kitti360 sequences:
        
        # two continuous sequences
        if kwargs.get('kitti_dual_seq', False):
            frame_start = kwargs.get('kitti_start', 0)
            frame_end   = kwargs.get('kitti_end', 100)
            
            pose_cam_0 = np.genfromtxt(os.path.join(dir_poses, 'cam0_to_world.txt')) #(n, 17)
            frame_id = pose_cam_0[:, 0]
            
            sample = np.logical_and(frame_id >= frame_start, frame_id <= frame_end)
            
            dual_frame_start = kwargs.get('kitti_dual_start', 0)
            dual_frame_end   = kwargs.get('kitti_dual_end', 100)
            dual_sample = np.logical_and(frame_id >= dual_frame_start, frame_id <= dual_frame_end)
            
            
            in_rgb_frame_id = np.array([int(png_name[:-4]) for png_name in os.listdir(dir_rgb_0)])
            in_rgb_frame_id_sample = np.array([f_id in in_rgb_frame_id for f_id in frame_id.tolist()], dtype=np.bool_).reshape(-1)
            sample = np.logical_and(sample, in_rgb_frame_id_sample)
            dual_sample = np.logical_and(dual_sample, in_rgb_frame_id_sample)
                
            sample = np.logical_or(sample, dual_sample)
            
            frame_id = frame_id[sample].astype(np.int32)
        # one continuous sequence
        else:
            frame_start = kwargs.get('kitti_start', 0)
            frame_end   = kwargs.get('kitti_end', 100)
            pose_cam_0 = np.genfromtxt(os.path.join(dir_poses, 'cam0_to_world.txt')) #(n, 17)
            frame_id = pose_cam_0[:, 0]
            sample = np.logical_and(frame_id >= frame_start, frame_id <= frame_end)
            
            in_rgb_frame_id = [int(png_name[:-4]) for png_name in os.listdir(dir_rgb_0)]
            in_rgb_frame_id_sample = np.array([f_id in in_rgb_frame_id for f_id in frame_id.tolist()], dtype=np.bool_).reshape(-1)
            sample = np.logical_and(sample, in_rgb_frame_id_sample)
                
            frame_id = frame_id[sample].astype(np.int32)
        
        # check whether pics exists
        print("frame_id: ", frame_id)
        self.get_paths(dir_rgb_0, dir_rgb_1, frame_id)
        cam2world_0 = pose_cam_0[sample, 1:].reshape(-1, 4, 4)[:, :3]
        sys2world = np.genfromtxt(os.path.join(dir_poses, 'poses.txt'))
        sys2world = sys2world[sample, 1:].reshape(-1, 3, 4)
        cam2sys_1 = parse_calib_file(os.path.join(dir_calib, 'calib_cam_to_pose.txt'), 'image_01')
        cam2sys_1 = np.concatenate([cam2sys_1.reshape(3, 4), np.array([[0, 0, 0, 1]])], axis=0)
        R_rect_01 = parse_calib_file(intrinsic_path, 'R_rect_01').reshape(3, 3)
        R_rect = np.eye(4)
        R_rect[:3:, :3] = R_rect_01.T
        cam2world_1 = sys2world @ cam2sys_1 @ R_rect
        test_id = np.array(kwargs['test_id']).astype(np.int32)
        
        test_id_normalized = []
        for tid in test_id:
            assert tid in frame_id.tolist()
            test_id_normalized.append(frame_id.tolist().index(tid))
        test_id_normalized = np.array(test_id_normalized, dtype=np.int32).reshape(-1)
        
        # dataset split
        if self.split == 'train' and kwargs.get('strict_split', False):
            train_id_normalized = []
            train_ids = []
            for train_i, train_id in enumerate(frame_id.tolist()):
                if train_id not in test_id:
                    train_id_normalized.append(train_i)
                    train_ids.append(train_id)
            train_id_normalized = np.array(train_id_normalized, dtype=np.int32).reshape(-1)
            train_ids = np.array(train_ids, dtype=np.int32).reshape(-1)
            kwargs['train_id_normalized'] = train_id_normalized
        
        # get camera poses
        self.setup_poses(cam2world_0, cam2world_1, test_id_normalized, **kwargs)
        
        if self.split != 'train':
            frame_id = test_id
        if self.split == 'train' and kwargs.get('strict_split', False):
            frame_id = train_ids

        if kwargs.get('load_imgs', True):

            rgb_0 = self.read_rgb(dir_rgb_0, frame_id)
            rgb_1 = self.read_rgb(dir_rgb_1, frame_id)
            self.rays = torch.FloatTensor(np.concatenate([rgb_0, rgb_1], axis=0))

            if kwargs.get('use_mask',False):
                mask_0 = self.read_mask(dir_mask_0, frame_id)
                mask_1 = self.read_mask(dir_mask_1, frame_id)
                self.masks = torch.BoolTensor(np.concatenate([mask_0, mask_1], axis=0))
            else:
                self.masks = None

            sem_0 = self.read_semantics(dir_sem_0, dir_sem_0_sup, frame_id)
            sem_1 = self.read_semantics(dir_sem_1, dir_sem_1_sup, frame_id)
            self.labels = torch.LongTensor(np.concatenate([sem_0, sem_1], axis=0))
            self.sky_label = 23
            if kwargs.get('normal_mono', False):
                normal_0 = self.read_normal(dir_normal_0, frame_id)
                normal_1 = self.read_normal(dir_normal_1, frame_id)
                self.normals = torch.FloatTensor(np.concatenate([normal_0, normal_1], axis=0))
            if kwargs.get('depth_mono', False):
                depth_0 = self.read_depth(dir_depth_0, frame_id)
                depth_1 = self.read_depth(dir_depth_1, frame_id)
                self.depths = torch.FloatTensor(np.concatenate([depth_0, depth_1], axis=0))
            if kwargs.get('lidar_proxy', False):
                lidarp_0, lidarp_0_mask = self.read_lidar_proxy(dir_lidarp_0, frame_id)
                lidarp_1, lidarp_1_mask = self.read_lidar_proxy(dir_lidarp_1, frame_id)
                self.lidarp = torch.FloatTensor(np.concatenate([lidarp_0, lidarp_1], axis=0))
                self.lidarp_mask = torch.BoolTensor(np.concatenate([lidarp_0_mask, lidarp_1_mask], axis=0))
            if kwargs.get('depth_dp_proxy', False):
                depth_dp = self.read_depth_dp(dir_depth_dp, frame_id)
                self.depth_dp = torch.FloatTensor(np.concatenate([depth_dp, depth_dp], axis=0)) / self.pos_trans['scale']
        
        # perspective projection matrix
        self.near = n = 1e-4
        self.far = f = 20 # infinite
        fx, fy, cx, cy = self.K[0,0], self.K[1,1], self.K[0,2], self.K[1,2]
        print("fx, fy, cx, cy: ", fx, fy, cx, cy)
        width = self.img_wh[0]
        height = self.img_wh[1]
        n00 = 2.0 * fx / width
        n11 = 2.0 * fy / height
        n02 = 1.0 - 2.0 * cx / width
        n12 = 2.0 * cy / height - 1.0
        n32 = -1.0
        n22 = (f + n) / (n - f)
        n23 = (2 * f * n) / (n - f)
        camera_projmat = np.array([[n00, 0, n02, 0],
								[0, n11, n12, 0],
								[0, 0, n22, n23], 
								[0, 0, n32, 0]], dtype=np.float32)

        self.projection = torch.from_numpy(camera_projmat)
        bottom = torch.tensor([0, 0, 0, 1]).reshape(1, 1, 4).expand((self.poses.shape[0], -1, -1))
        i_pose = self.poses.clone()
        i_pose[:, :3, 1:3] = -i_pose[:, :3, 1:3]
        square_pose = torch.cat((i_pose, bottom), dim=1)
        self.mvps = self.projection.unsqueeze(0) @ torch.inverse(square_pose)
        self.H = self.img_wh[1]
        self.W = self.img_wh[0]


    def mvp_permute(self, index):

        pose = self.poses[index].clone().unsqueeze(0)

        n = self.near
        f = self.far
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        width = self.img_wh[0]
        height = self.img_wh[1]
        n00 = 2.0 * fx / width
        n11 = 2.0 * fy / height

        rpx = np.random.random() - 0.5
        rpy = np.random.random() - 0.5
        cx = cx + rpx
        cy = cy + rpy
        n02 = 1.0 - 2.0 * cx / width
        n12 = 2.0 * cy / height - 1.0
        n32 = -1.0
        n22 = (f + n) / (n - f)
        n23 = (2 * f * n) / (n - f)
        camera_projmat = np.array([[n00, 0, n02, 0],
                                   [0, n11, n12, 0],
                                   [0, 0, n22, n23],
                                   [0, 0, n32, 0]], dtype=np.float32)

        projection = torch.from_numpy(camera_projmat)

        bottom = torch.tensor([0, 0, 0, 1]).reshape(1, 1, 4).expand((pose.shape[0], -1, -1))

        pose[:, :3, 1:3] = -pose[:, :3, 1:3]
        square_pose = torch.cat((pose, bottom), dim=1)

        mvp = projection.unsqueeze(0) @ torch.inverse(square_pose)

        u, v = self.grid.unbind(-1)
        directions =  torch.stack([(u - cx + 0.5) / fx, (v - cy + 0.5) / fy, torch.ones_like(u)], -1)
        directions = directions.reshape(-1, 3)

        return mvp.reshape(4, 4), directions
    
    def get_paths(self, dir_rgb_0, dir_rgb_1, frame_id):
        self.rgb_paths = []
        for i in frame_id:
            path = os.path.join(dir_rgb_0, '{:0>10d}.png'.format(i))
            self.rgb_paths.append(path)
        for i in frame_id:
            path = os.path.join(dir_rgb_1, '{:0>10d}.png'.format(i))
            self.rgb_paths.append(path)
        
    def setup_poses(self, cam2world_0, cam2world_1, test_id_normalized, **kwargs):
        self.original_poses = np.concatenate([cam2world_0, cam2world_1], axis=0)
        pos_0 = cam2world_0[:, :, -1]
        pos_1 = cam2world_1[:, :, -1]
        pos = np.concatenate([pos_0, pos_1], axis=0)

        # camera pose normalization
        if kwargs.get('kitti_dual_seq', False):
            center_x = (np.max(pos[:, 0]) + np.min(pos[:, 0]))/2
            center_y = (np.max(pos[:, 1]) + np.min(pos[:, 1]))/2
            center_z = (np.max(pos[:, 2]) + np.min(pos[:, 2]))/2
            center = np.array([center_x, center_y, center_z]).reshape(1, -1)

            scale_x = (np.max(pos[:, 0]) - np.min(pos[:, 0]))
            scale_y = (np.max(pos[:, 1]) - np.min(pos[:, 1]))
            scale_z = (np.max(pos[:, 2]) - np.min(pos[:, 2]))
            scale = 0.5 * np.max([scale_x, scale_y, scale_z]).reshape(1) / 0.9
            
            forward = np.array([0, 0, 0]).reshape(1, -1)
        else:
            center = np.mean(pos, axis=0)
            forward = pos_0[-1] - pos_0[0]
            forward = forward / np.linalg.norm(forward)
            diff = pos.reshape(-1, 1, 3) - pos.reshape(1, -1, 3)
            dist = np.sqrt(np.sum(diff**2, axis=-1))
            scale = dist.max() / 2

        self.depth_scale = scale

        pos = (pos - center.reshape(1, -1)) / scale
        pos = pos - forward.reshape(1, -1) * 0.5
        
        self.pos_trans = {
            'center': center.reshape(1, -1),
            'scale': scale,
            'forward': forward.reshape(1, -1)
        }
        
        cam2world = np.concatenate([cam2world_0, cam2world_1], axis=0)
        cam2world[:, :, -1] = pos
       
        if self.split != 'train':
            n_step = cam2world_0.shape[0]
            test_id = np.concatenate([test_id_normalized, test_id_normalized + n_step])
            cam2world = cam2world[test_id]
        
        if self.split == 'train' and kwargs.get('strict_split', False):
            n_step = cam2world_0.shape[0]
            train_id_normalized = kwargs['train_id_normalized']
            train_id = np.concatenate([train_id_normalized, train_id_normalized + n_step])
            cam2world = cam2world[train_id]

        print("images in dataset: ", cam2world.shape[0])
        
        self.poses = torch.FloatTensor(cam2world)

    def read_rgb(self, dir_rgb, frame_id):
        rgb_list = []
        for i in frame_id:
            path = os.path.join(dir_rgb, '{:0>10d}.png'.format(i))
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            img = (img / 255.0).astype(np.float32)
            rays = img.reshape(-1, 3)
            rgb_list.append(rays)
        rgb_list = np.stack(rgb_list)
        return rgb_list
    
    def read_mask(self, dir_rgb, frame_id):
        rgb_list = []
        for i in frame_id:
            path = os.path.join(dir_rgb, '{:0>10d}.png'.format(i))
            if not os.path.exists(path):
                rays = np.zeros((self.img_wh[0]*self.img_wh[1]), dtype=np.bool_)
            else:
                img = cv2.imread(path,cv2.IMREAD_UNCHANGED)
                img = (img / 255.0).astype(np.uint8).astype(np.bool_)
                rays = img.reshape(-1)
            rgb_list.append(rays)
        rgb_list = np.stack(rgb_list)
        return rgb_list
    
    def read_semantics(self, dir_sem, dir_sem_sup, frame_id):
        label_list = []
        for i in frame_id:
            path = os.path.join(dir_sem, '{:0>10d}.png'.format(i))
            _path = os.path.join(dir_sem, '{:0>10d}.pgm'.format(i))
            __path = os.path.join(dir_sem_sup, '{:0>10d}.pgm'.format(i))
            if os.path.exists(__path):
                label = imageio.imread(__path).astype(np.int32).reshape(-1)
            elif os.path.exists(path):
                label = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                label = self.label_mapping(label.flatten())
            elif os.path.exists(_path):
                label = imageio.imread(_path).astype(np.int32).reshape(-1)
            else:
                assert False, "no label"
            label_list.append(label)
        label_list = np.stack(label_list)
        return label_list
    
    def label_mapping(self, label):
        label_new = np.ones_like(label).astype(np.int32)
        label_new[:] = 256
        mask = np.logical_or(label==7, label==8)
        label_new[mask] = 0 # road
        mask = np.logical_and(label>=11, label<=20)
        mask = np.logical_or(label==35, mask)
        label_new[mask] = 1 # bulding
        label_new[label==21] = 2 # vegetation
        label_new[label==22] = 3 # terrain
        label_new[label==23] = 4 # sky
        mask = np.logical_or(label==24, label==25)
        label_new[mask] = 5 # person
        mask = np.logical_and(label>=26, label<=33)
        label_new[mask] = 6 # vehicle
        return label_new
    
    def read_normal(self, dir_normal, frame_id):
        poses = self.poses.numpy()
        normal_list = []
        for c2w, i in zip(poses, frame_id):
            path = os.path.join(dir_normal, '{:0>10d}.png'.format(i))
            img = Image.open(path)
            img = np.array(img)/255.0
            normal = ((img - 0.5) * 2).reshape(-1, 3)
            normal = normal @ c2w[:,:3].T
            normal_list.append(normal)
        normal_list = np.stack(normal_list)
        return normal_list
    
    def read_depth(self, dir_depth, frame_id):
        depth_list = []
        for i in frame_id:
            path = os.path.join(dir_depth, '{:0>10d}.npy'.format(i))
            depth_list.append(np.load(path).reshape(-1))
        depth_list = np.stack(depth_list)
        return depth_list
    
    def read_lidar_proxy(self, dir_lidar_proxy, frame_id):
        lidar_list = []
        lidar_mask_list = []
        w, h = self.img_wh
        for i in frame_id:
            path = os.path.join(dir_lidar_proxy, 'frame_{:0>10d}.pkl'.format(i))
            
            with open(path, 'rb') as f_pkl:
                frame_info = pickle.load(f_pkl)
                valid = frame_info['valid'].astype(np.bool_)
                valid_points = frame_info['points']
            
            frame_points = np.zeros((h, w, 3))
            frame_points[valid] = valid_points
            
            frame_points = frame_points.reshape(-1, 3)
            frame_points = (frame_points - self.pos_trans['center'].reshape(1, -1)) / self.pos_trans['scale']
            frame_points = frame_points - self.pos_trans['forward'].reshape(1, -1) * 0.5

            lidar_list.append(frame_points)
            lidar_mask_list.append(valid.reshape(-1))
        
        lidar_list = np.stack(lidar_list)
        lidar_mask_list = np.stack(lidar_mask_list)
        return lidar_list, lidar_mask_list

    def read_depth_dp(self, dir_depth_dp, frame_id):
        depth_dp_list = []
        w, h = self.img_wh
        for i in frame_id:
            path = os.path.join(dir_depth_dp, '{:0>10d}.npy'.format(i))
            depth_dp_list.append(np.load(path).reshape(-1))
        depth_dp_list = np.stack(depth_dp_list)
        return depth_dp_list
    
def parse_calib_file(path, key):
    file = open(path, 'r')
    lines = file.readlines()
    for line in lines:
        if key in line:
            tokens = line.strip().split(' ')
            nums = tokens[1:]
            array = np.array([float(i) for i in nums])
            return array
    return None

