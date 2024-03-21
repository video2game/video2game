from venv import create
import torch
from torch import nn
import torch.nn.functional as F
import tinycudann as tcnn
import vren
from .custom_functions import TruncExp, TruncTanh
import numpy as np
from einops import rearrange
from .rendering import NEAR_DISTANCE
from .contract import contract
from .ref_util import *

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=bias))

        self.net = nn.ModuleList(net)
    
    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
        return x

# texture-color NGP
class TCNGP(nn.Module):
    def __init__(self, scale, contraction_type, sphere_scale, ngp_params, specular_dim_mlp = 64, per_level_scale=16, specular_dim=3, color_net_params={}):
        super().__init__()
        
        # scene bounding box
        self.scale = scale
        self.contraction_type = contraction_type
        
        if self.contraction_type == "AABB":
            self.register_buffer('aabb', torch.cat((-torch.ones(1, 3) * self.scale,
                                                    torch.ones(1, 3) * self.scale),
                                                   dim=0).reshape(2, 3))
        else:
            self.register_buffer('aabb', torch.cat((-torch.ones(1, 3) * torch.tensor(sphere_scale[0]).reshape(1, 3), torch.ones(1, 3) * torch.tensor(sphere_scale[1]).reshape(1, 3)), dim=0).reshape(2, 3))

        # constants
        L_ = ngp_params['L']; F_ = ngp_params['F']; log2_T_ = ngp_params['log2_T']; N_min_ = 16
        b_ = np.exp(np.log(2048 * per_level_scale / N_min_) / (L_ - 1))
        print(f'GridEncoding for RGB: Nmin={N_min_} b={b_:.5f} F={F_} T=2^{log2_T_} L={L_}')

        self.encoder_color = \
            tcnn.Encoding(3, {
                "otype": "HashGrid",
                "n_levels": L_,
                "n_features_per_level": F_,
                "log2_hashmap_size": log2_T_,
                "base_resolution": N_min_,
                "per_level_scale": b_,
                "interpolation": "Linear"
            })

        self.specular_dim = specular_dim
        
        self.color_net = MLP(self.encoder_color.n_output_dims, 3 + specular_dim, color_net_params["dim_hidden"], color_net_params["num_layers"], bias=False)
        
        self.specular_net = MLP(specular_dim + 3, 3, specular_dim_mlp, 2, bias=False)
        
    def geo_feat(self, x, c=None, make_contract=False):
        
        if make_contract:
            x = contract(x, roi=self.aabb.to(x.device), type=self.contraction_type)

        h = self.encoder_color(x)
        if c is not None:
            h = torch.cat([h, c.repeat(x.shape[0], 1) if c.shape[0] == 1 else c], dim=-1)
        h = self.color_net(h)
        geo_feat = torch.sigmoid(h)
        
        # discretization
        discrete_geo_feat = (geo_feat * 255).int().float() / 255
        geo_feat = geo_feat + (discrete_geo_feat - geo_feat).detach()
        
        return geo_feat


    def rgb(self, x, d, c=None, shading='full', make_contract=False):

        # color
        if make_contract:
            x = contract(x, roi=self.aabb.to(x.device), type=self.contraction_type)

        geo_feat = self.geo_feat(x, c)
        diffuse = geo_feat[..., :3]

        if shading == 'diffuse':
            color = diffuse
            specular = None
        else: 
            specular = self.specular_net(torch.cat([d, geo_feat[..., 3:]], dim=-1))
            specular = torch.sigmoid(specular)
            if shading == 'specular':
                color = specular
            else: # full
                color = specular + diffuse # specular + albedo

        return color, specular
        
    def forward(self, x, d, c, shading='full', make_contract=True):
        
        if make_contract:
            x = contract(x, roi=self.aabb.to(x.device), type=self.contraction_type)
        
        color, specular = self.rgb(x, d, c, shading)

        return color, specular