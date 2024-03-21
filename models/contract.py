import torch 
def contract(
    x: torch.Tensor,
    roi: torch.Tensor,
    type: str
) -> torch.Tensor:
    if type == 'AABB':
        x = (x - roi[0]) / (roi[1] - roi[0])
    elif type == 'Sphere':
        x = (x - roi[0]) / (roi[1] - roi[0])
        x = x * 2 - 1
        
        norm_sq = (x * x).sum(-1).reshape(-1, 1)
        norm = torch.sqrt(norm_sq)
        
        out_sphere = norm > 1
        in_sphere = torch.logical_not(out_sphere)
        x = x * in_sphere + ((2 - 1 / norm) * (x / norm)) * out_sphere
        
        x = x * 0.25 + 0.5
    else:
        raise NotImplementedError
    return x

def contract_inv(
    x: torch.Tensor,
    roi: torch.Tensor,
    type: str
) -> torch.Tensor:
    if type == 'AABB':
        x = x * (roi[1] - roi[0]) + roi[0]
    elif type == 'Sphere':
        x = (x - 0.5) * 4
        norm_sq = (x * x).sum(-1).reshape(-1, 1)
        norm = torch.sqrt(norm_sq)
        out_sphere = norm > 1
        in_sphere = torch.logical_not(out_sphere)
        x = x * in_sphere + (x / torch.clamp(2 * norm - norm_sq, min=1e-10)) * out_sphere
        x = x * 0.5 + 0.5
        x = x * (roi[1] - roi[0]) + roi[0]
    else:
        raise NotImplementedError
    return x