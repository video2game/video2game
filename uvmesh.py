import pywavefront
import numpy as np

def read_obj(uv_save_path):
    scene = pywavefront.Wavefront(uv_save_path, collect_faces=True)

    v = np.array(scene.vertices)
    vt = np.array(scene.parser.tex_coords)
    if len(vt.shape) > 1:
        vt[:, 1] = 1 - vt[:, 1]
    f = np.array(scene.parser.mesh.faces).astype(np.int32)
    if len(vt.shape) > 1:
        ft = np.array(scene.parser.mesh.faces_textures).astype(np.int32)
    else:
        ft = None

    return v, vt, f, ft


