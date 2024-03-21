import os, subprocess
import argparse
import trimesh
import numpy as np
from meshutils import *
import json
import subprocess
# import pymesh

def cross(a, b):
    c = [a[1]*b[2] - a[2]*b[1],
    a[2]*b[0] - a[0]*b[2],
    a[0]*b[1] - a[1]*b[0]]

    return c

def computeNormal(va, vb, vc):
    ab = vb - va
    cb = vc - vb
    return np.array(cross(cb, ab)).reshape(3)

def getFaceNormal(vertices, faces, i):
    f = faces[i]
    va = vertices[f[0]]
    vb = vertices[f[1]]
    vc = vertices[f[2]]
    return computeNormal(va, vb, vc)

def dot(a, b):
    return float(a[0]*b[0]+a[1]*b[1]+a[2]*b[2])

def filter_mesh_from_vertices(keep, mesh_points, faces):
    # filter_mapping = np.arange(keep.shape[0])[keep]
    # filter_unmapping = -np.ones((keep.shape[0]))
    # filter_unmapping[filter_mapping] = np.arange(filter_mapping.shape[0])
    # mesh_points = mesh_points[keep]
    keep_0 = keep[faces[:, 0]]
    keep_1 = keep[faces[:, 1]]
    keep_2 = keep[faces[:, 2]]
    keep_faces = np.logical_or(keep_0, keep_1)
    keep_faces = np.logical_or(keep_faces, keep_2)

    faces = faces[keep_faces]
    # faces[:, 0] = filter_unmapping[faces[:, 0]]
    # faces[:, 1] = filter_unmapping[faces[:, 1]]
    # faces[:, 2] = filter_unmapping[faces[:, 2]]
    keep = np.sort(np.unique(faces.reshape(-1)))
    filter_mapping = keep
    filter_unmapping = -np.ones((mesh_points.shape[0]))
    filter_unmapping[filter_mapping] = np.arange(filter_mapping.shape[0])
    mesh_points = mesh_points[keep]

    faces[:, 0] = filter_unmapping[faces[:, 0]]
    faces[:, 1] = filter_unmapping[faces[:, 1]]
    faces[:, 2] = filter_unmapping[faces[:, 2]]
    return mesh_points, faces

def filter_mesh_from_vertices_strict(keep, mesh_points, faces):
    # filter_mapping = np.arange(keep.shape[0])[keep]
    # filter_unmapping = -np.ones((keep.shape[0]))
    # filter_unmapping[filter_mapping] = np.arange(filter_mapping.shape[0])
    # mesh_points = mesh_points[keep]
    keep_0 = keep[faces[:, 0]]
    keep_1 = keep[faces[:, 1]]
    keep_2 = keep[faces[:, 2]]
    keep_faces = np.logical_and(keep_0, keep_1)
    keep_faces = np.logical_and(keep_faces, keep_2)

    faces = faces[keep_faces]
    # faces[:, 0] = filter_unmapping[faces[:, 0]]
    # faces[:, 1] = filter_unmapping[faces[:, 1]]
    # faces[:, 2] = filter_unmapping[faces[:, 2]]
    keep = np.sort(np.unique(faces.reshape(-1)))
    filter_mapping = keep
    filter_unmapping = -np.ones((mesh_points.shape[0]))
    filter_unmapping[filter_mapping] = np.arange(filter_mapping.shape[0])
    mesh_points = mesh_points[keep]

    faces[:, 0] = filter_unmapping[faces[:, 0]]
    faces[:, 1] = filter_unmapping[faces[:, 1]]
    faces[:, 2] = filter_unmapping[faces[:, 2]]
    return mesh_points, faces

def partition_xyz(mesh_points, faces, x_min, y_min, z_min, x_max, y_max, z_max):
    keep_x = np.logical_and(mesh_points[:, 0] < x_max, mesh_points[:, 0] > x_min)
    keep_y = np.logical_and(mesh_points[:, 1] < y_max, mesh_points[:, 1] > y_min)
    keep_z = np.logical_and(mesh_points[:, 2] < z_max, mesh_points[:, 2] > z_min)
    keep = np.logical_and(np.logical_and(keep_x, keep_y), keep_z)
    if keep.any():
        return filter_mesh_from_vertices(keep, mesh_points, faces)
    else:
        return None

def slim_mesh_bounding_mesh(vertices, faces, step, min_v):
    target_v = max(min_v, vertices.shape[0]-step)
    print(f"[info] before: {vertices.shape[0]} vertices")
    # target_v = min_v
    mesh = trimesh.Trimesh(vertices, faces)
    mesh_path = args.file_path[:-4] + "_tmp.obj"
    trimesh.exchange.export.export_mesh(mesh, mesh_path)
    exec_cmd = f"{args.bmesh_path} --direction Any --vertices {target_v} --metric Average --init Midpoint {mesh_path} {mesh_path}"
    print(f"[exec] {exec_cmd}")
    subprocess.run(exec_cmd, shell=True)
    mesh = trimesh.load_mesh(mesh_path)
    print(f"[info] after: {mesh.vertices.shape[0]} vertices")
    return mesh.vertices, mesh.faces


#
# def slim_mesh_pymesh(vertices, faces, target_len = 0.002):
#     mesh = pymesh.form_mesh(vertices, faces)
#
#     print("Target resolution: {} mm".format(target_len))
#
#     count = 0
#     num_vertices = mesh.num_vertices
#     mesh, __ = pymesh.remove_degenerated_triangles(mesh, 100)
#     mesh, __ = pymesh.split_long_edges(mesh, target_len)
#     print("#v after split_long_edges: {}".format(mesh.num_vertices))
#
#     while True:
#         mesh, __ = pymesh.collapse_short_edges(mesh, 1e-6)
#         mesh, __ = pymesh.collapse_short_edges(mesh, target_len)
#         print("#v after collapse_short_edges: {}".format(mesh.num_vertices))
#         mesh, __ = pymesh.remove_obtuse_triangles(mesh, 150.0, 100)
#         print("#v after remove_obtuse_triangles: {}".format(mesh.num_vertices))
#
#         if mesh.num_vertices == num_vertices:
#             break
#         num_vertices = mesh.num_vertices
#         count += 1
#         if count >= 1: break
#
#     mesh = pymesh.resolve_self_intersection(mesh)
#     mesh, __ = pymesh.remove_duplicated_faces(mesh)
#     # mesh = pymesh.compute_outer_hull(mesh)
#     # mesh, __ = pymesh.remove_duplicated_faces(mesh)
#     mesh, __ = pymesh.remove_obtuse_triangles(mesh, 179.0, 1)
#     # mesh, __ = pymesh.remove_isolated_vertices(mesh)
#     vertices, faces = mesh.vertices, mesh.faces
#     return vertices, faces

def normalize_mesh(vertices, faces):

    x_min = vertices[:, 0].min()
    x_max = vertices[:, 0].max()
    y_min = vertices[:, 1].min()
    y_max = vertices[:, 1].max()
    z_min = vertices[:, 2].min()
    z_max = vertices[:, 2].max()

    center = np.array([(x_min+x_max)/2, (y_min+y_max)/2, (z_min+z_max)/2]).reshape(1, 3)
    scale = np.array([x_max - x_min, y_max - y_min, z_max - z_min]).reshape(-1).max()

    vertices = vertices - center
    vertices = vertices / scale

    x_min = vertices[:, 0].min()
    x_max = vertices[:, 0].max()
    y_min = vertices[:, 1].min()
    y_max = vertices[:, 1].max()
    z_min = vertices[:, 2].min()
    z_max = vertices[:, 2].max()

    bound = [
        [x_min, y_min, z_min],
        [x_max, y_max, z_max]
    ]

    return vertices, faces, center, scale, bound

def regularize_mesh(vertices, faces, bound):
    print(f"[info] regularize before: {vertices.shape[0]} vertices")
    x_in = np.logical_and(vertices[:, 0] >= bound[0][0], vertices[:, 0] <= bound[1][0])
    y_in = np.logical_and(vertices[:, 1] >= bound[0][1], vertices[:, 1] <= bound[1][1])
    z_in = np.logical_and(vertices[:, 2] >= bound[0][2], vertices[:, 2] <= bound[1][2])
    keep = np.logical_and(np.logical_and(x_in, y_in), z_in)
    vertices, faces = filter_mesh_from_vertices_strict(keep, vertices, faces)
    print(f"[info] regularize after: {vertices.shape[0]} vertices")
    return vertices, faces

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, required=True, help="the mesh path to generate corresponding convex collision model for three js")
    parser.add_argument('--collision_type', type=str, required=True, choices=['convex', 'trimesh'], help="the type of generated collision model")
    parser.add_argument('--vhacd_path', type=str, default=None, help="the executable file of v-hacd for convex decomposition, like: path/to/dir/v-hacd/app/build/TestVHACD")
    parser.add_argument('--bmesh_path', type=str, default=None, help="the executable file of bounding-mesh for trimesh collider, like: path/to/dir/bounding-mesh/bin/boundingmesh")
    parser.add_argument('--trimesh_target', type=int, default=5000, help="the target trimesh collider vertices count per partition")
    parser.add_argument('--scale', type=float, default=1, help="the scale of collision model")
    parser.add_argument('--slice', type=int, default=1, help="partition for the generated collision model")
    args = parser.parse_args()

    args.file_path = os.path.abspath(args.file_path)
    mesh = trimesh.load(args.file_path, force='mesh', skip_material=True, process=False)

    slice = args.slice

    if args.collision_type == "convex":
        assert args.vhacd_path is not None, "please specify a valid V-HACD executable"
        args.vhacd_path = os.path.abspath(args.vhacd_path)
        x_min = mesh.vertices[:, 0].min()
        x_max = mesh.vertices[:, 0].max()
        y_min = mesh.vertices[:, 1].min()
        y_max = mesh.vertices[:, 1].max()
        z_min = mesh.vertices[:, 2].min()
        z_max = mesh.vertices[:, 2].max()

        delta_x = (x_max - x_min) / slice
        delta_y = (y_max - y_min) / slice
        delta_z = (z_max - z_min) / 1
        partition_meshes = []

        # partition the mesh
        for x_i in range(slice):
            for y_i in range(slice):
                for z_i in range(1):
                    _x_min = x_min + x_i * delta_x
                    _y_min = y_min + y_i * delta_y
                    _z_min = z_min + z_i * delta_z

                    _x_max = _x_min + delta_x
                    _y_max = _y_min + delta_y
                    _z_max = _z_min + delta_z

                    result = partition_xyz(mesh.vertices, mesh.faces, _x_min, _y_min, _z_min, _x_max, _y_max, _z_max)
                    if result is None:
                        continue
                    else:
                        mesh_points, faces = result

                    if faces.shape[0] > 0:
                        partition_meshes.append((mesh_points, faces))

        dir_path = os.path.dirname(args.file_path)
        decomp_dir = os.path.join(dir_path, "decomp_"+os.path.splitext(os.path.basename(args.file_path))[0])
        output_file_path = os.path.join(args.file_path[:-4]+f'_scale{args.scale}_tmp.obj')
        print("Export to ", decomp_dir)

        decomp_scene = trimesh.Scene()
        centers = []
        n_cnt = 0

        # convex decomposition for each part of mesh
        for mesh_i, (points, triangles) in enumerate(partition_meshes):

            # scaling
            v = np.array(points) * args.scale
            f = np.array(triangles)
            mesh = trimesh.Trimesh(v, f)

            # decomp using vhacd
            trimesh.exchange.export.export_mesh(mesh, output_file_path)
            cmd = f"cd {dir_path}; {args.vhacd_path} {output_file_path} -e 0.001 -d 10000000 -r 64 -h 256; mkdir -p {decomp_dir}; mv decomp.* {decomp_dir}"
            subprocess.run(cmd, shell=True)
            mesh_path = os.path.join(decomp_dir, "decomp.obj")
            mesh = trimesh.load_mesh(mesh_path)

            # load the decomp result
            # post-process for loading into three js
            for ci, convex in enumerate(mesh.geometry.values()):
                _vertices = np.array(convex.vertices)
                faces = np.array(convex.faces, dtype=np.int32)
                center = _vertices.mean(axis=0).reshape(1, 3)

                # three js require each convex centering in origin and has the property of strict convex
                # we need to check so as to successfully load into three js
                vertices = _vertices - center
                broken = False
                for i in range(faces.shape[0]):
                    n = -getFaceNormal(vertices, faces, i)
                    vertex = vertices[faces[i][0]].reshape(3)
                    dot_rs = dot(n, vertex)
                    if dot_rs < 0:
                        broken = True
                        break

                if broken:
                    continue

                centers.append([float(item) for item in center.reshape(3).tolist()])
                fix_convex = trimesh.Trimesh(vertices, faces)
                decomp_scene.add_geometry(fix_convex, node_name=f"node_{n_cnt}", geom_name=f"geo_{n_cnt}")
                n_cnt += 1

        trimesh.exchange.export.export_scene(decomp_scene, os.path.join(decomp_dir, 'decomp.glb'))

        with open(os.path.join(decomp_dir, 'center.json'), 'w') as f_json:
            json.dump(centers, f_json)

    elif args.collision_type == "trimesh":
        assert args.bmesh_path is not None, "please specify a valid bounding-mesh executable"
        vertices = mesh.vertices
        faces = mesh.faces

        print("vertices: ", vertices.shape)

        x_min = mesh.vertices[:, 0].min()
        x_max = mesh.vertices[:, 0].max()
        y_min = mesh.vertices[:, 1].min()
        y_max = mesh.vertices[:, 1].max()
        z_min = mesh.vertices[:, 2].min()
        z_max = mesh.vertices[:, 2].max()

        delta_x = (x_max - x_min) / slice
        delta_y = (y_max - y_min) / slice
        delta_z = (z_max - z_min) / 1
        partition_meshes = []

        # partition the mesh
        for x_i in range(slice):
            for y_i in range(slice):
                for z_i in range(1):
                    _x_min = x_min + x_i * delta_x
                    _y_min = y_min + y_i * delta_y
                    _z_min = z_min + z_i * delta_z

                    _x_max = _x_min + delta_x
                    _y_max = _y_min + delta_y
                    _z_max = _z_min + delta_z

                    result = partition_xyz(mesh.vertices, mesh.faces, _x_min, _y_min, _z_min, _x_max, _y_max, _z_max)
                    if result is None:
                        continue
                    else:
                        mesh_points, faces = result

                    if faces.shape[0] > 0:
                        partition_meshes.append((mesh_points, faces))

        # convex decomposition for each part of mesh

        scene = trimesh.Scene()
        n_cnt = 0

        for mesh_i, (points, triangles) in enumerate(partition_meshes):
            points, triangles, center, scale, bound = normalize_mesh(points, triangles)

            min_v = args.trimesh_target
            step = 100000

            points, triangles = merge_close_vertices(points, triangles, 2)
            points, triangles = decimate_mesh(points, triangles, 200000, 'pyfqmr')
            points, triangles = decimate_mesh(points, triangles, 200000)

            while points.shape[0] > min_v:
                points, triangles = merge_close_vertices(points, triangles, 2)
                before_v = points.shape[0]
                points, triangles = slim_mesh_bounding_mesh(points, triangles, step, min_v)
                points, triangles = regularize_mesh(points, triangles, bound)
                if points.shape[0] == before_v:
                    step = step // 2
                if step == 1:
                    break

            points = points * scale + center

            scene.add_geometry(trimesh.Trimesh(points, triangles), node_name=f"node_{n_cnt}", geom_name=f"geo_{n_cnt}")
            n_cnt += 1

        trimesh.exchange.export.export_scene(scene, args.file_path[:-4] + '_trimesh_collider.glb')


    else:
        assert False