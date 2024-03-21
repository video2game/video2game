import numpy as np
import pymeshlab as pml

def isotropic_explicit_remeshing(verts, faces):

    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    m = pml.Mesh(verts, faces)
    ms = pml.MeshSet()
    ms.add_mesh(m, 'mesh') # will copy!

    # filters
    # ms.apply_coord_taubin_smoothing()
    ms.remeshing_isotropic_explicit_remeshing(iterations=3, targetlen=pml.Percentage(1))

    # extract mesh
    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    print(f'[INFO] isotropic explicit remesh: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}')

    return verts, faces
    

def decimate_mesh(verts, faces, target, backend='pymeshlab', remesh=False, optimalplacement=False, preserve_border=True):
    # optimalplacement: default is True, but for flat mesh must turn False to prevent spike artifect.

    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    if backend == 'pyfqmr':
        import pyfqmr
        solver = pyfqmr.Simplify()
        solver.setMesh(verts, faces)
        solver.simplify_mesh(target_count=target, preserve_border=preserve_border, verbose=False)
        verts, faces, normals = solver.getMesh()

        # repair
        m = pml.Mesh(verts, faces)
        ms = pml.MeshSet()
        ms.add_mesh(m, 'mesh')  # will copy!

        # ms.repair_non_manifold_edges_by_removing_faces()
        # ms.repair_non_manifold_vertices_by_splitting(vertdispratio=0)

        # extract mesh
        m = ms.current_mesh()
        verts = m.vertex_matrix()
        faces = m.face_matrix()
    else:

        m = pml.Mesh(verts, faces)
        ms = pml.MeshSet()
        ms.add_mesh(m, 'mesh') # will copy!

        # filters
        # ms.meshing_decimation_clustering(threshold=pml.Percentage(1))
        ms.simplification_quadric_edge_collapse_decimation(targetfacenum=int(target), optimalplacement=optimalplacement)

        if remesh:
            # ms.apply_coord_taubin_smoothing()
            ms.remeshing_isotropic_explicit_remeshing(iterations=3, targetlen=pml.Percentage(1))

        # ms.repair_non_manifold_edges_by_removing_faces()
        # ms.repair_non_manifold_vertices_by_splitting(vertdispratio=0)

        # extract mesh
        m = ms.current_mesh()
        verts = m.vertex_matrix()
        faces = m.face_matrix()

    print(f'[INFO] mesh decimation: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}')

    return verts, faces


def remove_masked_trigs(verts, faces, mask, dilation=10):
    # mask: 0 == keep, 1 == remove

    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    m = pml.Mesh(verts, faces, f_quality_array=mask) # mask as the quality
    ms = pml.MeshSet()
    ms.add_mesh(m, 'mesh') # will copy!

    # select faces
    ms.conditional_face_selection(condselect='fq == 0') # select kept faces
    # dilate to aviod holes...
    for _ in range(dilation):
        ms.dilate_selection()
    ms.invert_selection(invfaces=True) # invert

    # delete faces
    ms.delete_selected_faces()

    # clean unref verts
    ms.remove_unreferenced_vertices()

    # extract
    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    print(f'[INFO] mesh mask trigs: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}')

    return verts, faces    


def remove_masked_verts(verts, faces, mask):
    # mask: 0 == keep, 1 == remove

    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    m = pml.Mesh(verts, faces, v_quality_array=mask) # mask as the quality
    ms = pml.MeshSet()
    ms.add_mesh(m, 'mesh') # will copy!

    # select verts
    ms.conditional_vertex_selection(condselect='q == 1')

    # delete verts and connected faces
    ms.delete_selected_vertices()

    # extract
    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    print(f'[INFO] mesh mask verts: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}')

    return verts, faces


def remove_selected_verts(verts, faces, query='(x < 1) && (x > -1) && (y < 1) && (y > -1) && (z < 1 ) && (z > -1)'):

    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    m = pml.Mesh(verts, faces)
    ms = pml.MeshSet()
    ms.add_mesh(m, 'mesh') # will copy!

    # select verts
    ms.conditional_vertex_selection(condselect=query)

    # delete verts and connected faces
    ms.delete_selected_vertices()

    # extract
    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    print(f'[INFO] mesh remove verts: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}')

    return verts, faces

def merge_close_vertices(verts, faces, v_pct=1):
    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    m = pml.Mesh(verts, faces)
    ms = pml.MeshSet()
    ms.add_mesh(m, 'mesh')  # will copy!

    if v_pct > 0:
        ms.merge_close_vertices(threshold=pml.Percentage(v_pct)) # 1/10000 of bounding box diagonal

    # extract mesh
    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    print(f'[INFO] mesh cleaning: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}')

    return verts, faces


def clean_mesh(verts, faces, v_pct=1, min_f=8, min_d=5, repair=True, remesh=True):
    # verts: [N, 3]
    # faces: [N, 3]

    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    m = pml.Mesh(verts, faces)
    ms = pml.MeshSet()
    ms.add_mesh(m, 'mesh') # will copy!

    # filters
    ms.remove_unreferenced_vertices() # verts not refed by any faces
    
    if v_pct > 0:
        ms.merge_close_vertices(threshold=pml.Percentage(v_pct)) # 1/10000 of bounding box diagonal

    ms.remove_duplicate_faces() # faces defined by the same verts
    ms.remove_zero_area_faces() # faces with area == 0

    if min_d > 0:
        ms.remove_isolated_pieces_wrt_diameter(mincomponentdiag=pml.Percentage(min_d))
    
    if min_f > 0:
        ms.remove_isolated_pieces_wrt_face_num(mincomponentsize=min_f)

    if repair:
        # ms.meshing_remove_t_vertices(method=0, threshold=40, repeat=True)
        ms.repair_non_manifold_edges_by_removing_faces()
        ms.repair_non_manifold_vertices_by_splitting(vertdispratio=0)

    if remesh:
        # ms.apply_coord_taubin_smoothing()
        ms.remeshing_isotropic_explicit_remeshing(iterations=3, targetlen=pml.Percentage(1))

    # extract mesh
    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    print(f'[INFO] mesh cleaning: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}')

    return verts, faces


def clean_mesh2(verts, faces, v_pct=1, min_f=8, min_d=5, repair=True, remesh=True):
    # verts: [N, 3]
    # faces: [N, 3]

    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    m = pml.Mesh(verts, faces)
    ms = pml.MeshSet()
    ms.add_mesh(m, 'mesh')  # will copy!

    # filters
    ms.remove_unreferenced_vertices() # verts not refed by any faces

    if v_pct > 0:
        ms.merge_close_vertices(threshold=pml.Percentage(v_pct))  # 1/10000 of bounding box diagonal

    ms.remove_duplicate_faces() # faces defined by the same verts
    ms.remove_zero_area_faces() # faces with area == 0

    if min_d > 0:
        ms.remove_isolated_pieces_wrt_diameter(mincomponentdiag=pml.Percentage(min_d))

    if min_f > 0:
        ms.remove_isolated_pieces_wrt_face_num(mincomponentsize=min_f)

    if repair:
        # ms.meshing_remove_t_vertices(method=0, threshold=40, repeat=True)
        ms.repair_non_manifold_edges_by_removing_faces()
        ms.repair_non_manifold_vertices_by_splitting(vertdispratio=0)

    if remesh:
        # ms.apply_coord_taubin_smoothing()
        ms.remeshing_isotropic_explicit_remeshing(iterations=3, targetlen=pml.Percentage(1))

    # extract mesh
    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    print(f'[INFO] mesh cleaning: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}')

    return verts, faces



def decimate_and_refine_mesh(verts, faces, mask, decimate_ratio=0.1, refine_size=0.01, refine_remesh_size=0.02):
    # verts: [N, 3]
    # faces: [M, 3]
    # mask: [M], 0 denotes do nothing, 1 denotes decimation, 2 denotes subdivision

    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    m = pml.Mesh(verts, faces, f_quality_array=mask)
    ms = pml.MeshSet()
    ms.add_mesh(m, 'mesh') # will copy!

    # decimate and remesh
    ms.conditional_face_selection(condselect='fq == 1')
    if decimate_ratio > 0:
        ms.simplification_quadric_edge_collapse_decimation(targetfacenum=int((1 - decimate_ratio) * (mask == 1).sum()), selected=True)

    if refine_remesh_size > 0:
        ms.remeshing_isotropic_explicit_remeshing(iterations=3, targetlen=refine_remesh_size, selectedonly=True)

    # repair
    ms.select_none(allfaces=True)
    ms.repair_non_manifold_edges_by_removing_faces()
    ms.repair_non_manifold_vertices_by_splitting(vertdispratio=0)
    
    # refine 
    if refine_size > 0:
        ms.conditional_face_selection(condselect='fq == 2')
        ms.subdivision_surfaces_midpoint(threshold=refine_size, selected=True)

        # ms.remeshing_isotropic_explicit_remeshing(iterations=3, targetlen=pml.AbsoluteValue(refine_size), selectedonly=True)

    # extract mesh
    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    print(f'[INFO] mesh decimating & subdividing: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}')

    return verts, faces


# in meshutils.py
def select_bad_and_flat_faces_by_normal(verts, faces, usear=False, aratio=0.02, nfratio_bad=120, nfratio_flat=5):
    m = pml.Mesh(verts, faces)
    ms = pml.MeshSet()
    ms.add_mesh(m, 'mesh')

    ms.select_problematic_faces(usear=usear, aratio=aratio, usenf=True, nfratio=nfratio_bad)
    m = ms.current_mesh()
    bad_faces_mask = m.face_selection_array()
    # print('bad_faces_mask cnt: ', sum(bad_faces_mask * 1.0))
    ms.select_none(allfaces=True)

    ms.select_problematic_faces(usear=usear, aratio=aratio, usenf=True, nfratio=nfratio_flat)
    m = ms.current_mesh()
    flat_faces_mask = m.face_selection_array() == False  # reverse
    # print('flat_faces_mask cnt: ', sum(flat_faces_mask * 1.0))
    ms.select_none(allfaces=True)

    return bad_faces_mask, flat_faces_mask
