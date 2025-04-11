import trimesh

import numpy as np

import point_cloud_utils as pcu

N_vol = 250000
N_near = 125000

from pathlib import Path


def process(model_filename, save_filename):


    folder = Path(save_filename).parent.absolute()
    folder.mkdir(parents=True, exist_ok=True)

    if Path(save_filename).is_file():
        return
    else:
        print(save_filename)
        
    try:
        mesh = trimesh.load(model_filename, skip_materials=True, process=True, force='mesh')
        print('loading successfully', model_filename)
    except Exception as exc: 
        print(exc)
        return

    v, f = mesh.vertices, mesh.faces
    resolution = 50_000
    

    try:
        print('watertight', model_filename)
        vw, fw = pcu.make_mesh_watertight(v, f, resolution, seed=0)
        print('success', model_filename)

    except Exception as e:
        print(e)
        print('watertight failed', model_filename)
        return

    shifts = (vw.max(axis=0) + vw.min(axis=0)) / 2
    vw = vw - shifts
    distances = np.linalg.norm(vw, axis=1)
    scale = 1 / np.max(distances)
    vw *= scale
        
    
    fid, bc = pcu.sample_mesh_random(vw, fw, N_near)
    surface_points = pcu.interpolate_barycentric_coords(fw, fid, bc, vw)

    vol_points = np.random.randn(N_vol, 3)
    vol_points = vol_points / np.linalg.norm(vol_points, axis=1)[:, None] * np.sqrt(3)
    vol_points = vol_points * np.power(np.random.rand(N_vol), 1. / 3)[:, None]

    vol_sdf, _, _ = pcu.signed_distance_to_mesh(vol_points, vw, fw)

    near_points = [
        surface_points + np.random.normal(scale=0.005, size=(N_near, 3)),
        surface_points + np.random.normal(scale=0.05, size=(N_near, 3)),
    ]
    near_points = np.concatenate(near_points)
    near_sdf, _, _ = pcu.signed_distance_to_mesh(near_points, vw, fw)

    np.savez(
            save_filename, 
            shifts=shifts,
            scale=scale,
            vol_points=vol_points.astype(np.float32),
            vol_sdf=vol_sdf.astype(np.float32), 
            near_points=near_points.astype(np.float32), 
            near_sdf=near_sdf.astype(np.float32), 
            surface_points=surface_points.astype(np.float32),
    )