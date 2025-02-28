
# How to process SDF data for 3D generative models

This is simplest process I found out for SDF data processing, which is a necessary step for 3D generative models. It does not need explicit watertight conversion. My projects [3DILG](https://github.com/1zb/3DILG), [3DShape2VecSet](https://github.com/1zb/3DShape2VecSet), [Functional Diffusion](https://1zb.github.io/functional-diffusion/), and [LaGeM](https://1zb.github.io/LaGeM) are based on the code.

## :earth_asia: Environment Setup
You can skip some steps if you already have installed some packages.

```
# install pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# install nvcc
conda install cuda-nvcc=12.4 -c nvidia

# install necessary develop packages
conda install libcusparse-dev
conda install libcublas-dev
conda install libcusolver-dev

# install mesh processing packages (you can use either one)
pip install point_cloud_utils
pip install trimesh

# compile the cuda code (adapted from DualSDF)
cd mesh2sdf2_cuda
python setup.py install
```

## :floppy_disk: Mesh loading

The code snippet loads the mesh vertices and faces into `v` and `f`,
```python
import trimesh

mesh = trimesh.load(mesh_path, skip_materials=True, process=False, force='mesh')
v = mesh.vertices
f = mesh.faces
```

## :pencil: Mesh normalization
First, we need to normalize meshes consistently. You can either use the sphere normalization,
```python
shifts = (v.max(axis=0) + v.min(axis=0)) / 2
v = v - shifts
distances = np.linalg.norm(v, axis=1)
scale = 1 / np.max(distances)
v *= scale
```
or box normalization ([-1, 1]),

```python
shifts = (v.max(axis=0) + v.min(axis=0)) / 2
v = v - shifts
scale = (1 / np.abs(v).max()) * 0.99
v *= scale
```

I use box normalization in [3DILG](https://github.com/1zb/3DILG), [3DShape2VecSet](https://github.com/1zb/3DShape2VecSet), and [Functional Diffusion](https://1zb.github.io/functional-diffusion/). In my latest work [LaGeM](https://1zb.github.io/LaGeM), I use sphere normalization.

## :hammer: Processing

In this section, we will process the meshes. There are multiple type of points, including,

- Surface points: sampled on the mesh surfaces.
- Labeled points: sampled within the bounding volume.
    + Volume points: uniformly sampled in the bounding volume.
    + Near-surface points: sampled in the near-surface region. They are obtained by jittering surface points.

### We begin by determining how many points to sample.

```python
N_vol = 250000 # volume points
N_near = 125000 # near-surface points
```

### Surface points sampling.
```python
import point_cloud_utils as pcu

fid, bc = pcu.sample_mesh_random(v, f, N_near)
surface_points = pcu.interpolate_barycentric_coords(f, fid, bc, v) # N_near x 3
```

### Volume points sampling.

If we are using box nomralization,
```python
vol_points = np.random.rand(N_vol, 3) * 2 - 1
```
If we are using sphere normalization,
```python
vol_points = np.random.randn(N_vol, 3)
vol_points = vol_points / np.linalg.norm(vol_points, axis=1)[:, None] * np.sqrt(3)
vol_points = vol_points * np.power(np.random.rand(N_vol), 1. / 3)[:, None]
```

### Near-surface points sampling.
```python
near_points = [
    surface_points + np.random.normal(scale=0.005, size=(N_near, 3)),
    surface_points + np.random.normal(scale=0.05, size=(N_near, 3)),
]
near_points = np.concatenate(near_points)
```

### Calculation of signed distances
We transfer the mesh data to the GPU (using CUDA)
```python
v = torch.from_numpy(v).float().cuda()
f = torch.from_numpy(f).cuda()
mesh = v[f]
```
The package `mesh2sdf` is adapted from [DualSDF](https://github.com/zekunhao1995/DualSDF).
```python
import mesh2sdf

vol_points = torch.from_numpy(vol_points).float().cuda()
vol_sdf = mesh2sdf.mesh2sdf_gpu(vol_points, mesh_t)[0].cpu().numpy()

near_points = torch.from_numpy(near_points).float().cuda()
near_sdf = mesh2sdf.mesh2sdf_gpu(near_points, mesh_t)[0].cpu().numpy()
```

### Save data
```python
np.savez(
    save_filename, 
    shifts=shifts,
    scale=scale,
    vol_points=vol_points.cpu().numpy().astype(np.float32),
    vol_sdf=vol_sdf.astype(np.float32), 
    near_points=near_points.cpu().numpy().astype(np.float32), 
    near_sdf=near_sdf.astype(np.float32), 
    surface_points=surface_points.astype(np.float32),
)
```

## :blue_book: Citation

If you use the code in your projects, consider citing the related papers,
```bibtex
@inproceedings{Biao_2022_3DILG,
author = {Zhang, Biao and Nie\ss{}ner, Matthias and Wonka, Peter},
title = {{3DILG}: irregular latent grids for 3D generative modeling},
year = {2022},
isbn = {9781713871088},
publisher = {Curran Associates Inc.},
address = {Red Hook, NY, USA},
booktitle = {Proceedings of the 36th International Conference on Neural Information Processing Systems},
articleno = {1590},
numpages = {15},
location = {New Orleans, LA, USA},
series = {NIPS '22}
}
```

```bibtex
@article{Biao_2023_VecSet,
author = {Zhang, Biao and Tang, Jiapeng and Nie\ss{}ner, Matthias and Wonka, Peter},
title = {{3DShape2VecSet}: A 3D Shape Representation for Neural Fields and Generative Diffusion Models},
year = {2023},
issue_date = {August 2023},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {42},
number = {4},
issn = {0730-0301},
url = {https://doi.org/10.1145/3592442},
doi = {10.1145/3592442},
journal = {ACM Trans. Graph.},
month = jul,
articleno = {92},
numpages = {16},
keywords = {3D shape generation, 3D shape representation, diffusion models, shape reconstruction, generative models}
}
```

```bibtex
@InProceedings{Biao_2024_Functional,
    author    = {Zhang, Biao and Wonka, Peter},
    title     = {Functional Diffusion},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {4723-4732}
}
```

```bibtex
@inproceedings{Biao_2024_LaGeM,
title={{LaGeM}: A Large Geometry Model for 3D Representation Learning and Diffusion},
author={Biao Zhang and Peter Wonka},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=72OSO38a2z}
}
```
