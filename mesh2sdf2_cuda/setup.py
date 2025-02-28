## Adapted from https://github.com/zekunhao1995/DualSDF/tree/master/extensions/mesh2sdf2_cuda

import os
import torch

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cxx_args = ['-std=c++11', '-ffast-math']

include_dirs = []
library_dirs = []

nvcc_args = [
    '-gencode', 'arch=compute_70,code=sm_70',
    '-gencode', 'arch=compute_80,code=sm_80',
    '-gencode', 'arch=compute_90,code=sm_90',
]

setup(
    name='mesh2sdf',
    ext_modules=[
        CUDAExtension('mesh2sdf', [
            'mesh2sdf_kernel.cu'
        ],
        include_dirs = include_dirs,
        library_dirs = library_dirs,
        extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args})
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
    
