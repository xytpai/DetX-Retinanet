from setuptools import setup
import os
import glob
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='anchor_gen_cuda',
    ext_modules=[
        CUDAExtension(
            'anchor_gen_cuda', 
            ['anchor_gen_cuda.cpp', 'anchor_gen_kernel.cu'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
})
