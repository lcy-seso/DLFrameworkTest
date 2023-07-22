from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='my_add',
    ext_modules=[
        CUDAExtension('my_add', [
            'my_add_kernel.cu',
        ]),
    ],
    cmdclass={'build_ext': BuildExtension})
