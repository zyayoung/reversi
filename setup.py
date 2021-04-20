from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='reversi_layer_cpp',
    ext_modules=[
        CppExtension('reversi_layer_cpp', ['reversi_layer.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })