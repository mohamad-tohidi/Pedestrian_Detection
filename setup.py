from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np  # <-- Import numpy

# Define the extension module
ext_modules = [
    Extension(
        "processing_cython",
        ["processing_cython.pyx"],
        include_dirs=[np.get_include()]  # <-- Add numpy includes here
    )
]

setup(
    ext_modules=cythonize(ext_modules)
)
