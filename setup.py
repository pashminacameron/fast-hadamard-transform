# Copyright (c) 2023, Tri Dao.
import sys
import warnings
import os
import re
import ast
from pathlib import Path
import platform

from setuptools import setup, find_packages

from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
    CUDA_HOME,
)


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

PACKAGE_NAME = "fast_hadamard_transform"

# FORCE_BUILD: Force a fresh build locally, instead of attempting to find prebuilt wheels
# SKIP_CUDA_BUILD: Intended to allow CI to use a simple `python setup.py sdist` run to copy over raw files, without any cuda compilation
FORCE_BUILD = "TRUE"
SKIP_CUDA_BUILD = "FALSE"
# For CI, we want the option to build with C++11 ABI since the nvcr images use C++11 ABI
FORCE_CXX11_ABI = "TRUE"


def get_platform():
    """
    Returns the platform name as used in wheel filenames.
    """
    if sys.platform.startswith("linux"):
        return "linux_x86_64"
    elif sys.platform == "darwin":
        mac_version = ".".join(platform.mac_ver()[0].split(".")[:2])
        return f"macosx_{mac_version}_x86_64"
    elif sys.platform == "win32":
        return "win_amd64"
    else:
        raise ValueError("Unsupported platform: {}".format(sys.platform))


def check_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is not None:
        return
    # warn instead of error because user could be downloading prebuilt wheels, so nvcc won't be necessary
    # in that case.
    warnings.warn(
        f"{global_option} was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc."
    )


def append_nvcc_threads(nvcc_extra_args):
    return nvcc_extra_args + ["--threads", "4"]


cmdclass = {}
ext_modules = []

if not SKIP_CUDA_BUILD:
    print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
    TORCH_MAJOR = int(torch.__version__.split(".")[0])
    TORCH_MINOR = int(torch.__version__.split(".")[1])

    check_if_cuda_home_none("fast_hadamard_transform")
    # Check, if CUDA11 is installed for compute capability 8.0
    cc_flag = []
    # if CUDA_HOME is not None:
    #     _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)

    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_70,code=sm_70")
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_80,code=sm_80")
    # assume this to be the case because that's the case in AzureML cuda 11.8 images
    # if bare_metal_version >= Version("11.8"):
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_90,code=sm_90")

    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI is set to be the same as
    # torch._C._GLIBCXX_USE_CXX11_ABI
    # https://github.com/pytorch/pytorch/blob/8472c24e3b5b60150096486616d98b7bea01500b/torch/utils/cpp_extension.py#L920
    if FORCE_CXX11_ABI:
        torch._C._GLIBCXX_USE_CXX11_ABI = True

    ext_modules.append(
        CUDAExtension(
            name="fast_hadamard_transform_cuda",
            sources=[
                "csrc/fast_hadamard_transform.cpp",
                "csrc/fast_hadamard_transform_cuda.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": append_nvcc_threads(
                    [
                        "-O3",
                        "-U__CUDA_NO_HALF_OPERATORS__",
                        "-U__CUDA_NO_HALF_CONVERSIONS__",
                        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                        "--expt-relaxed-constexpr",
                        "--expt-extended-lambda",
                        "--use_fast_math",
                        "--ptxas-options=-v",
                        "-lineinfo",
                    ]
                    + cc_flag
                ),
            },
            include_dirs=[this_dir],
        )
    )


def get_package_version():
    with open(Path(this_dir) / "fast_hadamard_transform" / "__init__.py", "r") as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    public_version = ast.literal_eval(version_match.group(1))
    local_version = os.environ.get("FAST_HADAMARD_TRANSFORM_LOCAL_VERSION")
    if local_version:
        return f"{public_version}+{local_version}"
    else:
        return str(public_version)


class CachedWheelsCommand(_bdist_wheel):
    """
    The CachedWheelsCommand plugs into the default bdist wheel, which is ran by pip when it cannot
    find an existing wheel (which is currently the case for all installs). We use
    the environment parameters to detect whether there is already a pre-built version of a compatible
    wheel available and short-circuits the standard full build pipeline.
    """

    def run(self):
        if FORCE_BUILD:
            return super().run()


setup(
    name=PACKAGE_NAME,
    version=get_package_version(),
    packages=find_packages(
        exclude=(
            "build",
            "csrc",
            "include",
            "tests",
            "dist",
            "docs",
            "benchmarks",
            "fast_hadamard_transform.egg-info",
        )
    ),
    author="Tri Dao",
    author_email="tri@tridao.me",
    description="Fast Hadamard Transform in CUDA, with a PyTorch interface",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Dao-AILab/fast-hadamard-transform",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    ext_modules=ext_modules,
    cmdclass={"bdist_wheel": CachedWheelsCommand, "build_ext": BuildExtension}
    if ext_modules
    else {
        "bdist_wheel": CachedWheelsCommand,
    },
    python_requires=">=3.7",
    install_requires=[
        "torch",
        "ninja",
    ],
)
