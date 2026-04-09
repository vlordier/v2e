"""Setup script for v2e."""

from setuptools import find_packages, setup

version = "1.5.1"
package_name = "v2e"

setup(
    name=package_name,
    version=version,
    description="Generates synthetic DVS events from conventional video",
    author="Tobi Delbruck, Yuhuang Hu, Zhe He",
    author_email="yuhuang.hu@ini.uzh.ch, tobi@ini.uzh.ch",
    python_requires=">=3.10",
    packages=find_packages(),
    url="https://github.com/SensorsINI/v2e",
    install_requires=[
        "numpy>=1.24,<2.0",
        "argcomplete",
        "engineering-notation",
        "tqdm",
        "opencv-python",
        "h5py",
        "torch",
        "torchvision",
        "numba",
        "matplotlib",
        "plyer",
        "screeninfo",
        "easygui",
        "scikit-image",
        "dv-processing",
    ],
    scripts=["v2e.py", "dataset_scripts/ddd/ddd_extract_data.py"],
    entry_points={"console_scripts": ["v2e=v2e:main"]},
)
