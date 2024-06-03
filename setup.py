from setuptools import find_packages, setup

setup(
    name="pytorch_retinaface",
    version="0.1",
    packages=find_packages(include=['pytorch_retinaface', 'pytorch_retinaface.*']),
    install_requires=[
        'torch',
        'numpy',
        'opencv-python',
        # Add other dependencies here
    ],
)
