from setuptools import setup, find_packages

setup(
    name="autograd",
    version="0.1.0",
    author="Ayush Singh",
    author_email="ayush.rocketeer@gmail.com",
    description="A simple autograd library with ValueNode and MLP",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24"
    ],
    python_requires='>=3.8',
)
