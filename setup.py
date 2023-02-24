from setuptools import setup, find_packages

setup(
    name='kamo',
    version='0.0.1',
    url='https://github.com/ucsb-amo/k-amo',
    author='Jared Pagett',
    author_email='pagett.jared@gmail.com',
    description='K team AMO functions, modeling, and simulations',
    packages=find_packages(),
    install_requires=['arc','numpy'],
)