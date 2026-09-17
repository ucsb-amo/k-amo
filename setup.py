from setuptools import setup, find_packages

setup(
    name='kamo',
    version='0.0.1',
    url='https://github.com/ucsb-amo/k-amo',
    author='Jared Pagett',
    author_email='pagett.jared@gmail.com',
    description='K team AMO functions, modeling, and simulations',
    packages=find_packages(),
    package_data={'kamo.light_shift': ['data/udel_portal/*.json'],
                  'kamo.hamiltonian': ['examples/*.ipynb'],
                  'kamo.trap': ['examples/*.ipynb'],
                  'kamo.imaging': ['examples/*.ipynb'],
                  'kamo.atom_properties': ['data/*.csv'],
                  'kamo.scattering': ['data/*.npz']},
    install_requires=['arc','numpy','pandas'],
)