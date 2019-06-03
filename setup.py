"""Setup.py script for the aixcal-framework"""

import setuptools
import pip
from packaging import version

install_requires = ['numpy',
                    'scipy',
                    'pandas',
                    'matplotlib',
                    'scikit-learn',
                    'h5py',
                    'SALib']
setup_requires = install_requires.copy()  # Add all open-source packages to setup-requires

if version.parse(pip.__version__) > version.parse('18.1'):
    install_requires.append('modelicares @ git+https://github.com/RWTH-EBC/ModelicaRes@v_0_12_2')
else:
    raise ImportError("You have to upgrade your pip version "
                      "to >=18.1 for modelicares to be installed.")

setuptools.setup(name='aixcal',
                 version='0.1',
                 description='Framework used for classification, sensitivity-analysis'
                             'and optimization or calibration for models of HVAC '
                             'components.',
                 url='not set yet',
                 author='RWTH Aachen University, E.ON Energy Research Center, Institute\
                 of Energy Efficient Buildings and Indoor Climate',
                 # Specify the Python versions you support here. In particular, ensure
                 # that you indicate whether you support Python 2, Python 3 or both.
                 classifiers=['Programming Language :: Python :: 3.5',
                              'Programming Language :: Python :: 3.6',
                              'Programming Language :: Python :: 3.7', ],
                 packages=setuptools.find_packages(exclude=['img']),
                 setup_requires=setup_requires,
                 install_requires=install_requires)

# Additional packages:
# Dymola : https://github.com/RWTH-EBC/AixLib/wiki/How-to:-Dymola-Python-Interface
