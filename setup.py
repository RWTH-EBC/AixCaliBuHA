import setuptools
import pip
from packaging import version

install_requires = ['numpy',
					'scipy',
					'pandas',
					'modelicares',
					'matplotlib',
					'scikit-learn']
setup_requires = install_requires.copy() #Add all open-source packages to setup-requires
if version.parse(pip.__version__) > version.parse('18.1'):
    install_requires.append('ebcpython @ git+https://git.rwth-aachen.de/EBC/EBC_all/Python/EBC_Python_Library@master')
else:
    raise ImportError("You have to upgrade your pip version to >=18.1 for ebcpython to be installed.")

setuptools.setup(name='ebccalibration',
      version='0.1',
      description='Calibration helper',
      url='not set yet',
      author='RWTH Aachen University, E.ON Energy Research Center, Institute\
      of Energy Efficient Buildings and Indoor Climate',
      classifiers=[
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
      ],
      packages=setuptools.find_packages(exclude=['img', 'ebccalibration.*', 'ebccalibration']),
      setup_requires=setup_requires,
      install_requires=install_requires)
# Addtional packages:
# dymola : https://github.com/RWTH-EBC/AixLib/wiki/How-to:-Dymola-Python-Interface