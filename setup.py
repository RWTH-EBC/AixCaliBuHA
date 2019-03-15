from setuptools import setup

setup(name='ebccalibration',
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
      packages=['ebccalibration.calibration',
                'ebccalibration.calibration.examples',
                'ebccalibration.classifier',
                'ebccalibration.preprocessing'],
      setup_requires=['numpy',
					'scipy',
					'pandas',
					'modelicares',
					'matplotlib',
					'scikit-learn'],
      install_requires=['numpy',
					'scipy',
					'pandas',
					'modelicares',
					'matplotlib',
					'scikit-learn'])
# Addtional packages:
# ebcpython : https://git.rwth-aachen.de/EBC/EBC_all/Python/EBC_Python_Library/
# dymola : https://github.com/RWTH-EBC/AixLib/wiki/How-to:-Dymola-Python-Interface