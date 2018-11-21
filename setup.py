from setuptools import setup

setup(name='ebccalibration',
      version='0.1',
      description='Calibration helper',
      url='not set yet',
      author='RWTH Aachen University, E.ON Energy Research Center, Institute\
      of Energy Efficient Buildings and Indoor Climate',
      packages=['calibration',
                'calibration.examples',
                'classifier',
                'preprocessing'],
      setup_requires=['os',
					'sys',
					'numpy',
					'scipy',
					'subprocess',
					'multiprocessing',
					'pandas',
					'modelicares',
					'matplotlib',
					'sklearn'],
# Addtional packages:
# ebcpython : https://git.rwth-aachen.de/EBC/EBC_all/Python/EBC_Python_Library/
# dymola : https://github.com/RWTH-EBC/AixLib/wiki/How-to:-Dymola-Python-Interface
      install_requires=['os',
					'sys',
					'numpy',
					'scipy',
					'subprocess',
					'multiprocessing',
					'pandas',
					'modelicares',
					'matplotlib',
					'sklearn'])
