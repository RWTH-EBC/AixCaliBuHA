"""Setup.py script for the aixcalibuha-framework"""

import setuptools

INSTALL_REQUIRES = ['numpy',
                    'scipy',
                    'pandas',
                    'matplotlib',
                    'SALib==1.3.12',
                    'ebcpy>=0.2.0',
                    'toml'
                    ]
SETUP_REQUIRES = INSTALL_REQUIRES.copy()  # Add all open-source packages to setup-requires

setuptools.setup(name='aixcalibuha',
                 version='0.2.0',
                 description='Framework used for sensitivity-analysis'
                             'and calibration for models of HVAC '
                             'components.',
                 url='not set yet',
                 author='RWTH Aachen University, E.ON Energy Research Center, Institute\
                 of Energy Efficient Buildings and Indoor Climate',
                 # Specify the Python versions you support here. In particular, ensure
                 # that you indicate whether you support Python 2, Python 3 or both.
                 classifiers=['Programming Language :: Python :: 3.6',
                              'Programming Language :: Python :: 3.7', ],
                 packages=setuptools.find_packages(exclude=['img']),
                 setup_requires=SETUP_REQUIRES,
                 install_requires=INSTALL_REQUIRES,
                 entry_points={
                     'console_scripts': ['modelica_calibration=bin.run_modelica_calibration:main',
                                         'guided_setup=bin.guided_setup:main'],
                 }
                 )
