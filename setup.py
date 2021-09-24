"""Setup.py script for the aixcalibuha-framework"""

import setuptools

INSTALL_REQUIRES = [
    'numpy>=1.19.5',
    'matplotlib>=3.3.4',
    'pandas>=1.1.5',
    'SALib>=1.3.12',
    'ebcpy>=0.2.2',
    'toml>=0.10.2'
]

setuptools.setup(
    name='aixcalibuha',
    version='0.2.1',
    description='Framework used for sensitivity-analysis'
                'and calibration for models of HVAC '
                'components.',
    url='https://github.com/RWTH-EBC/AixCaliBuHA',
    download_url='https://github.com/RWTH-EBC/AixCaliBuHA/archive/refs/tags/0.2.1.tar.gz',
    license='MIT',
    author='RWTH Aachen University, E.ON Energy Research Center, Institute '
           'of Energy Efficient Buildings and Indoor Climate',
    author_email='fabian.wuellhorst@eonerc.rwth-aachen.de',
    # Specify the Python versions you support here. In particular, ensure
    # that you indicate whether you support Python 2, Python 3 or both.
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8'
    ],
    keywords=[
        'calibration', 'building', 'energy',
        'black-box optimization', 'sensitivity_analysis'
    ],
    packages=setuptools.find_packages(exclude=['img']),
    install_requires=INSTALL_REQUIRES
)
