"""Setup.py script for the aixcalibuha-framework"""

import setuptools

# read the contents of your README file
from pathlib import Path
readme_path = Path(__file__).parent.joinpath("README.md")
long_description = readme_path.read_text()

INSTALL_REQUIRES = [
    'numpy>=1.19.5',
    'matplotlib>=3.3.4',
    'pandas>=1.3.5',
    'SALib>=1.4.6',
    'ebcpy>=0.3.14',
    'toml>=0.10.2'
]

__version__ = "1.0.1"

setuptools.setup(
    name='aixcalibuha',
    version=__version__,
    description='Framework used for sensitivity-analysis'
                'and calibration for models of HVAC '
                'components.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/RWTH-EBC/AixCaliBuHA',
    download_url=f'https://github.com/RWTH-EBC/AixCaliBuHA/archive/refs/tags/{__version__}.tar.gz',
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
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11'
    ],
    keywords=[
        'calibration', 'building', 'energy',
        'black-box optimization', 'sensitivity_analysis'
    ],
    packages=setuptools.find_packages(exclude=['tests', 'tests.*', 'img']),
    install_requires=INSTALL_REQUIRES
)
