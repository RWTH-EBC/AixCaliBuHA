.. AixCaliBuHa documentation master file, created by
   sphinx-quickstart on Thu Jul 11 08:20:35 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to AixCaliBuHa's documentation!
=======================================

**Aix** (from French Aix-la-Chapelle) **Cali**\bration for **Bu**\ ilding and **H**\V\ **A**\C Systems

This framework attempts to make the process of calibrating models used in Building
and HVAC Systems easier. Different sub-packages help with the underlying tasks of:

- Performing a **Sensitivity Analysis** to discover tuner parameters for the calibration
- **Calibration** of given model based on the tuner parameters, the calibration classes and specified goals to evaluate the objective function of the underlying optimization


Installation
-------------------

For installation use pip. Run ``pip install -e "Path/to/this/repository"``

If environment variables are not set properly, try more explicit command in Windows shell:

``C:\Path\to\pythonDirectory\python.exe -c "import pip" & C:\Path\to\pythonDirectory\python.exe -m pip install -e C:\Path\to\this\repository``

Be aware of forward slashes (for python) and backslashes (for Windows). You might need to encompass paths in inverted commas (") in order to handle spaces.

**Note:** This package uses a custom modelicares. To have the costum version automatically installed,
run ``pip uninstall modelicares"`` before installing aixcal.


.. toctree::
   :maxdepth: 2

   tutorial
   senanalyzer
   calibration


Version History
---------------

**v.0.1**:

- **v0.1.0**: Implemented.
- **v0.1.1**: Split into different frameworks and adjust changes from based on new version of ebcpy
- **v0.1.2**: Move CalibrationClass from ebcpy and add it to the general module aixcalibuha. Adjust Goals etc. based on changes in ebcpy.
- **v0.1.3**: Remove Continuous Calibration methods and introduce new, better methods for calibration of multiple classes.

   - Issue 43: Same class now optimizes to one optimum instead of multiple. If an intersection in tuner parameters occurs, the statistics are logged and plotted so the user can better decide with what values to go on.
   - Issue 42: Visualizer is adjusted to better print the results more readable
   - Issue 39: Several kwargs are added for better user-interaction and plotting of multiple classes
   - Issue 46: Current best iterate is stored to ensure an interruption of a calibration won't yield in a lost optimized value. Keyboard interrupt is now possible.

- **v0.1.4**
   - Add Goals from ebcpy
   - Add new tutorial for a better start with the framework. (See Issue 49)
   - Make changes based on new version 0.1.5 in ebcpy

- **v0.1.5**
   - Add new scripts in bin folder to ease the setup of the calibration for new users
   - Add configuration files and save/load classes
   - Issue 54: Skip failed simulations using two new kwargs in ModelicaCalibrator class
   - Issue 53: Save final plots despite abortion of calibration process via STRG+C
   - Issue 51: Refactor reference_start_time to fix_start_time
   - Issue 23: Model Wrapper for MoCaTe files.

- **v0.1.6**
   - Add Re-Calibration code from master thesis of Sebastian Borges
   - Add fixed_parameters to calibration
   - Re-add tunerParas from ebcpy
   - Make changes based on ebcpy v.0.1.7
   - Split SensivitiyAnalyzer class and use object oriented programming


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
