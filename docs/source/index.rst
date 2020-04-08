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

.. _calibration-class:
CalibrationClass
------------------

Last but not least, a calibration class is an object wrapping the most important information for a calibration into one class.
The **Tuner parameters** and **Goals** are members, as well as the time-interval for the simulation and the name of the class.


.. autoclass:: aixcalibuha.data_types.CalibrationClass
   :members:

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

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
