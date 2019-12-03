.. AixCaliBuHa documentation master file, created by
   sphinx-quickstart on Thu Jul 11 08:20:35 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to AixCaliBuHa's documentation!
=======================================

**Aix** (from French Aix-la-Chapelle) **Cali**\bration for **Bu**\ ilding and **H**\V\ **A**\C Systems

This framework attempts to make the process of calibrating models used in Building
and HVAC Systems easier. Different sub-packages help with the underlying tasks of:

- **Preprocessing** measured or simulated data for the cohesive use in this framework
- **Segmentizing** the continuous time-series data into calibration classes
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

v.0.1:
- v.0.1.0: Implemented.
- v.0.1.1: Split into different frameworks and adjust changes from based on new version of ebcpy
- v.0.1.2: Move CalibrationClass from ebcpy and add it to the general module aixcalibuha. Adjust Goals etc. based on changes in ebcpy.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
