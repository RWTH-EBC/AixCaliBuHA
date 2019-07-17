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
- **Segmentizing** the continuous time-series data into :ref:`calibration classes <calibration-class>`
- Performing a **Sensitivity Analysis** to discover :ref:`tuner parameters <tuner-parameter>` for the calibration
- **Calibration** of given model based on the :ref:`tuner parameters <tuner-parameter>`, the :ref:`calibration classes <calibration-class>` and specified :ref:`goals <goals>` to evaluate the objective function of the underlying optimization


Installation
-------------------

For installation use pip. Run `pip install -e "Path/to/this/repository"`

If environment variables are not set properly, try more explicit command in Windows shell:

`C:\Path\to\pythonDirectory\python.exe -c "import pip" & C:\Path\to\pythonDirectory\python.exe -m pip install -e C:\Path\to\this\repository`

Be aware of forward slashes (for python) and backslashes (for Windows). You might need to encompass paths in inverted commas (") in order to handle spaces.

**Note:** This package uses a custom modelicares. To have the costum version automatically installed,
run `pip uninstall modelicares"` before installing aixcal.


.. toctree::
   :maxdepth: 2

   tutorial
   data_types
   preprocessor
   segmentizer
   senanalyzer
   optimizer
   simulationapi


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
