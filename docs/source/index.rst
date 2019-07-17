.. AixCaliBuHa documentation master file, created by
   sphinx-quickstart on Thu Jul 11 08:20:35 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to AixCaliBuHa's documentation!
=======================================

**Aix** (from French Aix-la-Chapelle) **Cali**\bration for **Bu**\ ilding and **H**\V\ **A**\C Systems

This framework is used to help one in the calibration of models used in Building
and HVAC Systems. Different sub-packages help with the underlying tasks of

- **Preprocessing** measured or simulated data for the usage in calibration
- **Segmentizing** the continuous time-series data into *calibration classes*
- Performing a **Sensitivity Analysis** to discover :meth:`tuner parameters <aixcal.data_types.TunerParas>` for the calibration
- **Calibration** of given model based on the *tuner parameters*, the *calibration classes* and specified *goals* to evaluate the objective function of the underlying optimization


.. toctree::
   :maxdepth: 2

   tutorial
   aixcal


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
