# Examples

This folder contains several example files which help with the understanding of AixCaliBuHA.

# Getting started

While these examples should run in any IDE, we advise using PyCharm.
Before being able to run these examples, be sure to:

1. Create a clean environment of python 3.7 or 3.8. In Anaconda run: `conda create -n py38_ebcpy python=3.8`
2. Activate the environment in your terminal. In Anaconda run: `activate py38_ebcpy` 
3. Clone the repository by running `git clone https://git.rwth-aachen.de/EBC/EBC_all/Python/ebcpy`
4. Clone the AixLib in order to use the models: `git clone https://github.com/RWTH-EBC/AixLib`
   Also check if you're on development using `cd AixLib && git status && cd ..`
5. Install the library using `pip install -e ebcpy`

# What can I learn in the examples?

## `e1_energy_system_analysis.py`

1. Learn how to analyze your energy system
2. Improve your `DymolaAPI` knowledge
3. Improve your skill-set on `TimeSeriesData`

## `e2_1_optimization_problem_definition.py`

1. Learn how to formulate your calibration problem using our data_types
2. Get to know `TunerParas`
3. Get to know `Goals`
4. Get to know `CalibrationClass`
5. Learn how to merge multiple classes

## `e2_2_optimization_problem_definition.py`

1. Deepen your understanding of data_types in AixCaliBuHA

## `e3_sensitivity_analysis_example.py`

1. Learn how to execute a sensitivity analysis
2. Learn how to automatically select sensitive tuner parameters

## `e4_calibration_example.py`

1. Learn the settings for a calibration
2. Learn how to use both Single- and MultiClassCalibration
3. Learn how to validate your calibration

## `e5_automated_process.py`

1. Learn how to run everything in one script