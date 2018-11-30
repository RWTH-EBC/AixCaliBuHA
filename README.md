Modelica Calibration
======

## How to install?

For installation use pip. Run `pip install -e "Path/to/this/repository"`

If environment variables are not set properly, try more explicit command in Windows shell:

`C:\Path\to\pythonDirectory\python.exe -c "import pip" & C:\Path\to\pythonDirectory\python.exe -m pip install -e C:\Path\to\this\repository`

Be aware of forward slashes (for python) and backslashes (for Windows). You might need to encompass paths in inverted commas (") in order to handle spaces.


## Important hints

1. The calabriation process changes the chosen tuner parameters for each simulation during the optimizer is running. It is strongly recommended to choose 
only paramters that are __not structural parameters__! It is possible, but each optimization step, the simulation model must be translated again (not just simulated with new paramter values). 
This __slows__ the process extremly down. Add in the Modelica code `annotation(Evaluate=false)` behind the parameters that are of interest for your calibration. However, not all parameters 
can be converted into non-structural ones (e.g. if an integer determines geometric informations like the number of layers in a tank). Also use the Dymola flag `Advanced.LogStructuredEvaluation = true` 
(p. 630 Dymola User Manual Volume 1) to receive further information in the log file and the translation tab.


## Associates
- Philipp Mehrfeld, pmehrfeld@eonerc.rwth-achen.de
- Marc Baranski, mbaranski@eonerc.rwth-achen.de


