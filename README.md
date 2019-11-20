# AixCaliBuHA

**Aix** (from French Aix-la-Chapelle) 
**Cali**bration for **Bu**ilding and **H**V**A**C Systems

## Work in Progress

### Link dependencies

Currently, this repo represents some kind of "master/overall" repo. Therefore, all related ongoing projects are linked here:

*  [Tool DyOS](http://www.avt.rwth-aachen.de/cms/AVT/Forschung/Software/~iptr/DyOS/) from AVT, RWTH Aachen University
*  `Design` Library included in Dymola (See also Dymola User Manual Volume 2)
*  [`Optimization` Library](https://www.modelica.org/libraries) (DLR-SR), s. help in [EBC-SharePoint](https://ecampus.rwth-aachen.de/units/eonerc/ebc/Wiki/Optimierung%20mit%20Dymola.aspx)
*  [GenOpt](https://simulationresearch.lbl.gov/GO/): Quotes:
   * "minimization of a cost function that is evaluated by an external simulation program"
   * "developed for optimization problems where the cost function is computationally expensive and its derivatives are not available"
   * "ocal and global multi-dimensional and one-dimensional optimization algorithms"
   * "parallel computation is done automatically"
   * "written in Java so that it is platform independent"
   * "GenOpt **has not been** designed for linear programming problems, ..."

## How to install?

**Note:** This repository works with a fork of the pypi-project [modelicares](https://github.com/kdavies4/ModelicaRes) ([our fork](https://github.com/RWTH-EBC/ModelicaRes)). To automatically 
install the newest version of the fork together with this repository, you have to options.

If you don't have modelicares installed, just go ahead and run the line below to install this repository:

`pip install -e "Path/to/this/repository"`

If you have modelicares installed, you have two options:

### 1. Uninstall modelicares first (fastest method) 

Run

`pip uninstall modelicares`

to uninstall modelicares and then run

`pip install -e "Path/to/this/repository"`

to install the repository.

### 2. Force reinstall

`pip install -e "Path/to/this/repository" --force-reinstall modelicares`

However, this method also re-installs all packages required for modelicares, such as numpy, scipy etc. This takes quite some time.

If environment variables are not set properly, try more explicit command in Windows shell:

`C:\Path\to\pythonDirectory\python.exe -c "import pip" & C:\Path\to\pythonDirectory\python.exe -m pip install -e C:\Path\to\this\repository`

Be aware of forward slashes (for python) and backslashes (for Windows). You might need to encompass paths in inverted commas (") in order to handle spaces.


## Important hints

### Framework structure
Adhere to the following UML diagram as overall structure!

Open the [*.xml file](https://git.rwth-aachen.de/EBC/EBC_intern/modelica-calibration/blob/master/img/Calibration_Framework_EBC.xml) (download an load from local drive) with the online plattform [draw.io](draw.io).


## Associates
- Philipp Mehrfeld, pmehrfeld@eonerc.rwth-aachen.de
- Thomas Storek, tstorek@eonerc.rwth-aachen.de
- David Wackerbauer, david.wackerbauer@eonerc.rwth-aachen.de
- Martin Rätz, mraetz@eonerc.rwth-aachen.de
- Fabian Wüllhorst, fabian.wuellhorst@eonerc.rwth-aachen.de
- Zhiyu Pan, zhiyu.pan@eonerc.rwth-aachen.de


