[![pylint](https://ebc.pages.rwth-aachen.de/EBC_all/Optimization-and-Calibration/AixCaliBuHA/master/pylint/pylint.svg )](https://ebc.pages.rwth-aachen.de/EBC_all/Optimization-and-Calibration/AixCaliBuHA/master/pylint/pylint.html)
[![coverage](https://ebc.pages.rwth-aachen.de/EBC_all/Optimization-and-Calibration/AixCaliBuHA/master/coverage.svg)](https://ebc.pages.rwth-aachen.de/EBC_all/Optimization-and-Calibration/AixCaliBuHA/master/coverage)

# AixCaliBuHA

**Aix** (from French Aix-la-Chapelle) 
**Cali**bration for **Bu**ilding and **H**V**A**C Systems

# Key features
* Calibration 
* Sensitivity Analysis
* Calibration Visualizer

# Installation
Basic knowlege about **git** and **python** are required to understand the following simple steps.  
We tested this with `cmd` on a *Windows* 10 machine.

Until this is not publically available, you have to install it (and [`ebcpy`](https://git.rwth-aachen.de/EBC/EBC_all/Python/ebcpy/-/blob/master/README.md)) via:
```
git clone https://git.rwth-aachen.de/EBC/EBC_all/Python/ebcpy
pip install -e ebcpy
git clone https://git.rwth-aachen.de/EBC/EBC_all/Optimization-and-Calibration/AixCaliBuHA
pip install -e AixCaliBuHA
```

## How to get started?
We recommend running our jupyter-notebook to be guided through a **helpful tutorial**.  
For this, run the following code:
```
# If jupyter is not already installed:
pip install jupyter
# Go into your ebcpy-folder (cd \path_to_\AixCaliBuHA) or change the path to tutorial.ipynb and run:
jupyter notebook AixCaliBuHA\examples\tutorial.ipynb
```

# Documentation
Visit hour official [Documentation](https://ebc.pages.rwth-aachen.de/EBC_all/Optimization-and-Calibration/AixCaliBuHA/master/docs).

# Problems?
Please [raise an issue here](https://git.rwth-aachen.de/EBC/EBC_all/Python/ebcpy/-/issues/new?issue%5Bassignee_id%5D=&issue%5Bmilestone_id%5D=).

## Important hints

### Framework structure
(Information still up to date?)

Adhere to the following UML diagram as overall structure!

Open the [*.xml file](https://git.rwth-aachen.de/EBC/EBC_intern/modelica-calibration/blob/master/img/Calibration_Framework_EBC.xml) (download an load from local drive) with the online plattform [draw.io](draw.io).

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


## Associates
- Philipp Mehrfeld, pmehrfeld@eonerc.rwth-aachen.de
- Thomas Storek, tstorek@eonerc.rwth-aachen.de
- David Wackerbauer, david.wackerbauer@eonerc.rwth-aachen.de
- Martin Rätz, mraetz@eonerc.rwth-aachen.de
- Fabian Wüllhorst, fabian.wuellhorst@eonerc.rwth-aachen.de
- Zhiyu Pan, zhiyu.pan@eonerc.rwth-aachen.de


