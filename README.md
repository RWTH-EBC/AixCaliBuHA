# AixCaliBuHA

**Aix** (from French Aix-la-Chapelle) 
**Cali**bration for **Bu**ilding and **H**V**A**C Systems

# Key features
* Calibration 
* Sensitivity Analysis
* Calibration Visualizer

# Installation
Until this is open source, you have to install it (and `ebcpy`) via:
```
git clone https://git.rwth-aachen.de/EBC/EBC_all/Python/ebcpy
pip install -e ebcpy
git clone https://git.rwth-aachen.de/EBC/EBC_all/Optimization-and-Calibration/AixCaliBuHA
pip install -e AixCaliBuHA
```
You may switch branches to `development` for newly available features.

## How to get started?
We recommend running our jupyter-notebook. For this, run the following code:
```
# If not jupyter is not already installed:
pip install jupyter
# Go into your ebcpy-folder (cd \path_to_\AixCaliBuHA) or change the path to tutorial.ipynb and run:
jupyter notebook AixCaliBuHA\examples\tutorial.ipynb
```
If you have any questions, please contact us or raise an issue.
Additionally we refer to the official [Documentation](#Documentation).


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


