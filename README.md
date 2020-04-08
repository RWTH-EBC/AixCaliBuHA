# AixCaliBuHA

**Aix** (from French Aix-la-Chapelle) 
**Cali**bration for **Bu**ilding and **H**V**A**C Systems

This framework attempts to make the process of calibrating models used in Building
and HVAC Systems easier. Different sub-packages help with the underlying tasks of:

# Key features
- Performing a **Sensitivity Analysis** to discover tuner parameters for the calibration
- **Calibration** of a given model based on the tuner parameters, the calibration classes and specified goals to evaluate the objective function of the underlying optimization


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


## Wiki on Calibration
We have gathered some information about Calibration and especially the underlying Optimization with focus on available frameworks automating task.
Please see our [Wiki]() for that. 


## Associates
- Philipp Mehrfeld, pmehrfeld@eonerc.rwth-aachen.de
- Thomas Storek, tstorek@eonerc.rwth-aachen.de
- David Wackerbauer, david.wackerbauer@eonerc.rwth-aachen.de
- Martin Rätz, mraetz@eonerc.rwth-aachen.de
- Fabian Wüllhorst, fabian.wuellhorst@eonerc.rwth-aachen.de
- Zhiyu Pan, zhiyu.pan@eonerc.rwth-aachen.de


