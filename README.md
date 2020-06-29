[![pylint](https://ebc.pages.rwth-aachen.de/EBC_all/Optimization-and-Calibration/AixCaliBuHA/master/pylint/pylint.svg)](https://ebc.pages.rwth-aachen.de/EBC_all/Optimization-and-Calibration/AixCaliBuHA/master/pylint/pylint.html)
[![documentation](https://ebc.pages.rwth-aachen.de/EBC_all/Optimization-and-Calibration/AixCaliBuHA/master/docs/doc.svg)](https://ebc.pages.rwth-aachen.de/EBC_all/Optimization-and-Calibration/AixCaliBuHA/master/docs/index.html)
[![coverage](https://ebc.pages.rwth-aachen.de/EBC_all/Optimization-and-Calibration/AixCaliBuHA/master/coverage/badge.svg)](https://ebc.pages.rwth-aachen.de/EBC_all/Optimization-and-Calibration/AixCaliBuHA/master/coverage)


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
git clone --recurse-submodules https://git.rwth-aachen.de/EBC/EBC_all/Optimization-and-Calibration/AixCaliBuHA
pip install -e AixCaliBuHA
```

# How to get started?
We differ this section into two parts. How to get started with the theory of calibration and how to get started with using this repo.

## How can I calibrate my model?
While we aim at automating most parts of a calibration process, you still have to specify the inputs and the methods you want to use.
We therefore recommend to:
1. Analyze the physical system and theoretical model you want to calibrate
2. Identify inputs and outputs of the system and model
3. Get to know your tuner parameters and how they affect your model
4. Plan your experiments and perform them
5. Learn about the methods provided for calibration (statistical measures (RMSE, etc.), optimization, ...)
6. **Always be critical** about the results of the process. If the model approach or the experiment is faulty, the calibration will perform accordingly. 

## How to start with AixCaliBuHa?
We have two services in place to help you with the setup of `AixCaliBuHa`. For the basics on using this repo, we recommend the Jupyter Notebook.
If you want to setup your calibration models (in Modelica) and quickly start your first calibration, we provide a guided setup.
### Jupyter Notebook
We recommend running our jupyter-notebook to be guided through a **helpful tutorial**.  
For this, run the following code:
```
# If jupyter is not already installed:
pip install jupyter
# Go into your ebcpy-folder (cd \path_to_\AixCaliBuHA) or change the to the absolute path of the tutorial.ipynb and run:
jupyter notebook AixCaliBuHA\examples\tutorial.ipynb
```
### Guided Setup
After installation, you can run (in your anaconda prompt or shell with the python path set):

**Note**: You will need some Modelica model to calibrate and measured data from experiments for inputs and targets of the model you want to calibrate..

```cmd
guided_setup
```
This will trigger a script to help you through the process. Be aware that this was tested for Windows where Dymola is installed in the default location.
After completing the steps, a configuration file is generated with most information already present. You have to specify some additional data in the generated config.
Afterwards, you can run
```cmd
run_modelica_calibration --config=my_generated_config.yaml
```
To give you an idea of what you have to do and where to click, here is a little example.
![Alt Text](img/guided_setup.gif)


### TimeSeriesData
Note that we use our own `TimeSeriesData` object which inherits from `pd.DataFrame`. The aim is to make tasks like loading different filetypes or applying multiple tags to one variable more convenient, while conserving the powerful tools of the DataFrame.
The class is defined in `ebcpy`, and you can also check the documentation over there. Just a quick intro here:

#### Variables and tags
```
>>> from ebcpy.data_types import TimeSeriesData
>>> tsd = TimeSeriesData(r"path_to_a_supported_file")
>>> print(tsd)
Variables    T_heater              T_heater_1            
Tags             meas         sim        meas         sim
Time                                                     
0.0        313.165863  313.165863  293.173126  293.173126
1.0        312.090271  310.787750  293.233002  293.352448
2.0        312.090027  310.796753  293.385925  293.719055
3.0        312.109436  310.870331  293.589233  294.141754
```

As you can see, our first column level is always a variable, and the second one a tag.
This is especially handy when dealing with calibration or processing tasks, where you will have multiple
versions (tags) for one variable. The default tag is `raw` to indicate the unmodified data.
#### FloatIndex and DateTimeIndex
Measured data typically holds a datetime stamps (`DateTimeIndex`) while simulation result files hold absolute seconds (`FloatIndex`). To compare the two when calibrating a model, be sure to convert your datetime based data to a `FloatIndex` using `tsd.to_float_index()`. 
To convert it back, you may use `tsd.to_datetime_index()`.
To account for different frequencies of the data, use `tsd.clean_and_space_equally()`. 
Be sure to match the frequency in your measured data with the `outputInterval` you use for your simulation.
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


