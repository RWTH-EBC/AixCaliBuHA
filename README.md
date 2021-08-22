[![pylint](https://ebc.pages.rwth-aachen.de/EBC_all/github_ci/AixCaliBuHA/master/pylint/pylint.svg)](https://ebc.pages.rwth-aachen.de/EBC_all/github_ci/AixCaliBuHA/master/pylint/pylint.html)
[![documentation](https://ebc.pages.rwth-aachen.de/EBC_all/github_ci/AixCaliBuHA/master/docs/doc.svg)](https://ebc.pages.rwth-aachen.de/EBC_all/github_ci/AixCaliBuHA/master/docs/index.html)
[![coverage](https://ebc.pages.rwth-aachen.de/EBC_all/github_ci/AixCaliBuHA/master/coverage/badge.svg)](https://ebc.pages.rwth-aachen.de/EBC_all/github_ci/AixCaliBuHA/master/coverage)


# AixCaliBuHA

**Aix** (from French Aix-la-Chapelle) 
**Cali**bration for **Bu**ilding and **H**V**A**C Systems

This framework attempts to make the process of calibrating models used in Building
and HVAC Systems easier.

# Key features
- Performing a **Sensitivity Analysis** to discover tuner parameters for the calibration
- **Calibration** of a given model based on the tuner parameters, the calibration classes and specified goals to evaluate the objective function of the underlying optimization

# Installation
Basic knowlege about **git** and **python** are required to understand the following simple steps.  
We tested this with `cmd` on a *Windows* 10 machine.

Until this is not publicly available, you have to install it (and [ebcpy](https://github.com/RWTH-EBC/ebcpy)) via:
```
git clone --recurse-submodules https://github.com/RWTH-EBC/AixCaliBuHA
pip install -e AixCaliBuHA/ebcpy
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
We have three services in place to help you with the setup of `AixCaliBuHa`. For the basics on using this repo, we recommend the Jupyter Notebook.
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

### Examples
Clone this repo and look at the examples\README.md file.
Here you will find several examples to execute.

# Documentation
Visit hour official [Documentation](https://ebc.pages.rwth-aachen.de/EBC_all/github_ci/AixCaliBuHA/master/docs).

# Problems?
Please [raise an issue here](https://github.com/RWTH-EBC/AixCaliBuHA/issues).
