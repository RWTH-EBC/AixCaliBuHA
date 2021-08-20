---
title: 'AixCaliBuHA: Automated calibration of building and HVAC systems'
tags:
  - Python
  - Calibration
  - Building
  - Dynamic models
authors:
  - name: Fabian Wüllhorst
    orcid: TODO
    affiliation: 1
  - name: Thomas Storek
    orcid: TODO
    affiliation: 1
  - name: Philipp Mehrfeld
    orcid: TODO
    affiliation: 1
  - name: Dirk Müller
    orcid: TODO
    affiliation: 1
affiliations:
 - name: Institute for Energy Efficient Buildings and Indoor Climate, RWTH Aachen University
   index: 1
date: 30 August 2021
bibliography: paper.bib
---

# Summary

`AixCaliBuHA` enables the automated calibration of dynamic building and HVAC (heating, ventilation and air conditioning) simulation models.
Currently, we support the calibration of Modelica models through FMU and Dymola.
As the former enables a software-independent simulation, our framework is applicable to any time-variant simulation software that supports the FM standard.
The overall toolchain automated by `AixCaliBuHA` is depicted in \autoref{fig:flowshart}.
At the heart of `AixCaliBuHA` lays the definition of data types, which link the python objects to the underlying optimization problem and are used for all subsequent steps.
This definition is explained in this \autoref{sec:problem_def}.

Before executing the calibration, an automated sensitivity analysis can be performed to identify relevant parameters using the `SALib` [@Herman2017].
In the calibration itself, the optimization is solved by using already published gradient-free solvers (e.g. [@2020SciPy-NMeth; @dlib09; @pymoo]).
The whole process is visualized with optional progress plots to inform the user about convergence and design space exploration.
While the process chain can be fully automated, users can also perform semi-automatic calibration using their expert knowledge.

Furthermore, most classes created for `AixCaliBuHA` are relevant to other research topics. 
All such classes are lumped in the new repository `ebcpy`.
This repository aims to collect modules commonly used to analyze and optimize building energy systems and building indoor environments.
Lastly, the coupling between Modelica and python is standardized using the small Modelica library `MoCaTe` (Modelica Calibration Templates).
However, it's usage is optional. 

![Steps to perform in order to calibrate a model using `AixCaliBuHA`.\label{fig:flowshart}](docs/img/paper_fig_1.png){ width=60% }


# Statement of need

Simulation based analysis is key towards a sustainable building energy sector.
In order for the simulation to yield profound results, model parameters have to be calibrated with experimental data. 
As 74 % of calibrations are performed manually [@coakley2014review], there is a desperate need for an automated software tool.
We therefore developed `AixCaliBuHA` to automate the calibration process of energy-related building and HVAC system models.
As such models are inherently time dependent and commonly created using Modelica, we focus the development onto such use cases.
However, the code can also be extended to static calibration or other simulation languages.
`AixCaliBuHA` was already used in various contributions concerning calibration and digital twins. [@vering_borges; @Mehrfeld.HPC.2020; @storek_applying_2019; @ModelicaConferenceWullhorst].
We hope to extend the circle of users and developers by publishing the software.

While implementing `AixCaliBuHA`, we identified a secondary need. 
Current simulation APIs and gradient-free optimization methods lack a common interface.
Switching between different frameworks requires substantial effort and programming knowledge.
We thus created wrapper classes and extracted them into `ebcpy`.
`ebcpy` can be used to optimize dynamic simulation models and analyze time series data.
It was already used for the design optimization of heat pump systems in a recent publication [@vering_wullhorst_ecm].

# Link between optimization and class definition
\label{sec:problem_def}
Before any automated calibration of models can take place, the underlying optimization problem has to be formulated.
The goal of any calibration is to minimize the deviation between some measured data $\hat{y}(t)$ and simulated data $y(t)$ by varying the model parameters $p$:

\begin{alignat*}{2}
\label{eq:problem}
&\min_p \quad &&\sum_{i=0}^N w_i\cdot f(y_i(t), \hat{y}_i(t))\\
&s.t. &&p_\mathrm{LB} \leq p \leq p_\mathrm{UB},\\
&     &&y(t) = F(t, p, u(t)) \quad \forall t\in [t_\mathrm{start}, t_\mathrm{end}]
\end{alignat*}

In this formulation, $N$ is the number of variables to be matched by the simulation, $w_i$ is the weighing of the $i$-th target variable and $f$ is some function to evaluate the difference between $y$ and $\hat{y}$, e.g. the root mean square error (RMSE).
As constraints, the parameter have some upper (UB) and lower boundaries (LB).
Additionally, the simulated data $y(t)$ is output of the simulation model $F$ which depends on the time $t$, tuneable model parameters $p$ and time-variant input data $u(t)$. 

This mathematical formulation is transformed into python using a `CalibrationClass`. 
This class contains the goal of the calibration (mathematically speaking the objective), the parameters to tune (the optimization variables) and further information like simulation time and inputs. 
Lastly, the simulation model $F$ is included by calling one of the `SimulationAPI` childs of `ebcpy`.
The overall link is displayed in \autoref{fig:link_problem}.

![Link between the optimization problem and the `CalibrationClass` object.\label{fig:link_problem}](docs/img/paper_fig_2.png){ width=80% }

Once these classes are set up, after the execution the calibration runs fully automated.
While the automated extraction of model outputs and parameters and thus a full automation can be used, we let the degree of automation in the hands of the user.

Using `CalibrationClass` objects is the foundation of the method initially described in [@storek_applying_2019].
While this method will be submitted to a corresponding journal shortly, the focus lays in the time series analysis, clustering and classification.
The focus is not, unlike the present paper, the automation of the calibration itself.

# Acknowledgements

TBD

# References
