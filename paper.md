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

`AixCaliBuHA` enables an automated calibration of dynamic building and HVAC (heating, ventilation and air conditioning) simulation models.
Currently, the package supports the calibration of Functional Mock-up Units (FMUs) based on the Functional Mock-up Interface (FMI) standard [@fmi_standard] as well as Modelica models through the `python_interface` of the Software Dymola [@dymola_software].
As the former enables a software-independent simulation, our framework is applicable to any time-variant simulation software that supports the FMI standard.
\autoref{fig:flowshart} illustrates the overall toolchain automated by `AixCaliBuHA`.
At the core of `AixCaliBuHA` lays the definition of data types, that link the python data types to the underlying optimization problem and are used for all subsequent steps.

Before executing the calibration, an automated sensitivity analysis can be performed to identify relevant parameters using the `SALib` [@Herman2017].
As the derivative of simulations is typically not available, the optimization behind the calibration is solved by using already published gradient-free solvers (e.g. [@2020SciPy-NMeth; @dlib09; @pymoo]).
The whole process is visualized by optional progress plots to inform the user about convergence and design space exploration.
Although the toolchain can be fully automated, users are free to perform semi-automatic calibration based on their expert knowledge.

As most of the classes originally created for `AixCaliBuHA` can be used to automate other tasks in simulation based research, 
we collect them in the separated project \href{https://github.com/RWTH-EBC/ebcpy}{ebcpy}.
`ebcpy` aims to collect aims to collect modules commonly used to analyze and optimize building energy systems and building indoor environments.
Last but not least the lightweight Modelica Library \href{https://github.com/RWTH-EBC/Modelica_Calibration_Templates}{Modelica Calibration Templates} (`MoCaTe`) provides a standardized interface for coupling of Modelica models to the calibration toolchain.
However, it's usage is optional. 

![Steps to perform in order to calibrate a model using `AixCaliBuHA`.\label{fig:flowshart}](docs/img/paper_fig_1.png){ width=60% }


# Statement of need

Simulation based analysis is a key enabler of a sustainable building energy sector.
In order for the simulation to yield profound results, model parameters have to be calibrated with experimental data. 
As $\SI{74}{\percent}$ of calibrations are performed manually [@coakley2014review], there is a desperate need for an automated software tool.
Therefore, we developed `AixCaliBuHA` to automate the calibration process of energy-related building and HVAC system models.
As such models are inherently time dependent and Modelica is quite popular in the context of building performance simulation, we focus the development onto such use cases.
However, the code can also be extended to static calibration or other simulation languages.
`AixCaliBuHA` was already used in various contributions concerning calibration and digital twins. [@vering_borges; @Mehrfeld.HPC.2020; @storek_applying_2019; @ModelicaConferenceWullhorst].
We hope to extend the circle of users and developers by making the code fully open-source.

While implementing `AixCaliBuHA`, we identified a secondary need. 
Current simulation APIs and gradient-free optimization methods lack a common interface.
Switching between different frameworks requires substantial implementation effort and programming knowledge.
Thus, we created wrapper classes and bundled them into `ebcpy`.
`ebcpy` can be used to optimize dynamic simulation models and analyze time series data.
It has been already used for the design optimization of heat pump systems in a recent publication [@vering_wullhorst_ecm].

# Link between optimization and class definition
\label{sec:problem_def}
Before any automated calibration of models can take place, the underlying optimization problem has to be formulated.
The goal of any calibration is to minimize the deviation between some measured data $y(t)$ and simulated data $\hat{y}(t)$ by varying the model parameters $p$:

\begin{alignat*}{2}
\label{eq:problem}
&\min_p \quad &&\sum_{i=0}^N w_i\cdot f(y_i(t), \hat{y}_i(t))\\
&s.t. &&p_\mathrm{LB} \leq p \leq p_\mathrm{UB},\\
&     &&\hat{y}(t) = F(t, p, u(t)) \quad \forall t\in [t_\mathrm{start}, t_\mathrm{end}]
\end{alignat*}

Inhere, $N$ is the number of variables to be matched by the simulation, $w_i$ is the weighing of the $i$-th target data and $f$ is some function to evaluate the deviation between $y$ and $\hat{y}$, e.g. the root mean square error (RMSE).
As constraints, the parameter may have some upper (UB) and lower boundaries (LB).
Additionally, the $\hat{y}(t)$ is output of the simulation model $F$ taking the time $t$, tuneable model parameters $p$ and time-variant input data $u(t)$ as an inputs. 

This mathematical formulation is transformed into python using a `CalibrationClass`. 
This class contains the goal of the calibration (mathematically speaking the objective), the parameters to tune (the optimization variables) and further information like simulation time and inputs. 
Lastly, $F$ is included by calling one child-class of the `SimulationAPI` of `ebcpy`.
\autoref{fig:link_problem} displays all mentioned links.

![Link between the optimization problem and the `CalibrationClass` object.\label{fig:link_problem}](docs/img/paper_fig_2.png){ width=80% }

Once instances of `CalibrationClass` and `SimulationAPI` are generated, the calibration can run fully automated.
However, the user can decide which steps to automate and which steps to perform manually using expert knowledge.

Even though one `CalibrationClass` is enough to run an automated calibration, the quality of calibration can be improved by assigning tuner parameters to different time intervals in the calibration. 
This idea, initially described in @storek_applying_2019, will be shortly submitted to a corresponding journal.
Using segmentation methods, input data can be automatically split into multiple `CalibrationClass` instances.
In future work we are going to link this segmentation onto multiple-class calibration. Thus, we further decrease manual user input and increase model accuracy.

# References
