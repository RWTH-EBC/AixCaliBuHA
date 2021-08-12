How to use AixCaliBuHa
=======================

While we aim at automating most parts of a calibration process, you still have to specify the inputs and the methods you want to use.
We therefore recommend to:

1. Analyze the physical system and theoretical model you want to calibrate
2. Identify inputs and outputs of the system and model
3. Get to know your tuner parameters and how they affect your model
4. Plan your experiments and perform them
5. Learn about the methods provided for calibration (statistical measures (RMSE, etc.), optimization, ...)
6. **Always be critical** about the results of the process. If the model approach or the experiment is faulty, the calibration will perform accordingly.

How to start with AixCaliBuHa?
-------------------------------
We have two services in place to help you with the setup of ``AixCaliBuHa``. For the basics on using this repo, we recommend the Jupyter Notebook.
If you want to setup your calibration models (in Modelica) and quickly start your first calibration, we provide a guided setup.


Jupyter Notebook
----------------

We recommend running our jupyter-notebook to be guided through a **helpful tutorial**.
For this, run the following code:

.. code-block::

    # If jupyter is not already installed:
    pip install jupyter
    # Go into your ebcpy-folder (cd \path_to_\AixCaliBuHA) or change the to the absolute path of the tutorial.ipynb and run:
    jupyter notebook AixCaliBuHA\examples\tutorial.ipynb


Guided Setup
--------------

After installation, you can run (in your anaconda prompt or shell with the python path set):

.. code-block::

    guided_setup

This will trigger a script to help you through the process. Be aware that this was tested for Windows where Dymola is installed in the default location.
After completing the steps, a configuration file is generated with most information already present. You have to specify some additional data in the generated config.
Afterwards, you can run

.. code-block::

    run_modelica_calibration --config=my_generated_config.toml

TimeSeriesData
---------------
Note that we use our own `TimeSeriesData` object which inherits from `pd.DataFrame`. The aim is to make tasks like loading different filetypes or applying multiple tags to one variable more convenient, while conserving the powerful tools of the DataFrame.
The class is defined in `ebcpy`, and you can also check the documentation over there. Just a quick intro here:

**Variables and tags**

.. code-block:: python

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


As you can see, our first column level is always a variable, and the second one a tag.
This is especially handy when dealing with calibration or processing tasks, where you will have multiple
versions (tags) for one variable. The default tag is `raw` to indicate the unmodified data.

**FloatIndex and DateTimeIndex**

Measured data typically holds a datetime stamps (``DateTimeIndex``) while simulation result files hold absolute seconds (``FloatIndex``). To compare the two when calibrating a model, be sure to convert your datetime based data to a ``FloatIndex`` using ``tsd.to_float_index()``.
To convert it back, you may use ``tsd.to_datetime_index()``.
To account for different frequencies of the data, use ``tsd.clean_and_space_equally()``.
Be sure to match the frequency in your measured data with the ``outputInterval`` you use for your simulation.


Goals
-----

.. autoclass:: aixcalibuha.Goals
   :members:


Calibration Classes
---------------------

.. autoclass:: aixcalibuha.CalibrationClass
   :members:
