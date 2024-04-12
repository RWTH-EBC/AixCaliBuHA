# # Example 2-A Optimization problem definition

# Goals of this part of the examples:
# 1. Learn how to formulate your calibration problem using our data_types
# 2. Get to know `TunerParas`
# 3. Get to know `Goals`
# 4. Get to know `CalibrationClass`
# 5. Learn how to merge multiple classes
#
# Start by importing all relevant packages
from pathlib import Path
# Imports from ebcpy
from ebcpy import TimeSeriesData
# Imports from aixcalibuha
from aixcalibuha import TunerParas, Goals, \
    CalibrationClass
from aixcalibuha.data_types import merge_calibration_classes


def main(
        examples_dir,
        statistical_measure="NRMSE",
        multiple_classes=True
):
    """
    Arguments of this example:

    :param [Path, str] examples_dir:
        Path to the examples folder of AixCaliBuHA
    :param str statistical_measure:
        Measure to calculate the scalar of the objective,
        One of the supported methods in
        ebcpy.utils.statistics_analyzer.StatisticsAnalyzer
        e.g. RMSE, MAE, NRMSE
    :param bool multiple_classes:
        If False, all CalibrationClasses will have the
        same name
    """

    # ## Tuner Parameters
    # Tuner parameters are the optimization variables we will be
    # changing to match the simulated onto the measured output.
    #
    # As described in the first example (e1_A_energy_system_analysis),
    # we've changed four parameters in the model. To show the usefulness
    # of sensitivity analysis prior to calibration, we will add a fifth without
    # any influence, the heating curve declination of the heat pump.
    # To define tuner parameters, you have to specify
    # - the name of the parameter
    # - an initial guess
    # - boundaries as a (min, max) tuple.
    # Note that the initial guess is not always used by optimization routines.
    # We've chosen to make it a requirement to prevent blindly accepting
    # calibration results. If the result is very far away from your initial guess
    # and you though you understand the model, maybe the parameter is just not
    # sensitive or influenced by another parameter.
    # How to load the data is up to you. To make the structure clear,
    # we use a 3 element tuple in this example:
    data = [
        # (name, initial_value, boundaries)
        ("heatPumpSystem.declination", 2, (1, 5)),
        ("vol.V", 40, (1, 100)),
        ("heaCap.C", 50000, (1000, 100000)),
        ("rad.n", 1.24, (1, 2)),
        ("theCon.G", 250, (1, 1000))
    ]
    tuner_paras = TunerParas(
        names=[entry[0] for entry in data],
        initial_values=[entry[1] for entry in data],
        bounds=[entry[2] for entry in data]
    )
    print(tuner_paras)
    print("Names of parameters", tuner_paras.get_names())
    print("Initial values", tuner_paras.get_initial_values())
    # Scaling (will be done internally)
    print("Scaled initial values:\n", tuner_paras.scale(tuner_paras.get_initial_values()))

    # ## Goals
    # The evaluation of your goals (or mathematically speaking 'objective function')
    # depends on the difference of measured to simulated data.
    # Thus, you need to specify both measured and simulated data.
    #
    # Start by loading the measured data generated in 1_A_energy_system_analysis.py:
    data_dir = Path(examples_dir).joinpath("data")
    meas_target_data = TimeSeriesData(data_dir.joinpath("measured_target_data.hdf"), key="example")
    # Map the measured keys to the names inside your simulation
    variable_names = {
        # Name of goal: Name of measured variable, Name of simulated variable
        # Either use list
        "Electricity": ["Pel", "Pel"],
        # Or dict
        "Room temperature": {"meas": "TAir", "sim": "vol.T"}
    }
    # To match the measured data to simulated data,
    # the index has to match with the simulation output
    # Thus, convert it:
    meas_target_data.to_float_index()
    # Lastly, setup the goals object. Note that the statistical_measure
    # is parameter of the python version of this example. It's a metric to
    # compare two set's of time series data. Which one to choose is up to
    # your expert knowledge. If you have no clue, raise an issue or read
    # basic literature on calibration.
    goals = Goals(
        meas_target_data=meas_target_data,
        variable_names=variable_names,
        statistical_measure=statistical_measure,
        weightings=[0.7, 0.3]
    )
    # Let's check if our evaluation is possible by creating some
    # dummy `sim_target_data` with the same index:
    sim_target_data = TimeSeriesData({"vol.T": 293.15, "Pel": 0},
                                     index=meas_target_data.index)
    print("Goals data before setting simulation data:\n", goals.get_goals_data())
    goals.set_sim_target_data(sim_target_data)
    print("Goals data after setting simulation data:\n", goals.get_goals_data())
    print(statistical_measure, "of goals: ", goals.eval_difference())
    print("Verbose information on calculation", goals.eval_difference(verbose=True))
    # Lastly we advice to play around with the index of the sim_target_data to
    # understand the error messages of this framework a little bit better.
    # Example:
    new_index = [0.0, 600.0, 1200.0, 1800.0, 2400.0, 3000.0, 3600.0]
    sim_target_data = TimeSeriesData({"vol.T": 293.15, "Pel": 0},
                                     index=new_index)
    try:
        goals.set_sim_target_data(sim_target_data)
    except Exception as err:
        print("I knew this error was going to happen. Do you understand "
              "why this happens based on the following message?")
        print(err)
    new_index = meas_target_data.index.values.copy()
    new_index[-10] += 0.05  # Change some value
    sim_target_data = TimeSeriesData({"vol.T": 293.15, "Pel": 0},
                                     index=new_index)
    try:
        goals.set_sim_target_data(sim_target_data)
    except Exception as err:
        print("I knew this error was going to happen. Do you understand "
              "why this happens based on the following message?")
        print(err)

    # ## Calibration Classes
    # We now are going to wrap everything up into a single object called
    # `CalibrationClass`.
    # Each class has a `name`, a `start_time`, `stop_time` and
    # `goals`, `tuner_paras` (tuner parameters) and `inputs`.
    # The latter three can be set for all
    # classes if a distinction is not required.
    # ### Why do we use a `CalibrationClass`?
    # Because this class contains all information necessary
    # to perform both sensitivity analysis and calibration automatically.
    # ### Can there be multiple classes?
    # Yes! Because we expect different tuner parameters
    # to influence the outputs based on the state of the system,
    # e.g. 'On' and 'Off' more or less. To reduce the complexity of the
    # optimization problem, separating tuner parameters into time intervals
    # can be handy. For example heat losses to the ambient may be most
    # sensitive if the device is just turned off, while efficiency is more
    # sensitive during runtime.
    calibration_classes = [
        CalibrationClass(
            name="On",
            start_time=0,
            stop_time=290
        ),
        CalibrationClass(
            name="Off" if multiple_classes else "On",
            start_time=290,
            stop_time=1280
        ),
        CalibrationClass(
            name="On",
            start_time=1280,
            stop_time=1570
        ),
        CalibrationClass(
            name="Off" if multiple_classes else "On",
            start_time=1570,
            stop_time=2080
        ),
        CalibrationClass(
            name="On",
            start_time=2080,
            stop_time=2360
        )
    ]
    # Set the latter three for all classes.
    # First load the inputs of the calibration:
    meas_inputs_data = TimeSeriesData(data_dir.joinpath("measured_input_data.hdf"), key="example")
    # Rename according to simulation input:
    meas_inputs_data = meas_inputs_data.rename(columns={"TDryBulSource.y": "TDryBul"})
    for cal_class in calibration_classes:
        cal_class.goals = goals
        cal_class.tuner_paras = tuner_paras
        cal_class.inputs = meas_inputs_data
    # ## Merge multiple classes
    # If wanted, we can merge multiple classes and optimize them as one.
    # Example:
    print([c.name for c in calibration_classes])
    calibration_classes_merged = merge_calibration_classes(calibration_classes)
    print([c.name for c in calibration_classes_merged])
    # Don't worry, the relevant_time_interval object keeps track
    # of which time intervals are relevant for the objective calculation
    print("Relevant time interval for class",
          calibration_classes_merged[0].name,
          calibration_classes_merged[0].relevant_intervals)
    # Let's also create an object to later validate our calibration:
    validation_class = CalibrationClass(
        name="Validation",
        start_time=2360,
        stop_time=3600,
        goals=goals,
        tuner_paras=tuner_paras,
        inputs=meas_inputs_data
    )
    return calibration_classes, validation_class


if __name__ == '__main__':
    main(
        examples_dir=Path(__file__).parent,
        multiple_classes=True
    )
