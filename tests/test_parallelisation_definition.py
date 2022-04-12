import pathlib
import matplotlib.pyplot as plt
# Imports from ebcpy
from ebcpy import TimeSeriesData
# Imports from aixcalibuha
from aixcalibuha import TunerParas, Goals, \
    CalibrationClass
from aixcalibuha.data_types import merge_calibration_classes


def main(
        statistical_measure="NRMSE",
        multiple_classes=True
):
    # ######################### Tuner Parameter ##########################
    data = [
        # (name, initial_value, boundaries)
        # heat transfer
        ("tunerParameters.correctionShah", 1, (0.5, 1.5)),
        ("tunerParameters.correctionHaaf", 80, (60, 120)),
        # pressure drop
        ("tunerParameters.correctionPressureDropExponent", 0.5, (0.4, 1)),
        ("tunerParameters.pressureDropMax", 40, (20, 100)),
        ("tunerParameters.maximalDiffusionCorrection", 40, (2.5, 7)),
        ("tunerParameters.minimalDiffusionCorrection", 40, (1, 2.5)),
        # moist air wall cell
        #("tunerParameters.correctionFrostVelocity", 0.985, (0.8, 1.1)),
        #("tunerParameters.correctionDefrostVelocity", 50, (25, 150)),
        # frosting model
        #("tunerParameters.correctionFrostDistribution", 5, (5, 8)),
        # diffusion model
        ("tunerParameters.correctionDiffusion", 0.426, (0.2, 0.8)),
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

    # ######################### Goals ##########################
    meas_target_data = TimeSeriesData(
        r"D:\GIT\vapor-compression-models\scripts\calibration\AixCaliBuHA\frosting_calibration\data\calibration_data.mat")

    # Map the measured keys to the names inside your simulation
    variable_names = {
        # Name of goal: Name of measured variable, Name of simulated variable
        # Either use list
        "m_frost": {"meas": "frost_mass", "sim": "calBusTargetSimed.m_frost"},
        "h_out": {"meas": "h_VLE_out", "sim": "calBusTargetSimed.h_out_ref"},
        "T_Fin": {"meas": "T_Fin", "sim": "calBusTargetSimed.T_Fin"},
        "dp": {"meas": "dp", "sim": "calBusTargetSimed.dp"},
    }

    # To match the measured data to simulated data,
    # the index has to match with the simulation output
    # Thus, convert it:
    # meas_target_data.to_float_index()
    # Goals
    goals = Goals(
        meas_target_data=meas_target_data,
        variable_names=variable_names,
        statistical_measure=statistical_measure,
        weightings=[0.05, 0.88, 0.02, 0.05]
    )

    # ######################### Calibration Classes ##########################
    # We now are going to wrap everything up into a single object called
    # `CalibrationClass`.
    # Each class has a name, a start_time, stop_time and
    # goals, tuner parameters and inputs. The latter three can be set for all
    # classes if a distinction is not required.
    # Why do we use CalibrationClasses? Because we expect different parameters
    # to influence the outputs based on the state of the system, e.g. 'On' and 'Off'.

    # disregarded intervall = [100,3180,3340,3700,3800,7300,7380,7520,7620,9920,10020,10180]
    calibration_classes = [
        CalibrationClass(  # Exp1: Frosting
            name="On",
            start_time=100,
            stop_time=1000
        )
        # CalibrationClass(
        #     name="On",  # Exp1: Defrosting
        #     start_time=4850,
        #     stop_time=9460
        # ),
        # CalibrationClass(
        #     name="On",  # Exp1: Defrosting
        #     start_time=9560,
        #     stop_time=11890
        # ),
        # CalibrationClass(
        #     name="On",  # Exp1: Defrosting
        #     start_time=12100,
        #     stop_time=14000
        # )
    ]
    # Set the latter three for all classes.
    # First load the inputs of the calibration:

    #meas_inputs_data = TimeSeriesData(
    #    r"D:\Software\vapour_compression_models\scripts\calibration\AixCaliBuHA\frosting_calibration\data"
    #    r"\Frosting_calibration_data.hdf",
    #    key="DataWithDateTimeIndex")

    for cal_class in calibration_classes:
        cal_class.goals = goals
        cal_class.tuner_paras = tuner_paras
        # cal_class.inputs = meas_inputs_data

    # ######################### Merge multiple classes ##########################
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
        start_time=5,
        stop_time=14050,
        goals=goals,
        # relevant_intervals=[100, 10180],
        tuner_paras=tuner_paras,
       # inputs=meas_inputs_data
    )

    return calibration_classes, validation_class


if __name__ == '__main__':
    main(multiple_classes=True)