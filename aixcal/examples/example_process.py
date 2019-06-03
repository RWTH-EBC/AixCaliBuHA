import os
from aixcal import data_types
from aixcal.classifier.classifier import DecisionTreeClassification
from aixcal.optimizer import calibration
from aixcal.sensanalyzer import sensitivity_analyzer
from aixcal.simulationapi import dymola_api
from aixcal.preprocessing import preprocessing
import pandas as pd
# pylint: disable-all


def input_handler_for_presentation(status=""):
    print(status)
    #while True:
    #    user_input = input("Continue? (y / n)")
    #    if user_input.lower() == "y":
    #        break
    #    else:
    #        print(globals().get(user_input, "not a valid variable"))


def example_dym_api_usage(dym_api):
    input_handler_for_presentation("Present the function get_all_tuner_parameters:")
    tuner_paras = dym_api.get_all_tuner_parameters()
    tuner_paras.show()
    print(tuner_paras)


def example_classifier_create_dtree(cd):
    # Analyze data into classes and return those classes.
    # e.g. classes = classifier.classify(MeasInputData)
    # Resulting list should look something like this:
    input_handler_for_presentation("Present creation of dtree:")
    classifier_input_data = data_types.MeasTargetData(cd + "//data//classifier_input.xlsx",
                                                      sheet_name="classifier_input")
    dtree_classifier = DecisionTreeClassification(cd,
                                                  time_series_data=classifier_input_data,
                                                  variable_list=["T_sink / K",
                                                                 "T_source / K",
                                                                 "m_flow_sink / kg/s"],
                                                  class_list="Class")
    dtree_classifier.create_decision_tree()
    dtree_classifier.validate_decision_tree()
    info_string = """
    Info for AixCalTest-Classification:
    column 1: Temperature of the source in K
    column 2: Temperature of the sink in K
    column 3: Mass-flow-rate if the sink in kg/s
    Possible classes: 
        -heat up
        -stationary
        -cool down
    """
    dtree_classifier.export_decision_tree_to_pickle(info=info_string)


def example_classifier_classify(cd, meas_target_data):
    pickle_file = cd + "//dtree_export.pickle"
    loaded_dtree, info = DecisionTreeClassification.load_decision_tree_from_pickle(pickle_file)
    print(info)
    dtree_classifier = DecisionTreeClassification(cd,
                                                  dtree=loaded_dtree)

    df = meas_target_data.get_df().copy()
    print(df.keys())
    relevant_cols_for_classifying = ["heater1.heatPorts[1].T", "heater.heatPorts[1].T",
                                     "sink_2.ports[1].m_flow"]
    cal_classes = dtree_classifier.classify(df[relevant_cols_for_classifying])
    return cal_classes


def example_pre_processing(df):
    input_handler_for_presentation("Pre-processing:")
    print(df)
    df = preprocessing.convert_index_to_datetime_index(df)
    print(df)
    df = preprocessing.clean_and_space_equally_time_series(df, desired_freq="10s")
    print(df)


def example_sensitivity_analysis(sim_api, cal_classes, stat_measure):
    input_handler_for_presentation("Perform the sensitivity analysis:")
    # Setup the class
    sen_problem = sensitivity_analyzer.SensitivityProblem("morris",
                                                          num_samples=2)

    sen_analyzer = sensitivity_analyzer.SenAnalyzer(dym_api.cd,
                                                    simulation_api=sim_api,
                                                    sensitivity_problem=sen_problem,
                                                    calibration_classes=cal_classes,
                                                    statistical_measure=stat_measure)

    # Choose initial_values and set boundaries to tuner_parameters
    # Evaluate which tuner_para has influence on what class
    sen_result = sen_analyzer.run()

    for result in sen_result:
        print(pd.DataFrame(result))
    input_handler_for_presentation("Wait and show the result")    

    # TODO-User: Select tuner-parameters based on the result for each class.
    cal_classes = sen_analyzer.automatic_select(cal_classes,
                                                sen_result,
                                                threshold=1)

    return cal_classes


def example_calibration(sim_api, cal_classes, stat_measure):
    input_handler_for_presentation("Perform the calibration:")
    continuous_cal = calibration.TimedeltaContModelicaCal(sim_api.cd,
                                                          sim_api,
                                                          stat_measure,
                                                          cal_classes,
                                                          timedelta=0,
                                                          num_function_calls=20,
                                                          show_plot=False,
                                                          save_files=False)
    continuous_cal.run(None, "dlib")


if __name__ == "__main__":
    # %% Define path in which you want ot work:
    filepath = os.path.dirname(__file__)
    cd = os.path.normpath(filepath + "//testzone_presentation")

    # Load measured data files
    measTargetData = data_types.MeasTargetData(cd + "//data//measTargetData.mat")

    # Define the name of your model and the packages needed for import
    # and setup the simulation api of choice
    model_name = "AixCalTest.TestModel"
    packages = [os.path.normpath(filepath + "//Modelica//AixCalTest//package.mo")]
    dym_api = dymola_api.DymolaAPI(cd, model_name, packages, show_window=True,
                                   get_structural_parameters=False)
    
    # Parameters for calibration and sen-analysis:
    statistical_measure = "RMSE"

    # %% Run Examples:
    example_dym_api_usage(dym_api)

    # %% Pre-processing
    example_pre_processing(measTargetData.get_df().copy())

    # %%Classifying:
    example_classifier_create_dtree(cd)
    
    val_dtree = data_types.MeasTargetData(cd + "//data//validate_dtree_simulation.mat")
    cal_classes_val = example_classifier_classify(cd, val_dtree)
    input_handler_for_presentation("Present validation with extended simulation:")
    for cal_class in cal_classes_val:
        print("{:10.10}: {:4.0f}-{:4.0f} s".format(cal_class.name, cal_class.start_time, cal_class.stop_time))
    
    calibration_classes = example_classifier_classify(cd, measTargetData)

    # TODO-User: Define goals and tuner-parameters
    tuner_paras = data_types.TunerParas(names=["C", "m_flow_2", "heatConv_a"],
                                        initial_values=[5000, 0.02, 200],
                                        bounds=[(4000, 6000), (0.01, 0.1), (10,300)])
    # TODO-Dev: This order is not-logical. Change the time to add simTargetData to a later stage.
    simTargetData = data_types.SimTargetData(cd + "//data//simTargetData.mat")
    goals = data_types.Goals(measTargetData,
                             simTargetData,
                             meas_columns=["heater.heatPorts[1].T", "heater1.heatPorts[1].T"],
                             sim_columns=["heater.heatPorts[1].T", "heater1.heatPorts[1].T"],
                             weightings=[0.7, 0.3])

    for cal_class in calibration_classes:
        cal_class.set_tuner_paras(tuner_paras)
        cal_class.set_goals(goals)

    # %% Sensitivity analysis:
    calibration_classes = example_sensitivity_analysis(dym_api, calibration_classes, statistical_measure)

    # %%Calibration:
    example_calibration(dym_api, calibration_classes, statistical_measure)

    # %%Validation:
    # TODO-Dev: Maybe provide function to split up data for validation:
    # TODO-Dev: Implement default validate-function into the Calibrator-Object
