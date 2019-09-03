import os
from aixcal import data_types
from aixcal.segmentizer.classifier import DecisionTreeClassification


def example_classifier_create_dtree(cd):
    """
    Analyze data into classes and return those classes.
    e.g. classes = DecicionTreeClassification.classify(MeasInputData)
    Resulting list should look something like this:

    :param str,os.path.normpath cd:
        Working Directory of example.
    :return:
    """

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


if __name__ == "__main__":
    # %% Define path in which you want ot work:
    filepath = os.path.dirname(__file__)
    cd = os.getcwd()

    # Load measured data files
    measTargetData = data_types.MeasTargetData(filepath + "//data//measTargetData.mat")

    # Parameters for calibration and sen-analysis:
    statistical_measure = "RMSE"

    # %%Classifying:
    example_classifier_create_dtree(cd)

    val_dtree = data_types.MeasTargetData(cd + "//data//validate_dtree_simulation.mat")
    cal_classes_val = example_classifier_classify(cd, val_dtree)
    for cal_class in cal_classes_val:
        print("{:10.10}: {:4.0f}-{:4.0f} s".format(cal_class.name, cal_class.start_time,
                                                   cal_class.stop_time))

    calibration_classes = example_classifier_classify(cd, measTargetData)
