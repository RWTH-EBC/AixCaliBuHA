from aixcal.optimizer import calibration
from aixcal.examples import data_types_example, dymola_api_example


def run_calibration(sim_api, cal_classes, stat_measure):
    continuous_cal = calibration.TimedeltaContModelicaCal(sim_api.cd,
                                                          sim_api,
                                                          stat_measure,
                                                          cal_classes,
                                                          timedelta=0,
                                                          num_function_calls=20,
                                                          show_plot=False,
                                                          save_files=False)
    continuous_cal.run("L-BFGS-B", "scipy")


if __name__ == "__main__":
    # Parameters for calibration:
    statistical_measure = "RMSE"

    dym_api = dymola_api_example.setup_dymola_api()
    cal_classes = data_types_example.setup_calibration_classes()

    # %%Calibration:
    run_calibration(dym_api,
                    cal_classes,
                    statistical_measure)
