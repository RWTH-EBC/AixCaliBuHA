import tests.test_parallelisation_definition
from tests.test_parallelisation_config import run_calibration
from ebcpy import DymolaAPI


def main():
    # Parameters for sen-analysis:
    sim_api = DymolaAPI(
        model_name="CalibrationFramework_OptiHorst.calibrationSimulator",
        cd=r"D:\Software\WorkingDirectory\test_parallelisation",
        packages=[
            r"C:\Program Files (x86)\TLK\Modelica\TILMedia 3.9.0\package.moe",  # TiLMedia
            r"C:\Program Files (x86)\TLK\Modelica\TIL 3.9.0\package.moe",  # Til Suite
            r"D:\GIT\vapor-compression-models\libraries\customTILComponents\package.mo",
            # vapor compression cycle lib
            r"D:\GIT\AixCaliBuHA\modelica_calibration_templates\CalibrationTemplates\package.mo",  # MoCaTe
            r"D:\GIT\vapor-compression-models\scripts\calibration\AixCaliBuHA\frosting_calibration"
            r"\calibration_interface\CalibrationFramework_OptiHorst\package.mo",  # CalibrationFramework
        ],
        show_window=True,
        n_restart=100,
        n_cpu=2,
        equidistant_output=False
    )
    sim_api.set_sim_setup({
        "stop_time": 14000,
        "output_interval": 20
    })
    calibration_classes, validation_class = tests.test_parallelisation_definition.main(
        statistical_measure="NRMSE",
        multiple_classes=True
    )
    # just testing
    # sim_api.simulate(return_option="savepath")
    # Calibration
    run_calibration(sim_api=sim_api,
                    cal_classes=calibration_classes,
                    validation_class=validation_class)


if __name__ == '__main__':
    main()