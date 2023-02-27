# # Example 3 sensitivity analysis with dymola api

# Goals of this part of the examples:
# 1. Learn how to execute a sensitivity analysis with the dymola api
#
# Import a valid analyzer, e.g. `SobolAnalyzer`
from aixcalibuha import SobolAnalyzer
from aixcalibuha.data_types import merge_calibration_classes
from ebcpy.utils.conversion import convert_tsd_to_modelica_txt


def run_sensitivity_analysis(
        examples_dir,
        aixlib_mo,
        example: str = "B",
        n_cpu: int = 1
):
    """
    Example process of a verbose sensitivity analysis with the dymola_api.

    :param [pathlib.Path, str] examples_dir:
        Path to the examples folder of AixCaliBuHA
    :param str example:
        Which example to run, "A" or "B"
    :param int n_cpu:
        Number of cores to use

    :return: A list of calibration classes
    :rtype: list
    """
    # ## Setup
    # Using a dymola api instead of the fmu api
    from examples import setup_dym_api, setup_calibration_classes
    sim_api = setup_dym_api(examples_dir=examples_dir, aixlib_mo=aixlib_mo, example=example, n_cpu=n_cpu)
    calibration_classes = setup_calibration_classes(
        examples_dir=examples_dir, example=example, multiple_classes=False
    )[0]
    merged_calibration_classes = merge_calibration_classes(calibration_classes)
    merged_calibration_classes[0].name = 'global'
    calibration_classes = setup_calibration_classes(
        examples_dir=examples_dir, example=example, multiple_classes=True
    )[0]
    merged_calibration_classes.extend(merge_calibration_classes(calibration_classes))

    if example == 'B':
        merged_calibration_classes[-1].tuner_paras = merged_calibration_classes[0].tuner_paras

    # Example of Sobol method
    # Set up Sobol analyzer
    sen_analyzer = SobolAnalyzer(
        sim_api=sim_api,
        num_samples=2,
        calc_second_order=True,
        cd=examples_dir.joinpath('testzone', f'verbose_sen_dymola_{example}'),
        save_files=True,
        load_files=False,
        savepath_sim=examples_dir.joinpath('testzone', f'verbose_sen_dymola_{example}', 'files')
    )

    # The only difference to the fmu example is the handling of inputs. There we have in the model now
    # a table for the inputs and generate an input file here. This is only necessary for example A because
    # example B has no inputs.
    # To generate the input in the correct format, use the convert_tsd_to_modelica_txt function:
    if example == "A":
        table_name = "InputTDryBul"
        file_name = r"D:\sbg-hst\Repos\AixCaliBuHA\examples\data\dymola_inputs_A.txt"
        print(file_name)
        filepath = convert_tsd_to_modelica_txt(
            tsd=merged_calibration_classes[0].inputs,
            table_name=table_name,
            save_path_file=file_name
        )
        for c in merged_calibration_classes:
            c._inputs = None
        print("Successfully created Dymola input file at", filepath)
    # run sensitivity analysis
    result, classes = sen_analyzer.run(calibration_classes=merged_calibration_classes,
                                       verbose=True,
                                       use_first_sim=True,
                                       plot_result=False,
                                       save_results=True,
                                       suffix='mat',
                                       n_cpu=1)
    print("Result of the sensitivity analysis")
    print('First and total order results of sobol method')
    print(result[0].to_string())
    print('Second order results of sobol method')
    print(result[1].to_string())

    # plotting Sensitivity results
    SobolAnalyzer.plot_single(result[0])

    # The plotting of second order results is only useful and working for more than 2 parameter.
    # So we only can take a look at them in example A
    if example == 'A':
        SobolAnalyzer.plot_second_order(result[1])
        SobolAnalyzer.plot_single_second_order(result[1], 'rad.n')


if __name__ == "__main__":
    import pathlib

    # Parameters for sen-analysis:
    EXAMPLE = "B"  # Or choose B
    N_CPU = 1

    # Sensitivity analysis:
    run_sensitivity_analysis(
        examples_dir=pathlib.Path(__file__).parent,
        aixlib_mo=r"D:\sbg-hst\Repos\AixLib\AixLib\package.mo",
        example=EXAMPLE,
        n_cpu=N_CPU
    )
