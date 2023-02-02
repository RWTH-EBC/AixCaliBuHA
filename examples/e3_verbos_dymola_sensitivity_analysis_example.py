# # Example 3 sensitivity analysis

# Goals of this part of the examples:
# 1. Learn how to execute a sensitivity analysis
# 2. Learn how to automatically select sensitive tuner parameters
#
# Import a valid analyzer, e.g. `SobolAnalyzer`
from aixcalibuha import SobolAnalyzer, FASTAnalyzer, MorrisAnalyzer
from aixcalibuha.data_types import merge_calibration_classes
import matplotlib.pyplot as plt
from ebcpy.utils.conversion import convert_tsd_to_modelica_txt
import matplotlib as mpl


def run_sensitivity_analysis(
        examples_dir,
        aixlib_mo,
        example: str = "B",
        n_cpu: int = 1
):
    """
    Example process of a verbose sensitivity analysis.
    First, the sensitivity problem is constructed, in this example
    the `morris` method is chosen.
    Afterwards, the sen_analyzer class is instantiated to run the
    sensitivity analysis in the next step.
    The result of this analysis is then printed to the user.
    The automatic_select function is presented as-well, using a threshold of 1
    and the default `mu_star` criterion.

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
    # Setup the class according to the documentation.
    # You just have to pass a valid simulation api and
    # some further settings for the analysis.
    # Let's thus first load the necessary simulation api:
    from examples import setup_dym_api, setup_calibration_classes
    sim_api = setup_dym_api(examples_dir=examples_dir, aixlib_mo=aixlib_mo, example=example, n_cpu=n_cpu)
    #
    # Now perform the analysis for multiple calibration classes.
    # First we creat one global calibration class of the complete time and name it 'global'
    calibration_classes = setup_calibration_classes(
        examples_dir=examples_dir, example=example, multiple_classes=False
    )[0]
    merged_calibration_classes = merge_calibration_classes(calibration_classes)
    merged_calibration_classes[0].name = 'global'
    # Now we add the calibration classes for the different states of the system
    calibration_classes = setup_calibration_classes(
        examples_dir=examples_dir, example=example, multiple_classes=True
    )[0]
    merged_calibration_classes.extend(merge_calibration_classes(calibration_classes))

    # For our verbose sensitivity analysis we want all classes to have the same tuner-parameters.
    # This comes in handy in a later part.
    # But the last class of example B has different tuner, so we reset them to the tuners of the other classes
    if example == 'B':
        merged_calibration_classes[-1].tuner_paras = merged_calibration_classes[0].tuner_paras

    # With the definition of the calibration classes and the loaded sim_api
    # we now take a look at the different options for sensitivity analysis.
    # First we perform the sobol method which is the most powerful currently supported method,
    # but is also the most computational demanding one.
    # Afterwards we will compare the results of the different methods.

    # Example of Sobol method
    # First we instantiate the Analyzer with the sim_api and the number of samples.
    # We do not specify a specific analysis variable and use the default option
    # to get the result of all possible options.
    # For this example we will use a small sample size to reduce the time needed,
    # which will lead to false results.
    # In the comparison of the different methods we will discuss the needed sample size.
    # We will also define a working directory were all results of the analysis will be stored.
    # Additionally, we can choose if the samples and corresponding simulation files will be saved.
    # These files can later be loaded and used for analysis of different calibration classes
    # without a new simulations. The simulations during the sensitivity analysis are the
    # main computational time factor.
    sen_analyzer = SobolAnalyzer(
        sim_api=sim_api,
        num_samples=2,
        calc_second_order=True,
        cd=examples_dir.joinpath('testzone', f'verbose_sen_dymola_{example}'),
        save_files=True,
        load_files=False,
        savepath_sim=examples_dir.joinpath('testzone', f'verbose_sen_dymola_{example}', 'files')
    )

    # sen_analyzer = FASTAnalyzer(
    #     sim_api=sim_api,
    #     num_samples=1024,
    #     cd=examples_dir.joinpath('testzone', f'verbose_fast_{example}_145'),
    #     save_files=True,
    #     load_files=False,
    #     savepath_sim=examples_dir.joinpath('testzone', f'verbose_fast_{example}_145', 'files')
    # )

    # Now we run the sensitivity analysis with the verbose option. With that we not only get the results for
    # combined target values, we also get the results for every target value alone.
    # Because we defined the first calibration class global and every calibration class has the same
    # tuner-parameters we can use the option use_first_sim, where only the first class will be simulated
    # and these simulations will be used for all other classes. For that the simulations are loaded for each class
    # and the statistical measure is only evaluated for the relevant time intervals of the class.
    # When we save or load the simulation files we can use with n_cpu>1 multiprocessing for loading and evaluation
    # of the simulations for each class. This option is specially useful for large models and large simulation data,
    # because the simulation data is stored in memory only one at a time for each process. This can prevent possible
    # memory errors, with only a small amount of more time needed.
    # We disable the automatic plot option here, but we save all results. Later we can use the plot function
    # of the analyzers to plot the results.
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
    result, classes = sen_analyzer.run(calibration_classes=merged_calibration_classes,
                                       verbose=True,
                                       use_first_sim=True,
                                       plot_result=False,
                                       save_results=True,
                                       suffix='mat',
                                       n_cpu=1)
    # After running the sensitivity analysis you can see that the working directory was created and the result
    # files were saved here. First the folder "files" were created as the result of save_files=True. In there the
    # simulation files are stored in an own folder which should not be tampered with. Additionally, the corresponding
    # samples are stored and an info.txt is created which saves the configuration to guarantee reproducibility.
    # The simulation are coupled with their samples.
    # As the results of analysis the statistical measure and corresponding sample are saved for each class. These
    # information can maybe be used for surrogate-based calibration, which is currently not implemented in AixCaliBuHA.
    # The main result of the sensitivity analysis are the sensitivity measures stored in "SobolAnalyzer_results.csv"
    # and "SobolAnalyzer_results_second_order.csv". These are also return from the run function as a tuple of tow
    # dataframes. This is specific to the sobol method all other methods only return one dataframe like the first of
    # the sobol method with maybe only other analysis variables.
    # Let´s take a look at these results.
    print("Result of the sensitivity analysis")
    # The first result has as columns the tuner-parameters and a multi level index with three levels.
    # The first level defines the calibration class. The second level defines Goal (target values) the index 'all'
    # is for the result of the combined target values in the goals. The last level defines the result of the
    # sensitivity measure for each class and goal. These analysis variables are specific for each method.
    # For their exact meaning I refer to the documentation of the SALib ot the literature.
    print('First and total order results of sobol method')
    print(result[0].to_string())
    # The specific second result of the sobol method is for second order sensitive measures.
    # These describe the interaction between to parameters, so this dataframe has fourth index level "Interaction".
    # In this level the tuner-parameters are listed again.
    print('Second order results of sobol method')
    print(result[1].to_string())

    # plotting Sensitivity results
    SobolAnalyzer.plot_single(result[0])

    # The plotting of second order results is only useful and working for more than 2 parameter.
    # So we only can take a look at them in example A
    if example == 'A':
        SobolAnalyzer.plot_second_order(result[1])
        SobolAnalyzer.plot_single_second_order(result[1], 'rad.n')

        result_sobol = SobolAnalyzer.load_from_csv(
            examples_dir.joinpath('data', f'SobolAnalyzer_results_A.csv')
        )
        result_sobol_2 = SobolAnalyzer.load_second_order_from_csv(
            examples_dir.joinpath('data', f'SobolAnalyzer_results_second_order_A.csv')
        )

        fig = plt.figure(figsize=plt.figaspect(1. / 4.))
        subfigs = fig.subfigures(1, 3, wspace=0)
        ax0 = subfigs[0].subplots(3, 1, sharex=True)
        SobolAnalyzer.plot_single(
            result=result_sobol,
            cal_classes=['global'],
            show_plot=False,
            figs_axes=([subfigs[0]], [ax0])
        )
        subfigs[0].suptitle('class: global')
        SobolAnalyzer.plot_second_order(
            result=result_sobol_2,
            cal_classes=['global'],
            goals=['all'],
            show_plot=False,
            figs=[[subfigs[1]]]
        )
        ax2 = subfigs[2].subplots(3, 1, sharex=True)
        SobolAnalyzer.plot_single_second_order(
            result=result_sobol_2,
            para_name='rad.n',
            show_plot=False,
            cal_classes=['global'],
            figs_axes=([subfigs[2]], [ax2])
        )
        plt.show()

    # Let´s compare some possible sensitivity methods.
    result_sobol = SobolAnalyzer.load_from_csv(
        examples_dir.joinpath('data', f'SobolAnalyzer_results_{example}.csv')
    )
    result_fast = FASTAnalyzer.load_from_csv(
        examples_dir.joinpath('data', f'FASTAnalyzer_results_{example}.csv')
    )
    result_morris = MorrisAnalyzer.load_from_csv(
        examples_dir.joinpath('data', f'MorrisAnalyzer_results_{example}.csv')
    )
    for c in merged_calibration_classes:
        fig_comp = plt.figure(figsize=plt.figaspect(1. / 4.))
        subfigs_comp = fig_comp.subfigures(1, 3, wspace=0)
        ax0_comp = subfigs_comp[0].subplots(3, 1, sharex=True)
        SobolAnalyzer.plot_single(
            result=result_sobol,
            cal_classes=[c.name],
            show_plot=False,
            figs_axes=([subfigs_comp[0]], [ax0_comp])
        )
        ax1_comp = subfigs_comp[1].subplots(3, 1, sharex=True)
        FASTAnalyzer.plot_single(
            result=result_fast,
            cal_classes=[c.name],
            show_plot=False,
            figs_axes=([subfigs_comp[1]], [ax1_comp])
        )
        ax2_comp = subfigs_comp[2].subplots(3, 1, sharex=True)
        SobolAnalyzer.plot_single(
            result=result_morris,
            show_plot=False,
            cal_classes=[c.name],
            figs_axes=([subfigs_comp[2]], [ax2_comp])
        )
    plt.show()

    # For each given class, you should see the given tuner parameters
    # and the sensitivity according to the selected method from the SALib.
    # Let's remove some less sensitive parameters based on some threshold
    # to remove complexity from our calibration problem:
    # print("Selecting relevant tuner-parameters using a fixed threshold:")
    # sen_analyzer.select_by_threshold(calibration_classes=classes,
    #                                  result=result[0],
    #                                  threshold=0.01,
    #                                  analysis_variable='S1')
    # for cal_class in classes:
    #     print(f"Class '{cal_class.name}' with parameters:\n{cal_class.tuner_paras}")
    # # Return the classes and the sim_api to later perform an automated process in example 5
    # return classes, sim_api


if __name__ == "__main__":
    import pathlib
    from examples import setup_fmu, setup_calibration_classes

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
