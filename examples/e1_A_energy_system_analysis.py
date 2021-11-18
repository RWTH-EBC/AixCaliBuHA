"""
Goals of this part of the examples:
1. Learn how to analyze the model of your energy system
2. Improve your `SimulationAPI` knowledge
3. Improve your skill-set on `TimeSeriesData`
4. Generate some measured data to later use in a calibration
"""
# Start by importing all relevant packages
import pathlib
import matplotlib.pyplot as plt
# Imports from ebcpy
from ebcpy import DymolaAPI, TimeSeriesData


def main(
        aixlib_mo,
        cd=None,
        with_plot=True
):
    """
    Arguments of this example:
    :param str aixlib_mo:
        Path to the package.mo of the AixLib.
        This example was tested for AixLib version 1.0.0.
    :param str cd:
        Path in which to store the output.
        Default is the examples\results folder
    :param bool with_plot:
        Show the plot at the end of the script. Default is True.
    """

    # General settings
    if cd is None:
        cd = pathlib.Path(__file__).parent.joinpath("results")
    else:
        cd = pathlib.Path(cd)
    example_path = pathlib.Path(__file__).parent
    aixcalibuha_mo = example_path.joinpath("model", "AixCaliBuHAExamples.mo")

    # ######################### System analysis ##########################
    # The best way to analyze the model which we later want to calibrate,
    # either pause here (set a debug point) or open the models in Dymola separately.
    # Click through the system and subsystem to understand what happens in the model.
    # As you may have guessed, the analysis of an energy system can be quite complex
    # and is thus hard to automize. Before using AixCaliBuHA, you should understand
    # what happens in your system. If you have questions regarding modeling assumptions,
    # ask e.g. the model developers of the AixLib or the IBPSA.
    # %% Setup the Dymola-API:
    dym_api = DymolaAPI(
        model_name="AixCaliBuHAExamples.HeatPumpSystemCalibration",
        cd=cd,
        packages=[
            aixlib_mo,
            aixcalibuha_mo
        ],
        show_window=True,
        equidistant_output=False
    )
    print("Pausing for analysis. Set the break point here if you like!")

    # ######################### Data generation ##########################
    # We want to exemplify the process of getting experimental data using
    # the model we later want to calibrate.
    # This is a good example for two reasons:
    # 1. You really know the optimal parameters
    # 2. We don't have to deal with measurement noise etc.
    # For this example, we simulate 1 h with a 1 s sampling rate.
    # For further simulation help, check out the ebcpy examples.
    dym_api.set_sim_setup({
        "stop_time": 3600,
        "output_interval": 10
    })
    file_path = dym_api.simulate(
        return_option="savepath"
    )

    # ######################### Data analysis ##########################
    # Now let's analyze the data we've generated.
    # Open the file first and extract variables of interest.
    # We want to match electrical power consumption (Pel) and room comfort (TAir)
    # in this example.
    # As an input of the model, TDryBulSource.y
    # represents the outdoor air temperature
    tsd = TimeSeriesData(file_path)
    tsd = tsd[["Pel", "TAir", "TDryBulSource.y"]]
    # Check the frequency of the data:
    print("Simulation had index-frequency of %s with "
          "standard deviation of %s" % tsd.frequency)
    # Due to state events, our data is not equally sampled.
    # To later match the simulation data with a fixed output_interval,
    # we thus have to process the data further.
    # To do this, we have the function 'clean_and_space_equally'.
    # It only works on datetime indexes, hence we convert first:
    # Note: Real measured data would already contain DateTimeIndex anyways.
    tsd.to_datetime_index()
    # Save a copy to check if our resampling induces data loss:
    tsd_reference = tsd.copy()
    # Apply function
    tsd.clean_and_space_equally(desired_freq="10s")
    print("Simulation now has index-frequency of %s with "
          "standard deviation of %s" % tsd.frequency)
    # Let's check if the sampling changed our measured data:
    plt.plot(tsd_reference['TAir'], color="blue", label="Reference")
    plt.plot(tsd['TAir'], color="red", label="Resampled")
    plt.legend()
    if with_plot:
        plt.show()

    # ######################### Data saving ##########################
    # In order to use this data in the other examples, we have to save it.
    tsd_inputs = tsd[["TDryBulSource.y"]]
    tsd_measurements = tsd[["Pel", "TAir"]]
    tsd_inputs.save(example_path.joinpath("data", "measured_input_data.hdf"), key="example")
    tsd_measurements.save(example_path.joinpath("data", "measured_target_data.hdf"), key="example")
    print("Saved data under", example_path.joinpath("data"))


if __name__ == '__main__':
    # TODO-User: Change the AixLib path!
    main(
        aixlib_mo=r"D:\02_workshop\AixLib\AixLib\package.mo",
    )
