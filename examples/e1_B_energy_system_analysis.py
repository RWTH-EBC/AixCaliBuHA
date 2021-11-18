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


def main(
        with_plot=True
):
    """
    Arguments of this example:

    :param str cd:
        Path in which to store the output.
        Default is the examples\results folder
    :param bool with_plot:
        Show the plot at the end of the script. Default is True.
    """
    example_path = pathlib.Path(__file__).parent

    # ######################### System analysis ##########################
    # Before using AixCaliBuHA, you should understand
    # what happens in your system. If you have questions regarding modeling assumptions,
    # ask e.g. the model developers.
    # Setup the FMU-API:
    from examples import setup_fmu
    fmu_api = setup_fmu(example="B")
    print("Pausing for analysis. Set the break point here if you like!")

    # ######################### Data generation ##########################
    # We want to exemplify the process of getting experimental data using
    # the model we later want to calibrate.
    # This is a good example for two reasons:
    # 1. You really know the optimal parameters
    # 2. We don't have to deal with measurement noise etc.
    # For this example, we simulate 10 s with a 10 ms sampling rate.
    # For further simulation help, check out the ebcpy examples.
    fmu_api.set_sim_setup({
        "stop_time": 10,
        "output_interval": 0.01
    })
    # Let's assume the real values to be:
    # speedRamp.duration = 0.432 and valveRamp.duration = 2.5423
    parameters = {"speedRamp.duration": 0.432, "valveRamp.duration": 2.5423}
    # Also set the outputs we may be interested in
    fmu_api.result_names = ["heatCapacitor.T", "pipe.T"]
    # Simulate
    tsd = fmu_api.simulate(parameters=parameters)

    # ######################### Data analysis ##########################
    # Now let's analyze the data we've generated.
    # Open the file first and extract variables of interest.
    # We want to match electrical power consumption (Pel) and room comfort (TAir)
    # in this example.
    # Check the frequency of the data:
    print("Simulation had index-frequency of %s with "
          "standard deviation of %s" % tsd.frequency)

    # Let's look at the data we've created:
    fig, ax = plt.subplots(1, 1, sharex=True)
    ax.plot(tsd['heatCapacitor.T'] - 273.15, label="Capacity")
    ax.plot(tsd['pipe.T'] - 273.15, label="Pipe")
    ax.set_ylabel("Temperature in °C")
    ax.set_xlabel("Time in s")
    ax.legend()
    if with_plot:
        plt.show()

    # ######################### Data saving ##########################
    # In order to use this data in the other examples, we have to save it.
    # Also, as data is typically generated using datetime stamps and different naming,
    # let's change some names and time index to ensure a realistic scenario:
    tsd = tsd.rename(columns={"pipe.T": "TPipe", "heatCapacitor.T": "TCapacity"})
    tsd.to_datetime_index()
    tsd.save(example_path.joinpath("data", "PumpAndValve.hdf"), key="examples")
    print("Saved data under", example_path.joinpath("data"))


if __name__ == '__main__':
    main()
