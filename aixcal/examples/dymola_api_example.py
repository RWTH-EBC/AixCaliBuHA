import os
from aixcal.simulationapi import dymola_api


def example_dymola_api(dym_api):
    """
    Function to show the usage of the function
    get_all_tuner_parameters() of the DymolAPI. The user
    can alter the values of the parameters if needed.

    :param dymola_api.DymolaAPI dym_api:
        DymolaAPI that can be generated using :meth:` this function<setup_dymola_api>`
    :return: tuner parameters which can be used for other examples
    :rtype: aixcal.data_types.TunerParas
    """
    tuner_paras = dym_api.get_all_tuner_parameters()
    tuner_paras.show()
    return tuner_paras


def setup_dymola_api(show_window=True):
    """
    Function to show how to setup the DymolaAPI.
    As stated in the DymolaAPI-documentation, you need to
    pass a current working directory, the name of the model
    you want to work with and the necessary packages.

    :param bool show_window:
        True if you want to see the Dymola instance on
        your machine. You can see what commands in the
        interface.
    :return: The DymolaAPI created in the function
    :rtype: dymola_api.DymolaAPI

    Example:
    --------
    >>> DYM_API = setup_dymola_api(show_window=True)
    >>> DYM_API.set_sim_setup({"startTime": 100,
    >>>                        "stopTime": 200})
    >>> DYM_API.simulate()
    >>> DYM_API.close()
    """
    # Define path in which you want ot work:
    cd = os.getcwd()

    # Define the name of your model and the packages needed for import
    # and setup the simulation api of choice
    model_name = "AixCalTest.TestModel"
    packages = [os.path.normpath(os.path.dirname(__file__) + "//Modelica//AixCalTest//package.mo")]
    # Setup the dymola api
    dym_api = dymola_api.DymolaAPI(cd,
                                   model_name,
                                   packages,
                                   show_window=show_window)
    return dym_api


if __name__ == "__main__":
    # Setup the dymola-api:
    DYM_API = setup_dymola_api()
    # Run example:
    example_dymola_api(DYM_API)
