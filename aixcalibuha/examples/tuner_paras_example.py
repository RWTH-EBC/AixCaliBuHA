"""
Example file for the TunerParas class. The usage of the class
should be clear when looking at the examples.
If not, please raise an issue.
"""
from aixcalibuha import TunerParas


def setup_tuner_paras():
    """
    Example setup of tuner parameters.

    The parameter names are based on the model TestModel from
    the package AixCalTest. Open the model in Modelica to see other
    possible tuner parameters or have a look at the example on how
    to :meth:`find all tuner parameters<ebcpy.examples.dymola_api_example.example_dymola_api>`.

    The bounds object is optional, however highly recommend
    for calibration or optimization in general. As soon as you
    tune parameters with different units, such as Capacity and
    heat conductivity, the solver will fail to find good solutions.

    :return: Tuner parameter class
    :rtype: aixcalibuha.TunerParas
    """
    tuner_paras = TunerParas(names=["C", "m_flow_2", "heatConv_a"],
                             initial_values=[5000, 0.02, 200],
                             bounds=[(4000, 6000), (0.01, 0.1), (10, 300)])

    return tuner_paras


if __name__ == "__main__":
    TUNER_PARAS = setup_tuner_paras()
    print(TUNER_PARAS)
