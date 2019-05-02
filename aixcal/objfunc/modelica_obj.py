import os


def obj_calibrate_dymola_model(set):
    """
    Default objective function for calibration of a modelica
    using the dymola interface.
    The usual function will be implemented here:
    1. Convert the set to modelica-units to get InitialValues
    2. Simulate with the initialValues
    3. Get simulation data as a dataFrame
    4. Calculate the objective based on statistical values
    :param set: np.array
    Array with normalized values for the minimizer
    :return:
    Objective value based on the used quality measurement
    """
    raise NotImplementedError
