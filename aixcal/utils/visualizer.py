"""
Module with classes and function to help visualize
different processes inside the framework. Both plots
and print-function/log-function will be implemented here.
The Visualizer Class inherits the Logger class, as logging
will always be used as a default.
"""
import matplotlib.pyplot as plt
import os
from datetime import datetime


class Logger:
    def __init__(self, savepath):
        """Base class for showing the process of functions in
        this Framework with print-statements and saving everything
        relevant as a log-file.
        :param savepath: str, os.path.normpath
            Directory where to store the output of the Logger and possible
            child-classes. If the given directory can not be created, an error
            will be raised."""

        self.savepath = savepath
        if not os.path.isdir(self.savepath):
            os.makedirs(self.savepath)
        # Setup the logger
        date_string = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath_log = os.path.join(savepath, "CalibrationLog_%s.log" % date_string)

        self.integer_prec = 4  # Number of integer parts
        self.decimal_prec = 6
        self._counter_calibration = 0  # Number of function calls of calibration
        self._prec = self.decimal_prec
        self._width = self.integer_prec + self.decimal_prec + 1  # Calculate the actual width

    def log(self, text):
        """
        Logs the given text to the given log.
        :param text:
        :return:
        """
        print(text)
        with open(self.filepath_log, "w+") as log_file:
            log_file.write("\n"+text)

    def calibration_callback_func(self, xk):

        self._counter_calibration += 1
        info_string = self._get_tuner_para_values_as_string(xk)
        self.log(info_string)

    def save_calibration_result(self, res, model_name):
        """
        Process the result, re-run the simulation and generate
        a logFile for the minimal quality measurement
        :param res: scipy.optimize.minimize.result
            Result object of the minimization
        :param model_name: str
            Name of the model being calibrated
        """
        result_log = "Results for calibration of model: %s\n" % model_name
        result_log += "Minimal %s: %s\n" % (self.qual_meas, res.fun)
        result_log += "Final parameter values:\n"
        result_log += "%s\n" % self._get_tuner_para_names_as_string()
        result_log += "%s\n" % self._get_tuner_para_values_as_string(res.x)
        result_log += "Number of iterations: %s\n" % self._counter_calibration
        self.log(result_log)

    def _get_tuner_para_names_as_string(self):
        """
        Returns a string with the names of current tunerParameters
        :param for_log: Boolean
        If the string is created for the final log file, the best obj is not of interest
        :return: str
        The desired string
        """
        raise NotImplementedError
        # TODO How to access tuner parameters class in here?
        initial_names = list(self.tuner_para.keys())

        info_string = "{0:9s}".format("Iteration")

        for i in range(0, len(initial_names)):
            # Limit string length to a certain amount.
            # The full name has to be displayed somewhere else
            if len(initial_names[i]) > self._width:
                num_dots = len(initial_names[i]) - self._width
                if num_dots > 3:
                    num_dots = 3
                formatted_name = "."*num_dots + initial_names[i][-(self._width-num_dots):]
            else:
                formatted_name = initial_names[i]
            info_string += "   {0:{width}s}".format(formatted_name, width=self._width)
        # Add string for qualitative measurement used (e.g. NRMSE, MEA etc.)
        info_string += "   {0:{width}s}".format(self.qual_meas, width=self._width)
        return info_string

    def _get_tuner_para_values_as_string(self, xk):
        """
        Returns a string with the values of current tuner parameters
        as well as the objective value.
        :param xk: np.array
        Array with the current values of the calibration
        :return: str
        The desired string
        """
        raise NotImplementedError
        # TODO How to cleverly descale to tuner-paras bounds?

        # ini_vals = self._conv_set(xk)

        # This will limit the number of iterations to 999999999 (for correct format).
        # More iterations will most likely never be used.
        info_string = '{0:9d}'.format(self._counter_calibration)

        for i in range(0, len(ini_vals)):
            info_string += "   {0{width}.{prec}f}".format(ini_vals[i], width=self._width, prec=self._prec)
        # Add the last return value of the objective function.
        info_string += "   {0{width}.{prec}f}".format(self.obj_his[-1], width=self._width, prec=self._prec)
        return info_string


class Visualizer(Logger):

    def __init__(self, *args, **kwargs):
        """More advanced class to not only log ongoing function
        evaluations but also show the process of the functions
        by plotting interesting causalities and saving these plots."""
        super().__init__(*args, **kwargs)
        self.fig, self.ax = plt.subplots(1, 1)

    def calibration_callback_func(self, xk):
        """
        The callback function of the calibration process only get's
        current set of parameters. With this, set,
        :param xk:
        :return:
        """
        # Call the logger function to print and log
        super().calibration_callback_func(xk)

        # Plot the current objective value
        self.ax.plot(self._counter_calibration, self.obj_his[-1], "ro")
        # Create Labels and titles
        self.ax.set_xlabel("Number iterations")
        # TODO find elegant way to access the model_name and objective returns
        # self.ax.set_title(self.dymola_api.modelName)
        # self.ax.set_ylabel(self.qual_meas)

        # If the changes are small, it seems like the plot does
        # not fit the printed values. This boolean assures that no offset is used.
        self.ax.ticklabel_format(useOffset=False)
        plt.draw()
        plt.pause(1e-5)

    def save_calibration_result(self, res, model_name, file_type="svg"):
        """
        Process the result, re-run the simulation and generate
        a logFile for the minimal quality measurement
        :param res: scipy.optimize.minimize.result
            Result object of the minimization
        :param model_name: str
            Name of the model being calibrated
        :param file_type:
            svg, pdf or png
        :return:
        """
        super().save_calibration_result(res, model_name)
        plt.savefig(os.path.join(self.savepath, "calibration_callback_plot.%s" % file_type))
