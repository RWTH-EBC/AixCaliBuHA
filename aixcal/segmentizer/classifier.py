"""Module for classification using supervised learning-techniques."""

import os
import pickle
import warnings
from aixcal import data_types
from aixcal.segmentizer import Classifier
from aixcal.preprocessing import preprocessing
from sklearn import model_selection
from sklearn.metrics import classification_report, confusion_matrix
import sklearn.tree as sktree
from sklearn import __version__ as sk_version
import pandas as pd


class DecisionTreeClassification(Classifier):
    """
    The decision tree classifier class is based on supervised learning
    methods to split time-series-data into classes.
    Either provide the data and column names to create a dtree,
    or pass a DecisionTreeClassifier-object itself to directly
    classify.
    :param cd: str, os.path.normpath
        Working Directory
    :param time_series_data: aixcal.data_types.TimeSeriesData
        Given object contains all trajectories necessary to train
        the decision tree.
    :param variable_list: list
        List containing keys of dataframe of the trajectories
    :param class_list:
        List containing keys of the dataframe relevant for classes.
    :param dtree: sklearn.tree.DecisionTreeClassifier
        Decision tree classifier that is already fitted."""

    # Dummy object for the later calculated decision tree.
    _dtree = sktree.DecisionTreeClassifier()
    _trained_successfully = False
    _class_used_for_fitting = True
    _X, _y = pd.DataFrame(), pd.DataFrame()
    _X_train, _X_test = pd.DataFrame(), pd.DataFrame()
    _y_train, _y_test = pd.DataFrame(), pd.DataFrame()
    test_size = 0.3
    # kwarg for exporting the created image to a png or not.
    save_image = False

    def __init__(self, cd, time_series_data=None, variable_list=None,
                 class_list=None, dtree=None, **kwargs):
        """Instantiate instance attributes"""
        super().__init__(cd, **kwargs)
        if dtree is None:
            self._class_used_for_fitting = True

            if not isinstance(time_series_data, data_types.TimeSeriesData):
                raise TypeError("Given time_series_data is of type {} but should"
                                "be of type TimeSeriesData".format(type(time_series_data).__name__))
            if not isinstance(variable_list, (list, str)):
                raise TypeError("Given variable_list is of type {} but should"
                                "be of type list or str".format(type(variable_list).__name__))
            if not isinstance(class_list, (list, str)):
                raise TypeError("Given class_list is of type {} but should"
                                "be of type list or str".format(type(class_list).__name__))
            self.variable_list = variable_list
            self.class_list = class_list
            self.df = time_series_data.df
            # Data frame with interesting values
            self._X = self.df[variable_list].copy()
            # Copy column with class descriptions and distribution to new Series
            self._y = self.df[class_list].copy()
            self._split_data()
        else:
            self._class_used_for_fitting = False
            self._trained_successfully = True
            if not isinstance(dtree, sktree.DecisionTreeClassifier):
                raise TypeError("Given dtree is of type {} but should"
                                "be of type DecisionTreeClassifier".format(type(dtree).__name__))
            self._dtree = dtree

        self.__dict__.update(kwargs)

    def _split_data(self):
        """Split data set randomly with test_size
        (if test_size = 0.30 --> 70 % are training data)"""
        self._X_train, self._X_test, self._y_train, self._y_test \
            = preprocessing.cross_validation(self._X, self._y, test_size=self.test_size)

    def create_decision_tree(self):
        """Creates a decision tree based on the training data
        defined in this class. If wanted, the decision tree can
        be exported as a image.
        :return dtree: sklearn.tree.DecisionTree
            May be used for storing of further processing."""
        if not self._class_used_for_fitting:
            raise AttributeError("When instantiating this class, you passed an existing"
                                 "dtree. Therefore, you can't create or validate a new one. "
                                 "Re-Instatiate the class with the necessary arguments.")

        # Create tree instance. Fit the known classes to the known training-data
        self._dtree.fit(self._X_train, self._y_train)

        # Export image
        if self.save_image:
            self.logger.export_decision_tree_image(self._dtree, self.variable_list)

        # Set info if the dtree was successfully created.
        self._trained_successfully = True
        return self._dtree

    def validate_decision_tree(self):
        """
        Validate the created decision tree based on the test-data
        defined when instantiating this class.
        :return:
        """
        if not self._class_used_for_fitting:
            raise AttributeError("When instantiating this class, you passed an existing"
                                 "dtree. Therefore, you can't create or validate a new one. "
                                 "Re-Instatiate the class with the necessary arguments.")
        # Predict classes for test data set
        predictions = self._dtree.predict(self._X_test)
        # Compare know classes with predicted classes (only possible if fitted manually)
        self.logger.log(classification_report(self._y_test, predictions) + "\n")
        self.logger.log(confusion_matrix(self._y_test, predictions))
        # Plot interesting causalities
        self.logger.plot_decision_tree(self.df, self.class_list)

    def classify(self, df, **kwargs):
        """
        Classification of given data in dataframe with
        a Decision-tree-classifier of sklearn. If no dtree
        object is given, the current dtree of this class will
        be used.
        :param df: pd.DataFrame
            Given dataframe may be extracted from the TimeSeriesData class. Should
            contain all relevant keys.
        :keyword dtree: sklearn.tree.DecisionTree
            If not provided, the current class dtree will be used.
            You can create a dtree and export it using this class's methods.
        :return: list
            List containing aixcal.data_types.CalibrationClass objects
        """
        # If no tree is provided, the class-decision tree will be used.
        if "dtree" not in kwargs:
            # If the class-decision tree is not trained yet, the training will be executed here.
            if not self._trained_successfully:
                self.create_decision_tree()
            dtree = self._dtree
        else:
            dtree = kwargs.get("dtree")
            if not isinstance(dtree, sktree.DecisionTreeClassifier):
                raise TypeError("Given dtree is of type {} but should be of type "
                                "sklearn.tree.DecisionTreeClassifier".format(type(dtree).__name__))

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Given df is of type {} but should "
                            "be pd.DataFrame".format(type(df).__name__))

        calibration_classes = []

        # Predict classes for test data set
        predictions = dtree.predict(df)

        # Convert predictions to calibration-classes.
        # TODO Convert given time-objects to total-seconds
        pred_df = pd.DataFrame({"time": df.index, "pred": predictions}).set_index("time")

        pred_df = pred_df.loc[pred_df["pred"].shift(-1) != pred_df["pred"]]
        _last_stop_time = df.index[0]
        for idx, row in pred_df.iterrows():
            stop_time = idx
            cal_class = data_types.CalibrationClass(row["pred"], _last_stop_time, stop_time)
            calibration_classes.append(cal_class)
            _last_stop_time = stop_time

        return calibration_classes

    def export_decision_tree_to_pickle(self, dtree=None, savepath=None, info=None):
        """
        Exports the given decision tree in form of a pickle and
        stores it on your machine. To avoid losses of data in future
        versions, the version number is stored alongside the dtree-object.
        :param dtree: DecisionTree, optional
            If no dtree is given, the dtree of this class will be saved.
        :param savepath: str, optional
            If not savepath is given, the pickle is stored in the
            current working directory.
        :param info: str
            Provide some info string on which columns should be passed when
            using this dtree.
        :type info: str
            """
        if dtree is None:
            _dtree = self._dtree
        else:
            _dtree = dtree
        if not isinstance(_dtree, sktree.DecisionTreeClassifier):
            raise TypeError("Given dtree is of type {} but should be"
                            "of type DecisionTreeClassifier".format(type(_dtree).__name__))

        if savepath is None:
            _savepath = os.path.join(self.cd, "dtree_export.pickle")
        else:
            _savepath = savepath

        with open(_savepath, "wb") as pickle_file:
            pickle.dump({"dtree": _dtree,
                         "version": sk_version,
                         "info": info}, pickle_file)

        return _savepath

    @staticmethod
    def load_decision_tree_from_pickle(filepath):
        """
        Loads the given pickle file and checks for the correct
        version of sklearn
        :param filepath: str, os.path.normpath
            Path the the pickle file
        :return: dtree
            Loaded Decision-Tree
        """

        with open(filepath, "rb") as pickle_file:
            dumped_dict = pickle.load(pickle_file)
        if dumped_dict["version"] != sk_version:
            warnings.warn("Saved dtree is under version {} but you are using {}. "
                          "Different behaviour of the dtree may "
                          "occur.".format(dumped_dict["version"], sk_version))
        return dumped_dict["dtree"], dumped_dict["info"]
