import numpy as np
import pandas as pd


class Scaler:
    _PMIN = 0.044545454545454545
    _PMAX = 1 - _PMIN

    def __init__(self, min: float, max: float):
        self.__min = min
        self.__max = max

        self.__ymin = None
        self.__ymax = None
        self.__calculate_abs_marigins()

    def __calculate_abs_marigins(self):
        # LOCAL TESTING #
        # min = 4.5
        # max = 95.5
        #
        # pmin = 0.044545454545454545
        # pmax = 1 - pmin
        # LOCAL TESTING #

        min = self.__min
        max = self.__max

        pmin = Scaler._PMIN
        pmax = Scaler._PMAX

        A = np.array([[pmin, pmax], [pmax, pmin]])
        B = np.array([[min], [max]])
        invA = np.linalg.inv(A)
        tmp = invA @ B

        self.__ymin = tmp[1][0]
        self.__ymax = tmp[0][0]

    def get_marigins(self):
        return (self.__ymin, self.__ymax)

    def scale(self, y):
        ymin = self.__ymin
        ymax = self.__ymax

        return (y - ymin) / (ymax - ymin)


