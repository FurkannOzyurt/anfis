# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 15:41:58 2014

@author: tim.meggs
"""

import numpy as np
from skfuzzy import gaussmf, gbellmf, sigmf

class MemFuncs:
    """
    Class for handling fuzzy membership functions.
    """

    funcDict = {
        'gaussmf': gaussmf,
        'gbellmf': gbellmf,
        'sigmf': sigmf,
    }

    def __init__(self, MFList):
        """
        Initialize membership functions with a list of functions and their parameters.

        Args:
            MFList (list): A list containing membership function specifications.
        """
        if not isinstance(MFList, list):
            raise ValueError("MFList must be a list of membership functions and parameters.")
        self.MFList = MFList

    def evaluateMF(self, rowInput):
        """
        Evaluate the membership functions for a given input row.

        Args:
            rowInput (numpy.ndarray): A numpy array of input values.

        Returns:
            list: A nested list with membership function results.
        """
        if not isinstance(rowInput, np.ndarray):
            raise ValueError("rowInput must be a numpy array.")

        if rowInput.shape[0] != len(self.MFList):
            raise ValueError("Number of variables does not match number of rule sets.")

        return [
            [
                self.funcDict[self.MFList[i][k][0]](rowInput[i], **self.MFList[i][k][1])
                for k in range(len(self.MFList[i]))
            ]
            for i in range(rowInput.shape[0])
        ]
