# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 15:41:58 2014

@author: tim.meggs
"""

from skfuzzy import gaussmf, gbellmf, sigmf

class MemFuncs:
    """Class for handling fuzzy membership functions."""
    
    funcDict = {
        'gaussmf': gaussmf,
        'gbellmf': gbellmf,
        'sigmf': sigmf
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
            rowInput (list): A list of input values to evaluate.

        Returns:
            list: A nested list with membership function results.
        """
        if not isinstance(rowInput, list):
            raise ValueError("rowInput must be a list of input values.")
        
        if len(rowInput) != len(self.MFList):
            raise ValueError("Number of variables does not match number of rule sets.")
        
        try:
            return [
                [
                    self.funcDict[self.MFList[i][k][0]](rowInput[i], **self.MFList[i][k][1])
                    for k in range(len(self.MFList[i]))
                ]
                for i in range(len(rowInput))
            ]
        except KeyError as e:
            raise ValueError(f"Invalid membership function name: {e}")
        except TypeError as e:
            raise ValueError(f"Parameter mismatch in membership function: {e}")
