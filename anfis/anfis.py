# -*- coding: utf-8 -*-
"""
Created on Thu Apr 03 07:30:34 2014

@author: tim.meggs
"""
import itertools
import numpy as np
import copy
from matplotlib import pyplot as plt
from membership import mfDerivs


class ANFIS:
    """
    Class to implement an Adaptive Network Fuzzy Inference System (ANFIS).
    """

    def __init__(self, X, Y, memFunction):
        """
        Initialize the ANFIS system with input data, output data, and membership functions.

        Args:
            X (numpy.ndarray): Input data of shape (n_samples, n_features).
            Y (numpy.ndarray): Output data of shape (n_samples,).
            memFunction: A class instance containing membership function definitions.
        """
        if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
            raise ValueError("X and Y must be numpy arrays.")
        if len(X) != len(Y):
            raise ValueError("Input and output data lengths must match.")

        self.X = X
        self.Y = Y
        self.XLen = len(self.X)
        self.memClass = copy.deepcopy(memFunction)
        self.memFuncs = self.memClass.MFList
        self.memFuncsByVariable = [
            [x for x in range(len(self.memFuncs[z]))] for z in range(len(self.memFuncs))
        ]
        self.rules = np.array(list(itertools.product(*self.memFuncsByVariable)))
        self.consequents = np.zeros(self.Y.ndim * len(self.rules) * (self.X.shape[1] + 1))
        self.errors = []
        self.memFuncsHomo = all(len(i) == len(self.memFuncsByVariable[0]) for i in self.memFuncsByVariable)
        self.trainingType = 'Not trained yet'

    def _forwardHalfPass(self, Xs):
        """
        Perform the forward pass for the ANFIS system.

        Args:
            Xs (numpy.ndarray): Input data.

        Returns:
            tuple: Layer four outputs, weight sums, and weights.
        """
        layerFour = []
        wSum = []

        for pattern in Xs:
            layerOne = self.memClass.evaluateMF(pattern)
            miAlloc = [
                [layerOne[x][self.rules[row][x]] for x in range(len(self.rules[0]))]
                for row in range(len(self.rules))
            ]
            layerTwo = np.array([np.product(x) for x in miAlloc])
            wSum.append(np.sum(layerTwo))

            layerThree = layerTwo / wSum[-1]
            rowHolder = np.concatenate([x * np.append(pattern, 1) for x in layerThree])
            layerFour.append(rowHolder)

        return np.array(layerFour), wSum, np.array(layerTwo)

    def _computeGradient(self, inputVarIndex, mfIndex, param):
        """
        Compute the gradient for a specific membership function parameter.

        Args:
            inputVarIndex (int): Index of the input variable.
            mfIndex (int): Index of the membership function.
            param (str): Parameter name (e.g., 'mean', 'sigma').

        Returns:
            float: Gradient value for the specified parameter.
        """
        gradient = 0.0
        for i in range(self.XLen):
            x = self.X[i, inputVarIndex]
            y_actual = self.Y[i]
            y_pred = np.dot(self._forwardHalfPass(self.X[i:i + 1])[0], self.consequents)

            partial_dmf = mfDerivs.partial_dMF(x, self.memFuncs[inputVarIndex][mfIndex], param)
            gradient += 2 * (y_actual - y_pred) * partial_dmf

        return gradient / self.XLen

    def trainHybridJangOffLine(self, epochs=5, k=0.001):
        """
        Train the ANFIS system using a hybrid learning algorithm.

        Args:
            epochs (int): Number of training epochs.
            k (float): Learning rate for backpropagation.
        """
        self.trainingType = 'trainHybridJangOffLine'
        convergence = False
        epoch = 1

        while epoch <= epochs and not convergence:
            layerFour, _, _ = self._forwardHalfPass(self.X)

            # Least Squares Estimation for consequents
            self.consequents = np.linalg.lstsq(layerFour, self.Y, rcond=None)[0]

            # Compute predictions
            predictions = np.dot(layerFour, self.consequents)

            # Compute error
            error = np.mean((self.Y - predictions) ** 2)
            self.errors.append(error)
            print(f"Epoch {epoch}: Error = {error}")

            # Backward propagation
            for inputVarIndex in range(len(self.memFuncs)):
                for mfIndex, mf in enumerate(self.memFuncs[inputVarIndex]):
                    for param in mf[1]:  # Iterate over all parameters in the membership function
                        gradient = self._computeGradient(inputVarIndex, mfIndex, param)
                        mf[1][param] -= k * gradient

            # Check convergence
            if len(self.errors) > 1 and abs(self.errors[-1] - self.errors[-2]) < 1e-5:
                convergence = True

            epoch += 1

    def plotErrors(self):
        """
        Plot the training errors over epochs.
        """
        if self.trainingType == 'Not trained yet':
            print("Model has not been trained yet.")
            return

        plt.plot(range(len(self.errors)), self.errors, 'ro-', label='Error')
        plt.xlabel("Epoch")
        plt.ylabel("Error")
        plt.title("Training Error")
        plt.legend()
        plt.show()

    def plotMF(self, inputVarIndex, x_range):
        """
        Plot the membership functions for a specific input variable.

        Args:
            inputVarIndex (int): Index of the input variable.
            x_range (numpy.ndarray): Range of x values for the plot.
        """
        if inputVarIndex >= len(self.memFuncs):
            raise ValueError(f"Input variable index {inputVarIndex} is out of range.")

        plt.figure(figsize=(8, 5))
        for mf_index, mf in enumerate(self.memFuncs[inputVarIndex]):
            mf_name = mf[0]
            mf_params = mf[1]
            if mf_name == 'gaussmf':
                y = self.memClass.funcDict['gaussmf'](x_range, **mf_params)
            elif mf_name == 'gbellmf':
                y = self.memClass.funcDict['gbellmf'](x_range, **mf_params)
            elif mf_name == 'sigmf':
                y = self.memClass.funcDict['sigmf'](x_range, **mf_params)
            else:
                raise ValueError(f"Unsupported membership function: {mf_name}")

            plt.plot(x_range, y, label=f"{mf_name} {mf_index + 1}")

        plt.title(f"Membership Functions for Input Variable {inputVarIndex + 1}")
        plt.xlabel("x")
        plt.ylabel("Membership Degree")
        plt.legend()
        plt.grid(True)
        plt.show()

    def predict(self, varsToTest):
        """
        Predict outputs for the given input data.

        Args:
            varsToTest (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Predicted output values.
        """
        layerFour, _, _ = self._forwardHalfPass(varsToTest)
        return np.dot(layerFour, self.consequents)


if __name__ == "__main__":
    print("ANFIS module loaded successfully!")
