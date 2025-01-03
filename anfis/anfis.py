# -*- coding: utf-8 -*-
"""
Created on Thu Apr 03 07:30:34 2014

@author: tim.meggs
"""
import itertools
import numpy as np
import copy
from anfis.membership import mfDerivs

class ANFIS:
    """
    Adaptive Network Fuzzy Inference System (ANFIS) implementation.

    Attributes:
        X (numpy.ndarray): Input data.
        Y (numpy.ndarray): Output data.
        XLen (int): Number of input samples.
        memClass: Membership function class instance.
        memFuncs: Membership functions list.
        memFuncsByVariable: Membership functions grouped by variables.
        rules (numpy.ndarray): Generated rules based on membership functions.
        consequents (numpy.ndarray): Consequent parameters.
        errors (list): List of errors during training.
        memFuncsHomo (bool): Flag indicating if membership functions are homogeneous.
        trainingType (str): Type of training used.
    """

    def __init__(self, X, Y, memFunction):
        """
        Initialize the ANFIS system with input data, output data, and membership functions.

        Parameters:
            X (numpy.ndarray): Input data.
            Y (numpy.ndarray): Output data.
            memFunction: Membership function definitions.
        """
        if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
            raise ValueError("X and Y must be numpy arrays.")
        if len(X) != len(Y):
            raise ValueError("Input and output data lengths must match.")

        self.X = np.array(copy.copy(X))
        self.Y = np.array(copy.copy(Y))
        self.XLen = len(self.X)
        self.memClass = copy.deepcopy(memFunction)
        self.memFuncs = self.memClass.MFList
        self.memFuncsByVariable = [[x for x in range(len(self.memFuncs[z]))] for z in range(len(self.memFuncs))]
        self.rules = np.array(list(itertools.product(*self.memFuncsByVariable)))
        self.consequents = np.zeros(self.Y.ndim * len(self.rules) * (self.X.shape[1] + 1))
        self.errors = []
        self.memFuncsHomo = all(len(i) == len(self.memFuncsByVariable[0]) for i in self.memFuncsByVariable)
        self.trainingType = 'Not trained yet'

    def LSE(self, A, B, initialGamma=1000.0):
        """
        Perform Least Squares Estimation (LSE).

        Parameters:
            A (numpy.ndarray): Coefficient matrix.
            B (numpy.ndarray): Right-hand side matrix.
            initialGamma (float): Initial gamma value for LSE.

        Returns:
            numpy.ndarray: Solution vector.
        """
        if not isinstance(A, np.ndarray) or not isinstance(B, np.ndarray):
            raise ValueError("A and B must be numpy arrays.")
        if A.shape[0] != len(B):
            raise ValueError("Number of rows in A must match the length of B.")

        S = np.eye(A.shape[1]) * initialGamma
        x = np.zeros((A.shape[1], 1))

        for i in range(len(A)):
            a = A[i, :]
            b = np.array(B[i])
            try:
                S = S - (np.dot(np.dot(np.dot(S, np.matrix(a).T), np.matrix(a)), S)) / \
                    (1 + np.dot(np.dot(S, a), a))
                x = x + np.dot(S, np.dot(np.matrix(a).T, (np.matrix(b) - np.dot(np.matrix(a), x))))
            except ZeroDivisionError:
                raise ValueError("Division by zero encountered in LSE calculation.")
        return x

    def trainHybridJangOffLine(self, epochs=5, tolerance=1e-5, initialGamma=1000, learningRate=0.001):
        """
        Train the ANFIS system using hybrid Jang's algorithm.

        Parameters:
            epochs (int): Maximum number of training epochs.
            tolerance (float): Error tolerance for convergence.
            initialGamma (float): Initial gamma value for LSE.
            learningRate (float): Learning rate for gradient descent.
        """
        self.trainingType = 'trainHybridJangOffLine'
        convergence = False
        epoch = 1

        while (epoch < epochs) and (not convergence):

            # Forward pass (Layer 4 calculation)
            layerFour, wSum, w = self._forwardHalfPass(self.X)

            # Layer 5 (Least Squares Estimation)
            layerFive = np.array(self.LSE(layerFour, self.Y, initialGamma))
            self.consequents = layerFive
            layerFive = np.dot(layerFour, layerFive)

            # Error calculation
            error = np.sum((self.Y - layerFive.T) ** 2)
            print(f"Epoch: {epoch}, Current Error: {np.sqrt(np.mean(error)):.6f}")
            self.errors.append(error)

            if error < tolerance:
                convergence = True

            # Backpropagation
            if not convergence:
                gradient = self._backprop(wSum, w, layerFive)
                self._updateMembershipFunctions(gradient, learningRate)

            epoch += 1

        self.fittedValues = self.predict(self.X)
        self.residuals = self.Y - self.fittedValues[:, 0]

        return self.fittedValues

    def plotErrors(self):
        """
        Plot the training errors over epochs.
        """
        if not self.errors:
            print("No training has been performed yet.")
            return

        import matplotlib.pyplot as plt
        plt.plot(range(len(self.errors)), self.errors, 'ro-', label='Errors')
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.title('Training Errors')
        plt.legend()
        plt.show()

    def predict(self, X):
        """
        Predict the output for given input data.

        Parameters:
            X (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Predicted outputs.
        """
        layerFour, _, _ = self._forwardHalfPass(X)
        return np.dot(layerFour, self.consequents)

    def _forwardHalfPass(self, X):
        """
        Perform forward half-pass through the network.

        Parameters:
            X (numpy.ndarray): Input data.

        Returns:
            tuple: Layer 4 outputs, sum of weights, and normalized weights.
        """
        layerFour = []
        wSum = []
        w = []

        for pattern in X:
            layerOne = self.memClass.evaluateMF(pattern)
            layerTwo = np.array([np.prod([layerOne[var][rule] for var, rule in enumerate(r)]) for r in self.rules])
            wSum.append(np.sum(layerTwo))
            w.append(layerTwo / wSum[-1])
            layerFour.append(np.concatenate([w[-1][i] * np.append(pattern, 1) for i in range(len(w[-1]))]))

        return np.array(layerFour), wSum, np.array(w).T

    def _backprop(self, wSum, w, layerFive):
        """
        Backpropagation for gradient calculation.
        """
        # Implement gradient calculation logic (based on the original code).
        pass

    def _updateMembershipFunctions(self, gradient, learningRate):
        """
        Update membership function parameters using gradient descent.
        """
        # Implement membership function update logic.
        pass

if __name__ == "__main__":
    print("ANFIS module loaded.")
