import numpy as np

def partial_dMF(x, mf_definition, partial_parameter):
    """
    Calculates the partial derivative of a membership function at a point x.

    Parameters
    ----------
    x : float
        The input value at which the partial derivative is evaluated.
    mf_definition : tuple
        A tuple containing the membership function name (str) and parameters (dict).
        Example: ('gaussmf', {'sigma': 1.0, 'mean': 0.0})
    partial_parameter : str
        The parameter with respect to which the partial derivative is computed.

    Returns
    -------
    float
        The value of the partial derivative at x.

    Raises
    ------
    ValueError
        If the membership function name or parameter is invalid.
    """
    # Extract membership function name
    mf_name = mf_definition[0]

    try:
        # Gaussian membership function
        if mf_name == 'gaussmf':
            sigma = mf_definition[1]['sigma']
            mean = mf_definition[1]['mean']
            if partial_parameter == 'sigma':
                result = (2.0 / sigma**3) * np.exp(-(((x - mean)**2) / sigma**2)) * (x - mean)**2
            elif partial_parameter == 'mean':
                result = (2.0 / sigma**2) * np.exp(-(((x - mean)**2) / sigma**2)) * (x - mean)
            else:
                raise ValueError(f"Invalid partial parameter '{partial_parameter}' for 'gaussmf'.")

        # Generalized bell membership function
        elif mf_name == 'gbellmf':
            a = mf_definition[1]['a']
            b = mf_definition[1]['b']
            c = mf_definition[1]['c']
            if partial_parameter == 'a':
                result = (2.0 * b * np.power((c - x), 2) * np.power(np.abs((c - x) / a), (2 * b - 2))) / \
                         (np.power(a, 3) * np.power((np.power(np.abs((c - x) / a), 2 * b) + 1), 2))
            elif partial_parameter == 'b':
                result = -1.0 * (2 * np.power(np.abs((c - x) / a), 2 * b) * np.log(np.abs((c - x) / a))) / \
                         (np.power((np.power(np.abs((c - x) / a), 2 * b) + 1), 2))
            elif partial_parameter == 'c':
                result = (2.0 * b * (c - x) * np.power(np.abs((c - x) / a), (2 * b - 2))) / \
                         (np.power(a, 2) * np.power((np.power(np.abs((c - x) / a), 2 * b) + 1), 2))
            else:
                raise ValueError(f"Invalid partial parameter '{partial_parameter}' for 'gbellmf'.")

        # Sigmoid membership function
        elif mf_name == 'sigmf':
            b = mf_definition[1]['b']
            c = mf_definition[1]['c']
            if partial_parameter == 'b':
                result = -1.0 * (c * np.exp(c * (b + x))) / \
                         np.power((np.exp(b * c) + np.exp(c * x)), 2)
            elif partial_parameter == 'c':
                result = ((x - b) * np.exp(c * (x - b))) / \
                         np.power((np.exp(c * (x - b)) + 1), 2)
            else:
                raise ValueError(f"Invalid partial parameter '{partial_parameter}' for 'sigmf'.")

        else:
            raise ValueError(f"Invalid membership function name '{mf_name}'.")

        return result

    except KeyError as e:
        raise ValueError(f"Missing parameter in membership function definition: {e}")
    except ZeroDivisionError as e:
        raise ValueError(f"Division by zero encountered in membership function calculation: {e}")
