import numpy as np

def partial_dMF(x, mf_definition, partial_parameter):
    """
    Calculates the partial derivative of a membership function at a point x.

    Args:
        x (float): The input value at which the partial derivative is evaluated.
        mf_definition (tuple): Membership function definition (name, parameters).
        partial_parameter (str): Parameter for which the derivative is computed.

    Returns:
        float: Partial derivative value.

    Raises:
        ValueError: If the membership function or parameter is invalid.
    """
    mf_name = mf_definition[0]

    try:
        if mf_name == 'gaussmf':
            sigma = mf_definition[1]['sigma']
            mean = mf_definition[1]['mean']
            if partial_parameter == 'sigma':
                return (2.0 / sigma**3) * np.exp(-(((x - mean)**2) / sigma**2)) * (x - mean)**2
            elif partial_parameter == 'mean':
                return (2.0 / sigma**2) * np.exp(-(((x - mean)**2) / sigma**2)) * (x - mean)
            else:
                raise ValueError(f"Invalid parameter '{partial_parameter}' for 'gaussmf'.")

        elif mf_name == 'gbellmf':
            a = mf_definition[1]['a']
            b = mf_definition[1]['b']
            c = mf_definition[1]['c']
            if partial_parameter == 'a':
                return (2.0 * b * np.power((c - x), 2) * np.power(np.abs((c - x) / a), (2 * b - 2))) / \
                       (np.power(a, 3) * np.power((np.power(np.abs((c - x) / a), 2 * b) + 1), 2))
            elif partial_parameter == 'b':
                return -1.0 * (2 * np.power(np.abs((c - x) / a), 2 * b) * np.log(np.abs((c - x) / a))) / \
                       (np.power((np.power(np.abs((c - x) / a), 2 * b) + 1), 2))
            elif partial_parameter == 'c':
                return (2.0 * b * (c - x) * np.power(np.abs((c - x) / a), (2 * b - 2))) / \
                       (np.power(a, 2) * np.power((np.power(np.abs((c - x) / a), 2 * b) + 1), 2))
            else:
                raise ValueError(f"Invalid parameter '{partial_parameter}' for 'gbellmf'.")

        elif mf_name == 'sigmf':
            b = mf_definition[1]['b']
            c = mf_definition[1]['c']
            if partial_parameter == 'b':
                return -1.0 * (c * np.exp(c * (b + x))) / np.power((np.exp(b * c) + np.exp(c * x)), 2)
            elif partial_parameter == 'c':
                return ((x - b) * np.exp(c * (x - b))) / np.power((np.exp(c * (x - b)) + 1), 2)
            else:
                raise ValueError(f"Invalid parameter '{partial_parameter}' for 'sigmf'.")

        else:
            raise ValueError(f"Unsupported membership function '{mf_name}'.")

    except KeyError as e:
        raise ValueError(f"Missing parameter in membership function definition: {e}")
    except ZeroDivisionError as e:
        raise ValueError(f"Division by zero in membership function calculation: {e}")
