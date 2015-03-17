#!/usr/bin/python
# -*- encoding: UTF-8 -*-
'''Script or Module Title
    
    This section should be a summary of important information to help the editor
    understand the purpose and/or operation of the included code.
    
    List of classes: -none-
    List of functions:
        main
'''

# third-party modules
import numpy as np

#===============================================================================
def SG_smooth(y, window_size, order, deriv=0):
    '''Savitzky-Golay Smoothing & Differentiating Function
    
    Args:
        Y (list): Objective data array.
        window_size (int): Number of points to use in the local regressions.
                        Should be an odd integer.
        order (int): Order of the polynomial used in the local regressions.
                    Must be less than window_size - 1.
        deriv = 0 (int): The order of the derivative to take.
    Returns:
        (ndarray)  The resulting smoothed curve data. (or it's n-th derivative)
    Test:
        t = np.linnp.ce(-4, 4, 500)
        y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
        ysg = sg_smooth(y, window_size=31, order=4)
        import matplotlib.pyplot as plt
        plt.plot(t, y, label='Noisy signal')
        plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
        plt.plot(t, ysg, 'r', label='Filtered signal')
        plt.legend()
        plt.show()
    '''
    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    # END try
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    
    order_range = range(order+1)
    half_window = (window_size -1) / 2
    
    # precompute coefficients
    b = np.mat(
        [
            [k**i for i in order_range] for k in range(
                -half_window, half_window+1
            )
        ]
    )
    m = np.linalg.pinv(b).A[deriv]
    
    # pad the function at the ends with EVEN reflections
    # *** Even reflection is chosen ONLY because it works well with the
    #     expected types of data in counit
    y = np.concatenate(
        (y[1:half_window+1][::-1], y, y[-half_window-1:-1][::-1])
    )
    
    return ((-1)**deriv) * np.convolve( m, y, mode='valid')
# END sg_smooth