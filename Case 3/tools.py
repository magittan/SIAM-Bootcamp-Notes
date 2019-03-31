import numpy as np
import matplotlib.pyplot as plt

def OLS(x,y,to_plot=False):
    """
    Possible Exception when there is only one data point present?
    """

    A = np.ones((x.shape[0], 2))
    A[:, 1] = x
    p = np.linalg.solve(np.dot(A.transpose(), A), np.dot(A.transpose(), y))

    f = lambda x: p[0] + p[1] * x
    E = np.linalg.norm(y - f(x), ord=2)

    if to_plot:
        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1)

        axes.plot(x, y, 'ko')
        axes.plot(x, f(x), 'r')
        axes.set_title("Least Squares Fit to Data")
        axes.set_xlabel("$x$")
        axes.set_ylabel("$f(x)$ and $y_i$")

        plt.show()
        print("E = %s" % E)

    return p, E
