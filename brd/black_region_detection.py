import matplotlib.pyplot as plt
import numpy as np

def detect(num):
    num = int(num)
    # Data for plotting
    t = np.arange(0.0, 2.0, 0.01)
    s = 1 + np.sin(num * np.pi * t)

    # Note that using plt.subplots below is equivalent to using
    # fig = plt.figure() and then ax = fig.add_subplot(111)
    fig, ax = plt.subplots()
    ax.plot(t, s)

    ax.set(xlabel='time (s)', ylabel='voltage (mV)',
           title='About as simple as it gets, folks '+ str(num))
    ax.grid()

    fig.savefig("static/brd/test.png")
    return ['black region detected']