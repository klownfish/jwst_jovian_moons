from astropy.io import fits
import numpy as np
import math
import sys
import matplotlib.pyplot as plt


hdul = fits.open(sys.argv[1])
hdul.info()

plt.ylim([0, 2000000])
if len(sys.argv) == 3:
    for i in range(hdul["SCI"].shape[1]):
        plt.plot(hdul["SCI"].data[:, i, int(sys.argv[2])])
if len(sys.argv) == 4:
    plt.plot(hdul["SCI"].data[:, int(sys.argv[2]), int(sys.argv[3])], marker="o")

plt.show()