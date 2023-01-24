# import os
# os.environ["CRDS_PATH"] = "~/crds_cache"
# os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"

from astropy.io import fits
from PIL import Image, ImageDraw
import numpy as np
import math
import colorsys

def data_to_img_grayscale(data, shape, name):
    [_, scale] = np.percentile(data, [1, 96])
    processed_data = data / scale
    processed_data = np.clip(processed_data, 0, 1)

    output_img = Image.new("RGB", shape)

    xmin = 0.08
    xmax = 1
    X0 = 0
    X1 = 255
    b = (X1 - X0) / math.log(xmax / xmin)
    a = X0 - b * math.log(xmin)

    processed_data = np.clip(processed_data, xmin, xmax)

    for x in range(shape[0]):
        for y in range(shape[1]):
            value = processed_data[x][y]
            value = a + b * math.log(value)
            value = np.clip(value, 0, 255)
            value = math.floor(value)
            output_img.putpixel((x, y), (value, value, value))
    output_img.save(name)

hdul = fits.open("./data/europa.fits")
hdul.info()

input_data = hdul["SCI"].data
shape = tuple(input_data.shape[1 : 3])
output_data = np.ndarray(shape=shape)
deviation_data = np.ndarray(shape=shape)
error_data = np.ndarray(shape=shape, dtype=np.float32)
for x in range(shape[0]):
    for y in range(shape[1]):
        output_data[x, y] = np.mean(input_data[:, x, y])
        deviation_data[x, y] = np.std(input_data[:, x, y])        
        error_data[x, y] = np.mean(hdul["ERR"].data[:, x, y])

data_to_img_grayscale(output_data, shape, "data.png")
data_to_img_grayscale(deviation_data, shape, "std.png")
data_to_img_grayscale(error_data, shape, "error.png")