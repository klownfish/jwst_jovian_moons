# import os
# os.environ["CRDS_PATH"] = "~/crds_cache"
# os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"

from astropy.io import fits
from PIL import Image, ImageDraw
import numpy as np
import math
import sys

def wave2rgb(wave):
    # This is a port of javascript code from  http://stackoverflow.com/a/14917481
    gamma = 0.8
    intensity_max = 1
 
    if wave < 380:
        red, green, blue = 0, 0, 0
    elif wave < 440:
        red = -(wave - 440) / (440 - 380)
        green, blue = 0, 1
    elif wave < 490:
        red = 0
        green = (wave - 440) / (490 - 440)
        blue = 1
    elif wave < 510:
        red, green = 0, 1
        blue = -(wave - 510) / (510 - 490)
    elif wave < 580:
        red = (wave - 510) / (580 - 510)
        green, blue = 1, 0
    elif wave < 645:
        red = 1
        green = -(wave - 645) / (645 - 580)
        blue = 0
    elif wave <= 780:
        red, green, blue = 1, 0, 0
    else:
        red, green, blue = 0, 0, 0
 
    # let the intensity fall of near the vision limits
    if wave < 380:
        factor = 0
    elif wave < 420:
        factor = 0.3 + 0.7 * (wave - 380) / (420 - 380)
    elif wave < 700:
        factor = 1
    elif wave <= 780:
        factor = 0.3 + 0.7 * (780 - wave) / (780 - 700)
    else:
        factor = 0
 
    def f(c):
        if c == 0:
            return 0
        else:
            return intensity_max * pow (c * factor, gamma)
 
    return (f(red), f(green), f(blue))


hdul = fits.open(sys.argv[1])
hdul.info()

input_data = hdul["SCI"].data
shape = tuple(input_data.shape[1 : 3] + (3,))
output_data = np.ndarray(shape=shape)

visible_light_start = 380
visible_light_end = 700
frequencies = input_data.shape[0]

for x in range(shape[0]):
    print(x / shape[0] * 100, "%", sep="")
    for y in range(shape[1]):
        r = 0
        g = 0
        b = 0
        for i, value in enumerate(input_data[:, x, y]):
            if abs(value) > 3000000:
                continue
            mapped_frequency = visible_light_start + (visible_light_end - visible_light_start) / frequencies * i
            (r_weight, g_weight, b_weight) = wave2rgb(mapped_frequency)
            r += r_weight * value
            g += g_weight * value
            b += b_weight * value
        output_data[x, y] = (r, g, b)

[_, scale] = np.percentile(output_data, [1, 96])
processed_data = output_data / scale
processed_data = np.clip(processed_data, 0, 1)

# processed_data = np.ndarray(shape=shape)
# for i in range(3):
#     [_, scale] = np.percentile(output_data[:, :, i], [1, 96])
#     processed_data[:, :, i] = output_data[:, :, i] / scale
#     processed_data[:, :, i] = np.clip(processed_data[:, :, i], 0, 1)

output_img = Image.new("RGB", shape[:2])
for x in range(shape[0]):
    for y in range(shape[1]):
        value = processed_data[x, y]
        value = np.floor(value * 255)
        output_img.putpixel((x, y), tuple(value.astype(int)))
output_img.save(sys.argv[2])