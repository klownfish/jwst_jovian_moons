import numpy as np
from alive_progress import alive_bar
from PIL import Image, ImageDraw
from scipy import interpolate
import bottleneck as bn

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


def data_to_rgb(data, output_path):
    visible_light_start = 380
    visible_light_end = 700
    rgb_data = np.ndarray(shape = [data.shape[0], data.shape[1], 3])
    with alive_bar(data.shape[0] * data.shape[1], title="creating image") as bar:
        for x in range(rgb_data.shape[0]):
            for y in range(rgb_data.shape[1]):
                r = 0
                g = 0
                b = 0
                for i, value in enumerate(data[x, y, :]):
                    mapped_frequency = visible_light_start + (visible_light_end - visible_light_start) / data.shape[2] * i
                    (r_weight, g_weight, b_weight) = wave2rgb(mapped_frequency)
                    r += r_weight * value
                    g += g_weight * value
                    b += b_weight * value
                rgb_data[x, y, :] = (r, g, b)
                bar()

    rgb_data /= np.max(rgb_data)
    rgb_data = np.clip(rgb_data, 0, 1)
    output_img = Image.new("RGB", rgb_data.shape[:2])
    for x in range(rgb_data.shape[0]):
        for y in range(rgb_data.shape[1]):
            value = rgb_data[x, y]
            value = np.floor(value * 255)
            output_img.putpixel((x, y), tuple(value.astype(int)))
    output_img.save(output_path)

def fits_reorder_axes(data):
    output = np.swapaxes(data, 0, 2)
    return output

#dependencyless version
# def glitch_filter(data, error):
#     [_, cutoff] = np.percentile(data, [0, 90])
#     output = np.empty_like(data)
#     with alive_bar(data.shape[0] * data.shape[1], title="removing glitches") as bar:
#         for x in range(data.shape[0]):
#             for y in range(data.shape[1]):
#                 bar()
#                 output[x, y, 0] = data[x, y, 0]
#                 for z in range(1, data.shape[2]):
#                     if error[x, y, z] == 0 or abs(error[x, y, z]) > cutoff * 3:
#                         output[x, y, z] = output[x, y, z - 1]
#                     else:
#                         output[x, y, z] = data[x, y, z]
#     return output

def glitch_filter(data, error):
    [_, cutoff] = np.percentile(data, [0, 90])
    mask = (error == 0) | (error > abs(cutoff) * 3)
    nans = np.full(data.shape, np.nan)
    nans[..., 0] = data[..., 0]
    return bn.push(np.where(mask, nans, data))

def combine_pixels(data):
    output = np.ndarray(shape=(1, 1, data.shape[2]))
    for z in range(data.shape[2]):
        output[0, 0, z] = np.mean(data[:, :, z])
    return output

def single_pixel(data, x, y):
    output = np.ndarray(shape=(1,1, data.shape[2]))
    output[0, 0, :] = data[x, y, :]
    return output