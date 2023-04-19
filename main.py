from astropy.io import fits
import numpy as np
import math
import sys
import matplotlib.pyplot as plt
import itertools
import argparse
import utils.utils as utils
import pathlib
import scipy.ndimage
from matplotlib.patches import Circle

RADIUS = 6.5
NORTH_ANGLE = 25.4 * 3.14 / 180

parser = argparse.ArgumentParser(description="Tools to process JWST data related to the jovian moons")
parser.add_argument("input", type=str)
parser.add_argument("--glitch", action="store_true", help="removes spikes in the data")
parser.add_argument("--combined", action="store_true", help="combines all pixels into one average")
parser.add_argument("--max", action="store_true", help="singles out the max pixel")
parser.add_argument("--single", type=int, nargs=2, help="singles out the pixel at coordinates [ARG1 ARG2]")
parser.add_argument("--plot", action="store_true", help="plots one pixel")
parser.add_argument("--rgb", type=str, help="outputs a visual light image")
parser.add_argument("--info", action="store_true", help="prints some random information about the input file")
parser.add_argument("--test", action="store_true", help="only for debugging")
parser.add_argument("--lowpass", action="store_true", help="applies a lowpass filter on the input")
parser.add_argument("--normalize", action="store_true", help="rescales the input file to 1")
parser.add_argument("--cut", action="store_true", help="removes the gap left by nrs1 and nrs2")
parser.add_argument("--title", type=str, help="sets the title for the plot")
parser.add_argument("--albedo", type=str, help="divides the input with the spectrum from ARG1")
parser.add_argument("--solar", action="store_true", help="replaces the input with the solar data (for debugging)")
parser.add_argument("--snr", type=str, help="calculates and stores the CO2 dips in an image")
parser.add_argument("--focus", action="store_true", help="removes all pixels outside of ganymede")

args = parser.parse_args()

def read_hdul(hdul):
    try:
        sci_data = np.abs(utils.fits_reorder_axes(hdul["SCI"].data)) * hdul["SCI"].header["PIXAR_SR"]
        error_data = np.abs(utils.fits_reorder_axes(hdul["ERR"].data))
        wavelength_edges = (hdul["SCI"].header["CRVAL3"], hdul["SCI"].header["CRVAL3"] + hdul["SCI"].header["CDELT3"] * hdul["SCI"].data.shape[0])
        wavelengths = np.linspace(wavelength_edges[0], wavelength_edges[1], num=sci_data.shape[2])

    except Exception as e:
        print(e)
        sci_data = np.ndarray(shape=[1, 1, len(hdul["EXTRACT1D"].data)])
        error_data = np.ndarray(shape=[1, 1, len(hdul["EXTRACT1D"].data)])
        wavelengths = np.ndarray(shape=[len(hdul["EXTRACT1D"].data)])
        for i in range(len(hdul["EXTRACT1D"].data)):
            wavelengths[i] = hdul["EXTRACT1D"].data[i][0]
            sci_data[0, 0, i] = hdul["EXTRACT1D"].data[i][1]
            error_data[0, 0, i] = hdul["EXTRACT1D"].data[i][2]

    return (sci_data, error_data, wavelengths)
hdul = fits.open(args.input, memmap=False)
(sci_data, error_data, wavelengths) = read_hdul(hdul)
com = scipy.ndimage.center_of_mass(np.median(sci_data, axis=-1))


if args.info:
    hdul.info()
    for entry in itertools.zip_longest(hdul[0].header.keys(), hdul[0].header.values()):
        print(entry[0], entry[1])
    for entry in itertools.zip_longest(hdul[1].header.keys(), hdul[1].header.values()):
        print(entry[0], entry[1])

    try:
        for entry in itertools.zip_longest(hdul["EXTRACT1D"].header.keys(), hdul["EXTRACT1D"].header.values()):
            print(entry[0], entry[1])
    except:
        pass

radius_to_ganymede = 649.884825 * 10**6


if args.single:
    sci_data = utils.single_pixel(sci_data, args.single[0], args.single[1])
    error_data = utils.single_pixel(error_data, args.single[0], args.single[1])

if args.focus:
    for x in range(sci_data.shape[0]):
        for y in range(sci_data.shape[1]):
            dx = x - com[0]
            dy = y - com[1]
            delta = math.floor(math.sqrt(dx * dx + dy * dy))
            if delta > RADIUS:
                sci_data[x, y, :] = np.nan

if args.glitch:
    sci_data = utils.glitch_filter(sci_data, error_data)
    error_data = utils.glitch_filter(error_data, error_data)
    # sci_data = utils.advanced_glitch_filter(sci_data)

if args.max:
    (x, y) = utils.get_max(sci_data)
    print("max pixel", x, y)
    sci_data = utils.single_pixel(sci_data, x, y)
    error_data = utils.single_pixel(error_data, x, y)

if args.combined:
    sci_data = utils.combine_pixels(sci_data)
    error_data = utils.combine_errors(error_data)

if args.lowpass:
    # sci_data[0, 0, :] = np.sin(wavelengths * 100)
    sci_data[0, 0, :] = utils.decent_lowpass(sci_data[0, 0, :], 0.9)

if args.albedo:
    solar_hdul = fits.open(args.albedo, memmap=False)
    (solar_sci_data, solar_error_data, solar_wavelengths) = read_hdul(solar_hdul)
    solar_sci_data = utils.glitch_filter(solar_sci_data, solar_error_data)
    solar_sci_data = utils.advanced_glitch_filter(solar_sci_data)
    solar_sci_data[0, 0, :] = utils.decent_lowpass(solar_sci_data[0, 0, :], 0.995)
    solar_wave_offset = 0
    for i in range(len(solar_wavelengths)):
        solar_wave_offset = i
        if solar_wavelengths[i] > wavelengths[0]:
            break
    minimum_length = min(len(solar_wavelengths) - solar_wave_offset, len(wavelengths))
    sci_data[..., 0:minimum_length] /= solar_sci_data[..., solar_wave_offset:solar_wave_offset + minimum_length]
    if args.solar:
        sci_data = solar_sci_data
        wavelengths = solar_wavelengths
        error_data = solar_error_data

if args.normalize:
    (_, factor) = [0, np.nanpercentile(sci_data, [0, 99])[1]]
    sci_data /= factor
    error_data /= factor

if args.cut:
    begin = 3.9
    end = 4.23
    for x in range(sci_data.shape[0]):
        for y in range(sci_data.shape[1]):
            for i in range(len(wavelengths)):
                if begin < wavelengths[i] < end:
                    sci_data[x, y, i] = np.nan

if args.title:
    plt.title(args.title)

if args.plot:
    plt.ylim([0, np.nanpercentile(sci_data, [0, 99])[1]])
    plt.plot(wavelengths, sci_data[0, 0, :], marker = "o")
    plt.plot(wavelengths, error_data[0, 0, :], marker = "o")
    plt.legend(["science", "error"])
    plt.xlabel("Wavelength (μm)")
    if args.normalize:
        plt.ylabel("Albedo")
    else:
        plt.ylabel("Amplitude (MJy)")
    plt.show()


if args.rgb:
    utils.data_to_rgb(sci_data, args.rgb, com, RADIUS, NORTH_ANGLE)

if args.snr:
    cot = np.ndarray(shape = sci_data.shape[:2])
    signal = utils.find_freq(wavelengths, 4.2650)
    noise = utils.find_freq(wavelengths, 4.29)
    for x in range(sci_data.shape[0]):
        for y in range(sci_data.shape[1]):
            cot[x, y] = (sci_data[x, y, noise] - sci_data[x, y, signal]) / sci_data[x, y, noise]
    cot = 10 * np.log10(cot)
    # utils.data_to_grayscale(cot, args.snr, com)
    # cot = np.log(cot)
    fig, ax = plt.subplots()
    plt.imshow(cot.transpose(), origin='lower', cmap="gray", vmin=np.nanpercentile(cot, 2), vmax=np.nanpercentile(cot, 99))
    plt.title("CO₂ Band Depth")
    cbar = plt.colorbar()
    cbar.set_label("SNR (dB)")
    plt.arrow(com[0], com[1], math.sin(NORTH_ANGLE) * (RADIUS - 1), math.cos(NORTH_ANGLE) * (RADIUS - 1), color="black", width=0.1)
    circle = Circle(com, RADIUS, fill=False, edgecolor='black', linewidth=5)
    if not args.focus:
        ax.add_patch(circle)
    plt.axis("off")
    plt.show()