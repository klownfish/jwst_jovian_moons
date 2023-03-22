from astropy.io import fits
import numpy as np
import math
import sys
import matplotlib.pyplot as plt
import itertools
import argparse
import utils.utils as utils
import pathlib

parser = argparse.ArgumentParser(description="Tools to process JWST data related to the jovian moons")
parser.add_argument("input", type=str)
parser.add_argument("--glitch", action="store_true")
parser.add_argument("--combined", action="store_true")
parser.add_argument("--max", action="store_true")
parser.add_argument("--single", type=int, nargs=2)
parser.add_argument("--plot", action="store_true")
parser.add_argument("--rgb", type=str)
parser.add_argument("--info", action="store_true")
parser.add_argument("--test", action="store_true")
parser.add_argument("--lowpass", action="store_true")
parser.add_argument("--normalize", action="store_true")
parser.add_argument("--cut", action="store_true")
parser.add_argument("--albedo", type=str)

args = parser.parse_args()

def read_hdul(hdul):
    try:
        sci_data = np.abs(utils.fits_reorder_axes(hdul["SCI"].data))
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




if args.info:
    hdul.info()
    for entry in itertools.zip_longest(hdul[0].header.keys(), hdul[0].header.values()):
        print(entry[0], entry[1])

radius_to_ganymede = 649.884825 * 10**6


if args.single:
    sci_data = utils.single_pixel(sci_data, args.single[0], args.single[1])
    error_data = utils.single_pixel(error_data, args.single[0], args.single[1])

if args.glitch:
    sci_data = utils.glitch_filter(sci_data, error_data)
    error_data = utils.glitch_filter(error_data, error_data)
    sci_data = utils.advanced_glitch_filter(sci_data)

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
    sci_data[0, 0, :] = utils.decent_lowpass(sci_data[0, 0, :], 0.995)

if args.albedo:
    solar_hdul = fits.open(args.albedo, memmap=False)
    (solar_sci_data, solar_error_data, solar_wavelengths) = read_hdul(solar_hdul)
    solar_sci_data = utils.glitch_filter(solar_sci_data, solar_error_data)
    solar_sci_data = utils.advanced_glitch_filter(solar_sci_data)
    solar_sci_data[0, 0, :] = utils.decent_lowpass(solar_sci_data[0, 0, :], 0.995)
    lol = min(solar_sci_data.shape[2], sci_data.shape[2])
    sci_data[..., 0:lol] /= solar_sci_data[..., 0:lol]

if args.normalize:
    (_, factor) = [0, np.nanpercentile(sci_data, [0, 99])[1]]
    sci_data /= factor
    error_data /= factor

if args.cut:
    begin = 3

if args.plot:
    plt.ylim([0, np.nanpercentile(sci_data, [0, 99])[1]])
    plt.plot(wavelengths, sci_data[0, 0, :], marker = "o")
    plt.plot(wavelengths, error_data[0, 0, :], marker = "o")
    plt.legend(["science", "error"])
    plt.xlabel("Wavelength (μm)")
    plt.ylabel("Amplitude (MJy)")
    plt.show()


if args.rgb:
    utils.data_to_rgb(sci_data, args.rgb)