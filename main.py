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
parser.add_argument("--albedo", type=str)

args = parser.parse_args()

hdul = fits.open(args.input, memmap=False)

sci_data = np.abs(utils.fits_reorder_axes(hdul["SCI"].data))
error_data = np.abs(utils.fits_reorder_axes(hdul["ERR"].data))
wavelength = (hdul["SCI"].header["CRVAL3"], hdul["SCI"].header["CRVAL3"] + hdul["SCI"].header["CDELT3"] * hdul["SCI"].data.shape[0])

MJy_to_W = hdul["SCI"].header["CRVAL1"] * hdul["SCI"].header["CRVAL2"]

if args.info:
    hdul.info()
    for entry in itertools.zip_longest(hdul[0].header.keys(), hdul[0].header.values()):
        print(entry[0], entry[1])

    for entry in itertools.zip_longest(hdul["SCI"].header.keys(), hdul["SCI"].header.values()):
        print(entry[0], entry[1])

if args.single:
    sci_data = utils.single_pixel(sci_data, args.single[0], args.single[1])
    error_data = utils.single_pixel(error_data, args.single[0], args.single[1])

if args.glitch:
    sci_data = utils.glitch_filter(sci_data, error_data)
    error_data = utils.glitch_filter(error_data, error_data)

if args.max:
    (x, y) = utils.get_max(sci_data)
    print("max pixel", x, y)
    sci_data = utils.single_pixel(sci_data, x, y)
    error_data = utils.single_pixel(error_data, x, y)

if args.combined:
    sci_data = utils.combine_pixels(sci_data)
    error_data = utils.combine_errors(error_data)

if args.albedo:
    solar_hdul = fits.open(args.albedo, memmap=False)
    solar_sci_data = utils.fits_reorder_axes(solar_hdul["SCI"].data)
    solar_error_data = utils.fits_reorder_axes(solar_hdul["ERR"].data)

    solar_sci_data = utils.glitch_filter(solar_sci_data, solar_error_data)
    (x, y) = utils.get_max(solar_sci_data)
    solar_sci_data = utils.single_pixel(solar_sci_data, x, y)
    solar_error_data = utils.single_pixel(solar_error_data, x, y)

if args.plot:
    x = np.linspace(wavelength[0], wavelength[1], num=sci_data.shape[2])
    # sci_data /= np.max(np.abs(np.nanpercentile(sci_data, [3,97])))
    # error_data /= np.max(np.abs(np.nanpercentile(error_data, [3, 97])))
    plt.ylim(np.nanpercentile(sci_data, [5, 95]))
    plt.plot(x, sci_data[0, 0, :], marker = "o")
    plt.plot(x, error_data[0, 0, :], marker = "o")
    plt.show()

if args.rgb:
    utils.data_to_rgb(sci_data, args.rgb)