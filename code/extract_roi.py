"""
This document implements the method inspired by:

Z. Zhang et al. "Optic disc region of interest localization in fundus image for Glaucoma detection in ARGALI." 2010 5th IEEE Conference on Industrial Electronics and Applications.
DOI: 10.1109/ICIEA.2010.5515221.
"""

import numpy as np
import pandas as pd
import math

from PIL import Image
from skimage.draw import disk


def pad_array(array):

    '''
    Pad array on shortest side to create a square array.

    Args:
        array (nd array): Rectangular array to pad.

    Returns:
        padded_array (nd array): Padded square array.
    '''

    # get array shape
    n_rows, n_cols = array.shape[:1]

    # return if already square
    if n_rows == n_cols:
        return array
    
    # create square array of larger dimension
    dim = max(n_rows,n_cols)
    padded_array = np.zeros((dim,dim))

    # get amount of padding on each side of shorter edge
    extra = dim - min(n_rows, n_cols)
    pad = extra // 2

    # correction for 1 even, 1 odd edge
    if extra % 2 != 0:
        pad = math.ceil(pad)
    else: 
        pad = int(pad)

    # create square zero array 
    padded_array = np.zeros((dim,dim,3))

    # pad horizontally
    if n_rows > n_cols:
        padded_array[:, pad-1:dim-pad-1] = array

    # pad vertically
    else:
        padded_array[pad-1:dim-pad-1] = array

    return padded_array


def circle_mask(array, radius_prop=5/6):

    '''
    Mask out outer border of retina.

    Args:
        array (nd array): Retinal image array.
        radius_prop (float): Proportion of masking circle radius compared to image radius.

    Returns:
        masked_array (nd array): Retinal image array with outer border masked out.
    '''

    # calculate center and radius coords of masking circle
    dim = array.shape[0]
    circle_center = int(dim / 2)
    circle_radius = circle_center * radius_prop

    # create masking circle array
    mask = np.zeros((dim,dim), dtype=np.uint8)
    rr,cc = disk(center=(circle_center, circle_center), radius=circle_radius, shape=None)
    mask[rr,cc] = 1

    # apply masking circle to original array
    masked_array = np.multiply(array, mask)

    return masked_array


def get_cropped_df(array, indices, top_pix_prop=0.5/100):

    '''
    Create DataFrame of top percent of brightest pixels.

    Args:
        array (nd array): Retinal image array.
        indices (nd array): Array of xy index of each pixel.
        top_pix_prop (float): Proportion of brightest pixels to keep.

    Returns:
        reduced_df (pd DataFrame): DataFrame with top proportion of brightest pixel intensities and their xy indices.
    '''

    # create df with intensity and (x,y) dim for each pixel
    df = pd.DataFrame({'value': array.flatten(), 'x': indices[:, 0], 'y': indices[:, 1]})

    # create new df with top proportion of pixel intensities
    sorted = df.sort_values(by='value', ascending=False)
    num_top_pix = math.floor(len(df) * top_pix_prop)
    reduced_df = sorted.head(num_top_pix)

    return reduced_df


def make_pix_grid(array, crop_df, grid_dim=8):

    '''
    Create square grid of tiles with number of brightest pixels per tile.

    Args:
        array (nd array): Retinal image array.
        crop_df (pd DataFrame): DataFrame with brightest pixels and their xy indices.
        grid_dim (int): Number of tiles per side of square grid.

    Returns:
        grid_arr (nd array): Grid tile array.
        tile_length (int): Side length of grid tiles.
    '''

    # create empty grid array
    grid_arr = np.zeros((grid_dim, grid_dim))

    # get side length of grid tiles
    dim = array.shape[0]
    tile_length = math.floor(dim / grid_dim)

    # iterate through pixels
    for ind in crop_df.index:
      
        # get xy coords of grid tile for pixel
        x = crop_df['x'][ind]
        row = math.floor(x / tile_length)
        if row == grid_dim: 
            row = grid_dim - 1

        y = crop_df['y'][ind]
        col = math.floor(y / tile_length)
        if col == grid_dim: 
            col = grid_dim - 1

        # iterate pixel count in grid tile 
        grid_arr[row,col] += 1

    return grid_arr, tile_length


def get_roi_coords(
        tile_idx, 
        grid_shape, 
        tile_length, 
        first_pass
    ):

    '''
    Get xy coordinates of upper left and lower right corner of ROI from tile index.

    Args:
        tile_idx (int): Flattened index of grid tile.
        grid_shape (tuple): Shape of tile grid.
        tile_length (int): Side length of grid tiles.
        first_pass (boolean): Whether to run first or second round of algorithm (changes padding).

    Returns:
        x1, y1 (int): Image coordinates for upper left corner of ROI.
        x2, y2 (int): Image coordinates for lower right corner of ROI.
    '''

    # convert tile index from 1-D to 2-D
    row, col = np.unravel_index(tile_idx, grid_shape)

    # calculate xy coords (top left, bottom right)
    upper_left_row = tile_length*row
    upper_left_col = tile_length*col
    bottom_right_row = upper_left_row + tile_length - 1
    bottom_right_col = upper_left_col + tile_length - 1

    # expand ROI size by padding with extra tiles
    # one for first pass, two for second
    pad = 1 if first_pass else 2
    upper_left_row = upper_left_row - pad*tile_length + 1
    upper_left_col = upper_left_col - pad*tile_length + 1
    bottom_right_row = bottom_right_row + pad*tile_length - 1
    bottom_right_col = bottom_right_col + pad*tile_length - 1

    return upper_left_col, upper_left_row, bottom_right_col, bottom_right_row


def algorithm_iteration(img, first_pass):

    '''
    Run a single iteration of ROI extraction algorithm.

    Args:
        img (nd array): Image from which to extract ROI.
        first_pass (boolean): Whether to run first or second round of algorithm.

    Returns:
        roi_img (nd array): ROI image array.
    '''

    # convert to numpy array
    img_array = np.array(img)

    # get square padded array
    padded_array = pad_array(img_array)

    # convert to grayscale array
    padded_img = Image.fromarray(padded_array.astype('uint8'), 'RGB')
    gray_img = padded_img.convert("L")
    gray_array = np.array(gray_img)

    # mask out retina border on first pass only
    if first_pass: 
        gray_array = circle_mask(gray_array)

    # get DataFrame of brightest pixels
    indices = np.array(list(np.ndindex(gray_array.shape)))
    crop_df = get_cropped_df(gray_array, indices)

    # get tile grid of brightest pixels
    grid_arr, tile_length = make_pix_grid(gray_array, crop_df)

    # locate brightest tile (most brightest pixels) and get coordinates
    brightest_tile = np.argmax(grid_arr)
    x1, y1, x2, y2 = get_roi_coords(
        brightest_tile, 
        grid_arr.shape, 
        tile_length, 
        first_pass
    )

    # crop image to ROI (i.e. brightest tile + padding)
    roi_img = padded_img.crop((x1, y1, x2, y2))

    return roi_img


def extract_roi(img_array):
  
    '''
    Run both rounds of ROI extraction algorithm on single image.

    Args:
        img_array (nd array): Image array from which to extract ROI.

    Returns:
        roi_img (nd array): ROI image array.
    '''

    # run two iterations of roi algorithm
    first_pass_img = algorithm_iteration(img_array, first_pass=True)
    roi_img = algorithm_iteration(first_pass_img, first_pass=False)

    return roi_img
