import numpy as np
import tensorflow as tf

np.set_printoptions(threshold=np.nan)

### For cropping and splitting images while they are in numpy arrays

#
# def check_valid_split(im, num_cuts):
#     rem1 = len(im[0]) % num_cuts
#     rem2 = len(im[1]) % num_cuts
#     if rem1+rem2 == 0:
#         return True
#     else:
#         return False

def crop(im):
    im = im[2:-2,2:-2,:]
    return im

def cropandsplit_image(im, num_cuts):
    im = crop(im)
    cells = np.reshape(im, newshape=[num_cuts, num_cuts, len(im[0])/num_cuts, len(im[1])/num_cuts, -1])
    return cells


def cropandsplit_image2(im,num_cuts):
    cellsize = 80/num_cuts
    cells = np.zeros(shape=[num_cuts, num_cuts, cellsize, cellsize, 4], dtype=np.int8)
    im = crop(im)
    for i in range(num_cuts):
        for j in range(num_cuts):
            cells[i,j,:,:,:] = im[i*cellsize:((i+1)*cellsize), j*cellsize:((j+1)*cellsize),:]
    return cells

def cropandsplit_image3(im, num_cuts):
    cellsize = 80/num_cuts
    cells = np.zeros(shape=[num_cuts, num_cuts, cellsize, cellsize, 3])
    for i in range(num_cuts):
        for j in range(num_cuts):
            cells[i,j,:,:,:] = im.crop((i*cellsize, j*cellsize, (i+1)*cellsize, (j+1)*cellsize))

def average_absolute_intensity_change(cell1, cell2):
    return np.abs(np.average(cell1 - cell2))



def calculate_intensity_change(im1, im2, num_cuts):
    cells1 = cropandsplit_image(im1, num_cuts)
    cells2 = cropandsplit_image(im2, num_cuts)
    intensities = np.zeros(shape=[num_cuts, num_cuts])
    for i in range(num_cuts):
        for j in range(num_cuts):
            intensities[i][j] = average_absolute_intensity_change(cells1[i][j], cells2[i][j])

    return intensities


### For cropping and splitting images when they are tensors

def tf_crop(im):
    # im is a batch of images [None, 84, 84, 3]
    cropped_im = tf.slice(im, begin=[0,2,2,0], size=[-1,80,80,-1])
    return cropped_im

def tf_cropandsplit(im, num_cuts):
    im = tf_crop(im)
    cells = tf.reshape(im, shape=[-1, num_cuts, num_cuts, 80/num_cuts, 80/num_cuts, -1])
    return cells



def tf_calculate_intensity_change(im1, im2, num_cuts):
    cells1 = tf_cropandsplit(im1, num_cuts)
    cells2 = tf_cropandsplit(im2, num_cuts)
    intensities = tf.abs(tf.reduce_mean(cells1-cells2, axis=[3,4,5]))
    return intensities




