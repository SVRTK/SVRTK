import numpy as np
import gadgetron
import ismrmrd
import logging
import time
import io
import os
from datetime import datetime

from ismrmrd.meta import Meta
import itertools
import ctypes
import numpy as np
import copy
import io
import warnings
import scipy.ndimage as ndi

warnings.simplefilter('default')

from ismrmrd.acquisition import Acquisition
from ismrmrd.flags import FlagsMixin
from ismrmrd.equality import EqualityMixin
from ismrmrd.constants import *

import matplotlib.image
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

# ismrmrd_to_nifti
import sys

# from python_version import extract_ismrmrd_parameters_from_headers as param, flip_image as fi, set_nii_hdr as tools
import nibabel as nib

import src.utils as utils
from src.utils import ArgumentsTrainTestLocalisation, plot_losses_train
from src import networks as md

import numpy as np
from numpy.fft import fftshift, ifftshift, fftn, ifftn
import subprocess


def transform_kspace_to_image(k, dim=None, img_shape=None):
    """ Computes the Fourier transform from k-space to image space
    along a given or all dimensions
    :param k: k-space data
    :param dim: vector of dimensions to transform
    :param img_shape: desired shape of output image
    :returns: data in image space (along transformed dimensions)
    """
    if not dim:
        dim = range(k.ndim)

    img = fftshift(ifftn(ifftshift(k, axes=dim), s=img_shape, axes=dim), axes=dim)
    img *= np.sqrt(np.prod(np.take(img.shape, dim)))
    return img


def transform_image_to_kspace(img, dim=None, k_shape=None):
    """ Computes the Fourier transform from image space to k-space space
    along a given or all dimensions
    :param img: image space data
    :param dim: vector of dimensions to transform
    :param k_shape: desired shape of output k-space data
    :returns: data in k-space (along transformed dimensions)
    """
    if not dim:
        dim = range(img.ndim)

    k = fftshift(fftn(ifftshift(img, axes=dim), s=k_shape, axes=dim), axes=dim)
    k /= np.sqrt(np.prod(np.take(img.shape, dim)))
    return k


def get_first_index_of_non_empty_header(header):
    # if the data is under-sampled, the corresponding acquisition Header will be filled with 0
    # in order to catch valuable information, we need to catch a non-empty header
    # using the following lines

    print(np.shape(header))
    dims = np.shape(header)
    for ii in range(0, dims[0]):
        # print(header[ii].scan_counter)
        if header[ii].scan_counter > 0:
            break
    print(ii)
    return ii


def append_new_line(file_name, text_to_append):
    """Append given text as a new line at the end of file"""
    # Open the file in append & read mode ('a+')
    with open(file_name, "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)
        # If file is not empty then append '\n'
        data = file_object.read(100)
        if len(data) > 0:
            file_object.write("\n")
        # Append text at the end of file
        file_object.write(text_to_append)


def send_reconstructed_images_wcm(connection, data_array, rotx, roty, rotz, cmx, cmy, cmz, acq_header):
    # this function sends the reconstructed images with centre-of-mass stored in the image header
    # the fonction creates a new ImageHeader for each 4D dataset [RO,E1,E2,CHA]
    # copy information from the acquisitionHeader
    # fill additional fields
    # and send the reconstructed image and ImageHeader to the next gadget

    # get header info
    hdr = connection.header
    enc = hdr.encoding[0]

    if enc.encodingLimits.slice is not None:
        nslices = enc.encodingLimits.slice.maximum + 1
    else:
        nslices = 1

    if enc.encodingLimits.repetition is not None:
        nreps = enc.encodingLimits.repetition.maximum + 1
    else:
        nreps = 1

    ncoils = 1

    dims = data_array.shape

    # print(acq_header)

    # base_header = acq_header
    ndims_image = (dims[0], dims[1], dims[2])

    base_header = ismrmrd.ImageHeader()
    base_header.version = 1
    # ndims_image = (dims[0], dims[1], dims[2], dims[3])
    base_header.measurement_uid = acq_header.measurement_uid
    base_header.position = acq_header.position
    base_header.read_dir = acq_header.read_dir
    base_header.phase_dir = acq_header.phase_dir
    base_header.slice_dir = acq_header.slice_dir
    base_header.patient_table_position = acq_header.patient_table_position
    base_header.acquisition_time_stamp = acq_header.acquisition_time_stamp
    base_header.physiology_time_stamp = acq_header.physiology_time_stamp
    base_header.user_float[0] = rotx
    base_header.user_float[1] = roty
    base_header.user_float[2] = rotz
    base_header.user_float[3] = cmx
    base_header.user_float[4] = cmy
    base_header.user_float[5] = cmz

    # base_header.user_float = (rotx, roty, rotz, cmx, cmy, cmz)

    print("cmx ", base_header.user_float[3], "cmy ", base_header.user_float[4], "cmz ", base_header.user_float[5])
    # print("------ BASE HEADER ------", base_header)

    ninstances = nslices * nreps
    # r = np.zeros((dims[0], dims[1], dims[2], dims[3]))
    r = data_array
    # print(data_array.shape)
    base_header.image_type = ismrmrd.IMTYPE_COMPLEX
    image_array = ismrmrd.Image.from_array_wcm(rotx, roty, rotz, cmx, cmy, cmz, r, headers=acq_header)

    # image_array = ismrmrd.ImageHeader.from_acquisition(acq_header)
    print("..................................................................................")
    logging.info("Last slice of the repetition reconstructed - sending to the scanner...")
    connection.send(image_array)
    # print(base_header)
    logging.info("Sent!")
    print("..................................................................................")


def send_reconstructed_images(connection, data_array, acq_header):
    # the fonction creates a new ImageHeader for each 4D dataset [RO,E1,E2,CHA]
    # copy information from the acquisitionHeader
    # fill additional fields
    # and send the reconstructed image and ImageHeader to the next gadget

    # get header info
    hdr = connection.header
    enc = hdr.encoding[0]

    if enc.encodingLimits.slice is not None:
        nslices = enc.encodingLimits.slice.maximum + 1
    else:
        nslices = 1

    if enc.encodingLimits.repetition is not None:
        nreps = enc.encodingLimits.repetition.maximum + 1
    else:
        nreps = 1

    ncoils = 1

    dims = data_array.shape

    base_header = ismrmrd.ImageHeader()
    base_header.version = acq_header.version
    ndims_image = (dims[0], dims[1], dims[2], dims[3])
    base_header.channels = ncoils  # The coils have already been combined
    base_header.matrix_size = (data_array.shape[0], data_array.shape[1], data_array.shape[2])
    base_header.position = acq_header.position
    base_header.read_dir = acq_header.read_dir
    base_header.phase_dir = acq_header.phase_dir
    base_header.slice_dir = acq_header.slice_dir
    base_header.patient_table_position = acq_header.patient_table_position
    base_header.acquisition_time_stamp = acq_header.acquisition_time_stamp
    base_header.image_index = 0
    base_header.image_series_index = 0
    base_header.data_type = ismrmrd.DATATYPE_CXFLOAT
    base_header.image_type = ismrmrd.IMTYPE_COMPLEX
    base_header.repetition = acq_header.repetition

    ninstances = nslices * nreps
    r = np.zeros((dims[0], dims[1], ninstances))
    r = data_array
    base_header.image_type = ismrmrd.IMTYPE_COMPLEX
    image_array = ismrmrd.image.Image.from_array(r, headers=base_header)

    print("..................................................................................")
    logging.info("Sending reconstructed slice to the scanner...")
    connection.send(image_array)
    logging.info("Sent!")
    print("..................................................................................")


def figstring(name):
    date_path = datetime.today().strftime("%Y-%m-%d")
    timestamp = f"{datetime.today().strftime('%H-%M-%S')}"

    if not os.path.isdir("/home/sn21/data/t2-stacks/" + date_path):
        os.mkdir("/home/sn21/data/t2-stacks/" + date_path)

    final_str_nii = f"{date_path}{os.sep}{timestamp}-{name}"
    final_str_no_ext = f"{date_path}{os.sep}{timestamp}-{name}"
    return final_str_nii


def IsmrmrdToNiftiGadget(connection):
    date_path = datetime.today().strftime("%Y-%m-%d")
    timestamp = f"{datetime.today().strftime('%H-%M-%S')}"

    logging.info("Initializing data processing in Python...")
    # start = time.time()

    # get header info
    hdr = connection.header
    enc = hdr.encoding[0]

    if enc.encodingLimits.slice is not None:
        nslices = enc.encodingLimits.slice.maximum + 1
    else:
        nslices = 1

    if enc.encodingLimits.repetition is not None:
        nreps = enc.encodingLimits.repetition.maximum + 1
    else:
        nreps = 1

    if enc.encodingLimits.contrast is not None:
        ncontrasts = enc.encodingLimits.contrast.maximum + 1
    else:
        ncontrasts = 1
    print("Number of echoes =", ncontrasts)

    ncoils = 1

    # Matrix size
    eNx = int(enc.encodedSpace.matrixSize.x * 2)
    eNy = enc.encodedSpace.matrixSize.y
    eNz = enc.encodedSpace.matrixSize.z
    print("eNx = ", eNx, "eNy = ", eNy, "eNz = ", eNz)

    # Initialise a storage array
    # eNy = enc.encodingLimits.kspace_encoding_step_1.maximum + 1

    ninstances = nslices * nreps
    # print("Number of instances ", ninstances)

    im = np.zeros((eNx, eNy, nslices), dtype=np.complex64)
    print("Image Shape ", im.shape)

    # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # #  SETTING UP LOCALISER JUST ONCE # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # #

    rotx = 0.0
    roty = 0.0
    rotz = 0.0
    xcm = 0.0
    ycm = 0.0
    zcm = 0.0
    x = 0.0
    y = 0.0
    z = 0.0

    for acquisition in connection:
        # print(acquisition)
        imag = np.abs(acquisition.data.transpose(3, 2, 1, 0))
        print("Slice Dimensions ", imag.shape)

        ndim = imag.shape
        # print("ndim ", ndim)

        # Get crop image, flip and rotate to match with true Nifti image
        img = imag[:, :, :, 0]

        # Stuff into the buffer
        slice = acquisition.slice
        repetition = acquisition.repetition
        contrast = acquisition.contrast
        print("Repetition ", repetition, "Slice ", slice, "Contrast ", contrast)

        logging.info("Storing each slice into the 3D data buffer...")
        im[:, :, slice] = np.squeeze(img[:, :, 0])

        # rotx = acquisition.user_float[0]
        # roty = acquisition.user_float[1]
        # rotz = acquisition.user_float[2]
        #
        # cmx = acquisition.user_float[3]
        # cmy = acquisition.user_float[4]
        # cmz = acquisition.user_float[5]

        # if the whole stack of slices has been acquired >> apply network to the entire 3D volume
        if slice == nslices - 1:
            logging.info("All slices stored into the data buffer!")

            # SNS modified - for interleaved acquisitions!
            if nslices % 2 != 0:
                mid = int(nslices / 2) + 1
            else:
                mid = int(nslices / 2)
            # print("This is the mid slice: ", mid)
            im_corr2a = im[:, :, 0:mid]
            im_corr2b = im[:, :, mid:]

            im_corr2ab = np.zeros(np.shape(im), dtype='complex_')

            im_corr2ab[:, :, ::2] = im_corr2a
            im_corr2ab[:, :, 1::2] = im_corr2b

            print("..................................................................................")

            # Save as nib file - IMG GT
            gt_img = nib.Nifti1Image(np.abs(im_corr2ab), np.eye(4))
            name = figstring("gadgetron-fetal-brain-reconstruction")
            nib.save(gt_img, '/home/sn21/data/t2-stacks/' + name + '_img.nii.gz')
            img_tmp_info = nib.load('/home/sn21/data/t2-stacks/' + name + '_img.nii.gz')

            # os.system("docker run --rm  --mount type=bind,source=/home/sn21/data/t2-stacks,target=/home/data  "
            #           "fetalsvrtk/svrtk:auto-2.20 sh -c ' bash /home/auto-proc-svrtk/auto-brain-reconstruction.sh "
            #           "/home/data/2023-07-20 /home/data/2023-07-20-result 1 4.5 1.0 1 ; ' ")

            # List all files in the directory
            date_path = datetime.today().strftime("%Y-%m-%d")
            files = os.listdir('/home/sn21/data/t2-stacks/' + date_path)

            # Count the number of files
            num_files = len(files)

            print(f"There are {num_files} files in the directory.")

            if num_files >= 6:
                print("Launching docker now...")
                # command = '''gnome-terminal -- bash -c "docker run --rm  --mount type=bind,
                # source=/home/sn21/data/t2-stacks, target=/home/data fetalsvrtk/svrtk:auto-2.20 sh -c ' bash
                # /home/auto-proc-svrtk/auto-brain-reconstruction.sh /home/data/2023-07-20
                # /home/data/2023-07-20-result 1 4.5 1.0 1 ; ' "'''

                command = ''' gnome-terminal -- bash -c "docker run --rm  --mount type=bind,source=/home/sn21/data/t2-stacks,target=/home/data fetalsvrtk/svrtk:auto-2.20 sh -c ' bash /home/auto-proc-svrtk/auto-brain-reconstruction.sh /home/data/2023-08-10 /home/data/2023-08-10-result 1 4.5 1.0 1 ; ' " '''

                subprocess.call(['bash', '-c', command])

            send_reconstructed_images_wcm(connection, imag, rotx, roty, rotz, x, y, z, acquisition)

        else:
            send_reconstructed_images_wcm(connection, imag, rotx, roty, rotz, xcm, ycm, zcm, acquisition)

        continue
