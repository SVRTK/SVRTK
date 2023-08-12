#! /usr/bin/python


# ============================================================================
# SVRTK : SVR reconstruction based on MIRTK
#
# Copyright 2018-... King's College London
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import sys; import os.path; __dir__ = os.path.dirname(os.path.realpath(__file__)); sys.path.insert(0, os.path.realpath(os.path.join(__dir__, '../python'))); sys.path.insert(0, os.path.realpath(os.path.join(__dir__, '../'))) # <-- added by BASIS
import sys
import os.path
import argparse

from numpy.linalg import inv, norm, det
from numpy import matmul, square, sqrt, transpose
import untangle
import SimpleITK as sitk
import numpy as np
from scipy.linalg import orthogonal_procrustes as procrustes

def readpointset(in_file):
    obj = untangle.parse(in_file)
    itemlist = obj.point_set_file.point_set.time_series.point
    pointset = []
    for s in itemlist:
        p=[]
        p.append(float(s.x.cdata))
        p.append(float(s.y.cdata))
        p.append(float(s.z.cdata))
        pointset.append(p)
    return pointset

def pointsettomatrixandcenter(pointset):
    matrix = np.matrix(np.zeros([3,3]))
    vectors = []
    if len(pointset) < 4:
        print('The pointSet should include 4 (or 5) points')
    for dim in range(2):
        v = np.array(pointset[dim*2+1]) - np.array(pointset[dim*2])
        v = v / norm(v)
        vectors.append(v)
    vectors.append(np.cross(vectors[0],vectors[1]))
    vectors[1] = np.cross(vectors[2],vectors[0])
    for dim in range(3):
        matrix[:,dim] = np.transpose(np.matrix(vectors[dim]))
    center = np.array([0,0,0])
    if len(pointset) > 4:
        center = np.array(pointset[4])
    return matrix, center


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input",
                        type=str,
                        help="input image file")
    parser.add_argument("-p", "--pointset",
                        type=str,
                        help="4 (or 5) points from MITK. order is LRIS(c) (Left, Right, Inferior, Superior, center). The center is optional")
    parser.add_argument("-o", "--output",
                        type=str,
                        help="output image file",
                        default='output.mha')
    args = parser.parse_args()

    if (args.input == None and args.pointset == None):
        parser.print_help()
        sys.exit(1)

    if not os.path.isfile(args.input):
        parser.print_help()
        sys.exit(1)

    if not os.path.isfile(args.pointset):
        parser.print_help()
        sys.exit(1)

    # read input image
    input_image =  sitk.ReadImage(args.input)
    # calculate its direction
    M1 = np.matrix(np.resize(input_image.GetDirection(), [3,3]))

    # read pointset
    pointset = readpointset(args.pointset)
    # calculate axes
    M2, center = pointsettomatrixandcenter(pointset)
    center_to_parse = center

    # calculate transformation matrix
    transform_matrix = M2 * np.transpose(M1)


    # transform_matrix = np.transpose(np.matrix(procrustes(M1, M2)[0]))
    matrix_to_parse = np.array(transform_matrix.flatten())[0,:]

    # calculate rotation center
    if len(pointset) < 5:
        center = np.transpose(np.matrix(input_image.GetOrigin())) + M1 * np.transpose(np.matrix(np.multiply(input_image.GetSize() - np.ones(3), input_image.GetSpacing()) / 2))
        center_to_parse = np.array(center)[:,0]

    print(center)

    # define transformation
    transform = sitk.Euler3DTransform()
    transform.SetMatrix(matrix_to_parse.tolist(), 1e-4)
    transform.SetTranslation([0,0,0])
    # center of rotation is the center of the image (or the user-defined one)
    transform.SetCenter(center_to_parse.tolist())

    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(input_image)
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetTransform(transform)

    moving_resampled = resample.Execute(input_image)

    moving_resampled.SetDirection([-1,0,0,0,0,1,0,1,0])
    sitk.WriteImage(moving_resampled, args.output)

if __name__ == "__main__":
    main()
