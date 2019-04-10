"""Utility to convert a .sdf text file into a TF Record for ScanComplete.

Input format:
<ni> <nj> <nk> (x, y, z)
<origin_x> <origin_y> <origin_z>
<dx>
<value_1> <value_2> <value_3> [...]

(ni,nj,nk) are the integer dimensions of the resulting distance field.
(origin_x,origin_y,origin_z) is the 3D position of the grid origin.
<dx> is the grid spacing.
<value_n> are the signed distance data values, in ascending order of i, then j, then k.

---------------------------
Output format:



"""
import argparse
import tensorflow as tf
import util
import numpy as np


def make_parser():
    """ Makes an argument parser for sdf_to_record.

    :return: returns an argument parser.
    """
    parser = argparse.ArgumentParser(description="Converts a .sdf text file into"
                                                 " a TFRecord for ScanComplete."
                                                 " This script will output the"
                                                 " .tfrecords file to the same"
                                                 " location as the .sdf.")
    parser.add_argument('filename', type=str, help="The input .sdf file to read.")

    return parser

def import_sdf(filename):
    """Imports sdf file and returns parsed values.

    :param filename: The file to read.
    :return: dims, origin, voxel_size, values
    """
    f = open(filename, "r")

    # Get dimensions as first line.
    dims = [int(d) for d in f.readline().split()]

    # Origin is second line.
    origin = [float(o) for o in f.readline().split()]

    # voxel size is 3rd line.
    voxel_size = float(f.readline())

    # Preallocate values.
    values = np.zeros(dims[0] * dims[1] * dims[2])

    for count, line in enumerate(f):
        values[count] = float(line)

        if count % 100000 == 0:
            print("Copied {} values! There are {} remaining.".format(count, len(values) - count))

    values = values.tolist()

    return dims, origin, voxel_size, values


def create_tfrecord(outfile, dims, values):
    """Writes output to .tfrecords file.

    :param outfile: The output file.
    :param dims: The dimensions.
    :param values: The signed values.
    :return:
    """
    with tf.python_io.TFRecordWriter(outfile) as writer:
        print("Generating dictionary.")
        out_feature = {
            'input_sdf/dim': util.int64_feature(dims),
            'input_sdf': util.float_feature(values)
        }

        print("Creating TF.Example.")
        example = tf.train.Example(features=tf.train.Features(feature=out_feature))

        print("Writing to file.")
        writer.write(example.SerializeToString())


def main():
    parser = make_parser()
    args = parser.parse_args()

    # Create output filename.
    outfile = args.filename.split('.')[0] + '.tfrecords'

    dims, origin, voxel_size, values = import_sdf(args.filename)

    # Reorganize dimensions into z, y, x for ScanComplete. the values order
    # should be correct as is.
    dims[0], dims[2] = dims[2], dims[0]

    create_tfrecord(outfile, dims, values)

    print("Done!")


if __name__ == '__main__':
    main()
