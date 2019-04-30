""" File to convert a TSDF grid in .pb file to a TF Record for use with
ScanComplete.
"""
import tensorflow as tf
import numpy as np

from google.protobuf.internal.decoder import _DecodeVarint32
import Block_pb2
import Layer_pb2
import util

from tqdm import tqdm
import struct
import argparse


class Colour:
    """Python adaptaion of the colour struct from Voxblox.
    """
    def __init__(self):
        self.r = 0
        self.g = 0
        self.b = 0
        self.a = 0


class Voxel:
    """Python adaptation of the TSDF voxel struct from Voxblox.
    """
    def __init__(self):
        self.distance = 0.0
        self.weight = 0.0
        self.colour = Colour()
        self.voxel_size = 0.0

    def set_params(self, distance, weight, colour, voxel_size):
        self.distance = distance
        self.weight = weight
        self.colour = colour
        self.voxel_size = voxel_size

    def valid(self):
        """Determines if this particular voxel is valid. Determined by checking
        the weight compared to a minimum value, then by checking the distance
        relative to a truncation parameter.

        :return: True if valid, False if not.
        """
        if self.weight < 1e-4:
            return False

        if self.distance > self.voxel_size * 3.0:
            return False

        if self.distance < -(self.voxel_size * 3.0):
            return False

        return True


class Block:
    """Python adaptation of the "Block" class  from Voxblox. To be used in the
     TSDF grid.
    """
    def __init__(self, block_proto):
        # Number of voxels per side in the block and the voxel size.
        self.voxels_per_side = block_proto.voxels_per_side
        self.voxel_size = block_proto.voxel_size

        # Origin of the block in the full TSDF grid.
        self.origin = np.array([block_proto.origin_x,
                                block_proto.origin_y,
                                block_proto.origin_z])
        self.raw_voxel_data = np.array(block_proto.voxel_data, dtype=np.uint32)

        self.block_proto = block_proto

        # Define variables now but populate later.
        self.voxel_data = [Voxel() for _ in range(self.voxels_per_side ** 3)]
        self.block_index = np.empty(len(self.origin), dtype=np.int64)

        self.deserialize_voxel_data()
        self.set_block_index_from_origin()

    def deserialize_voxel_data(self):
        """Deserialize the raw voxel data into the appropriate values. This is
        adapted from:

        `void Voxblox::Block<TsdfVoxel>::deserializeFromIntegers(uint32_t)`
        """
        # Loop through raw voxel data. Serialized as 3 "packets" per voxel, so
        # we can increment by 3.
        for i in range(len(self.voxel_data)):
            distance = struct.unpack("f", self.raw_voxel_data[3 * i])[0]
            weight = struct.unpack("f", self.raw_voxel_data[3 * i + 1])[0]

            raw_colour = self.raw_voxel_data[3 * i + 2]

            colour = Colour()
            colour.r = np.uint8(raw_colour >> 24)
            colour.g = np.uint8((raw_colour & 0x00FF0000) >> 16)
            colour.b = np.uint8((raw_colour & 0x0000FF00) >> 8)
            colour.a = raw_colour & 0x000000FF

            self.voxel_data[i].set_params(distance, weight, colour, self.voxel_size)

    def set_block_index_from_origin(self):
        """Computes the block index in the grid using the origin of the block,
        and sets the value in the block_index member variable.

        From Voxblox: "This function is safer than `getGridIndexFromPoint`
        because it assumes we pass in not an arbitrary point in the grid cell,
        but the ORIGIN. This way we can avoid the floating point precision issue
        that arises for calls to `getGridIndexFromPoint` for arbitrary points
        near the border of the grid cell.

        Adapted from
        `voxblox::IndexType getGridIndexFromOriginPoint(const Point&,
                                                        const FloatingPoint)`
        """
        block_size = self.voxel_size * self.voxels_per_side
        block_size_inv = 1.0 / block_size

        self.block_index[0] = int(round(self.origin[0] * block_size_inv))
        self.block_index[1] = int(round(self.origin[1] * block_size_inv))
        self.block_index[2] = int(round(self.origin[2] * block_size_inv))

    def is_valid_voxel_index(self, voxel_index):
        """Determines whether nor not this voxel index lies in the current block

        :param voxel_index: The voxel index.
        :return: True if it is valid, False if not.
        """
        if voxel_index[0] < 0 or voxel_index[0] >= self.voxels_per_side:
            return False

        if voxel_index[1] < 0 or voxel_index[1] >= self.voxels_per_side:
            return False

        if voxel_index[2] < 0 or voxel_index[2] >= self.voxels_per_side:
            return False

        return True


class Layer:
    """Python adaptation of the voxblox::Layer class. Contains bookkeeping
    information for the TSDF grid.
    """
    def __init__(self, layer_proto):
        # Layer params
        self.voxel_size = layer_proto.voxel_size
        self.voxels_per_side = layer_proto.voxels_per_side

        # Block params
        self.block_size = self.voxel_size * self.voxels_per_side
        self.block_size_inv = 1.0 / self.block_size

        self.layer_proto = layer_proto

        # Blocks within the layer
        self.blocks = []

        # Dictionary storing block hash key and the index in the above array
        self.block_map = {}

        self.min_block_index = np.empty(3, dtype=np.int64)
        self.max_block_index = np.empty(3, dtype=np.int64)
        self.indices_initialized = False

    def add_block(self, block):
        if not self.indices_initialized:
            self.min_block_index = np.copy(block.block_index)
            self.max_block_index = np.copy(block.block_index)

            self.indices_initialized = True
        else:
            # Check values in x
            if block.block_index[0] < self.min_block_index[0]:
                self.min_block_index[0] = np.copy(block.block_index[0])
            elif block.block_index[0] > self.max_block_index[0]:
                self.max_block_index[0] = np.copy(block.block_index[0])

            # Check values in y
            if block.block_index[1] < self.min_block_index[1]:
                self.min_block_index[1] = np.copy(block.block_index[1])
            elif block.block_index[1] > self.max_block_index[1]:
                self.max_block_index[1] = np.copy(block.block_index[1])

            # Check values in z
            if block.block_index[2] < self.min_block_index[2]:
                self.min_block_index[2] = np.copy(block.block_index[2])
            elif block.block_index[2] > self.max_block_index[2]:
                self.max_block_index[2] = np.copy(block.block_index[2])

        self.blocks.append(block)

    def shift_origin(self):
        """Move the origin of the entire grid to be at 0, 0, 0. Done by
        shifting each block by the value of min_block.
        """
        # Shift each block
        for idx, block in enumerate(self.blocks):
            block.block_index -= self.min_block_index

            # Generate block hash for each block index.
            key = util.get_block_hash(block.block_index)

            if not str(key) in self.block_map:
                self.block_map[str(key)] = idx
            else:
                raise RuntimeError("Key already exists for block index!")

        # Shift limits
        self.max_block_index -= self.min_block_index
        self.min_block_index -= self.min_block_index

    @staticmethod
    def check_inner_voxel_validity(voxel_index, block):
        """Sets "invalid" voxels inside the block to -inf. Invalid voxels are
        determined in the same manner as
        `voxblox::MeshIntegrator::extractMeshInsideBlock`.

        We will check the validity of all neighbours around a block. If any of
        the neighbours is invalid, then we set the value of the voxel to -inf.

        The validity of a voxel is based on its weight. If the weight is less
        than 1e-4, then it's invalid.

        :param voxel_index: The index of the voxel under test.
        :param block: The current block.
        :return: True if valid, False if not.
        """
        cube_index_offsets = util.create_cube_index_offset()

        all_neighbours_observed = True
        for i in range(8):
            corner_index = voxel_index + cube_index_offsets[i]
            corner = block.voxel_data[util.get_linear_voxel_index(corner_index,
                                                                 block.voxels_per_side)]

            if not corner.valid():
                all_neighbours_observed = False
                break

        return all_neighbours_observed

    def check_edge_voxel_validity(self, voxel_index, block):
        """Same as self.check_inner_voxel_validity(), however there is slightly
        different logic for voxels that lay on the edge of blocks. Invalid
        voxels are determined in the same manner as
        `voxblox::MeshIntegrator::extractMeshOnBorder`

        :return:
        """
        cube_index_offsets = util.create_cube_index_offset()

        all_neighbours_observed = True

        for i in range(8):
            corner_index = voxel_index + cube_index_offsets[i]

            if block.is_valid_voxel_index(corner_index):
                corner = block.voxel_data[util.get_linear_voxel_index(corner_index,
                                                                     block.voxels_per_side)]

                if not corner.valid():
                    all_neighbours_observed = False
                    break
            else:
                # We now have to access a different block
                block_offset = np.zeros(3, dtype=np.int)

                for j in range(3):
                    if corner_index[j] < 0:
                        block_offset[j] = -1
                        corner_index[j] = corner_index[j] \
                                          + block.voxels_per_side
                    elif corner_index[j] >= block.voxels_per_side:
                        block_offset[j] = 1
                        corner_index[j] = corner_index[j] \
                                          - block.voxels_per_side

                # Extract the neighbouring block.
                neighbour_block_index = block.block_index + block_offset

                # Make sure this neighbouring block actually exists.
                key = util.get_block_hash(neighbour_block_index)

                if str(key) in self.block_map:
                    idx = self.block_map[str(key)]
                    neighbour_block = self.blocks[idx]

                    # Make sure that the corner voxel exists in the neighbour
                    if neighbour_block.is_valid_voxel_index(corner_index):
                        corner = neighbour_block.voxel_data[
                            util.get_linear_voxel_index(corner_index,
                                                        neighbour_block.voxels_per_side)]

                        if not corner.valid():
                            all_neighbours_observed = False
                            break
                else:
                    all_neighbours_observed = False
                    break

        return all_neighbours_observed


def make_parser():
    """Creates an argument parser for tsdf_to_record.

    :return: Returns the argument parser.
    """
    parser = argparse.ArgumentParser("Script to convert a TSDF grid exported "
                                     "from voxblox (.pb) into a TF Record "
                                     "format for ScanComplete.")
    parser.add_argument('tsdf_pb', type=str, help="Path to the TSDF grid as a "
                                                  "Protobuf (.pb) file.")

    return parser


def import_pb(tsdf_pb):
    """Imports the TSDF grid from the protobuf file.

    :param tsdf_pb: The filepath to the protobuf file.
    :return:
    """
    with open(tsdf_pb, 'rb') as f:
        data = f.read()
        pos = 0

        # First message is just the number of messages.
        num_msgs, start_pos = _DecodeVarint32(data, pos)
        pos = start_pos

        # Second message should be the Layer header.
        print('Extracting layer header...')
        msg_len, pos = _DecodeVarint32(data, pos)
        layer_proto = Layer_pb2.LayerProto()
        layer_proto.ParseFromString(data[pos:pos + msg_len])
        pos += msg_len

        layer = Layer(layer_proto)

        # Can now extract the remaining blocks.
        # There should be num_msgs - 1 blocks.
        print('Extracting blocks...')
        for _ in tqdm(range(num_msgs - 1)):
            msg_len, pos = _DecodeVarint32(data, pos)
            block_proto = Block_pb2.BlockProto()
            block_proto.ParseFromString(data[pos:pos + msg_len])
            pos += msg_len

            block = Block(block_proto)

            layer.add_block(block)

    # Now that we have read the layer, shift it to have origin at 0, 0, 0
    layer.shift_origin()

    return layer


def convert_layer_to_record(filename, layer):
    """Converts the extracted layer into the TF record format.

    :param filename: The output file to save the record to.
    """
    print('Converting layer to TF Record...')

    # Start by calculating the max size of the flattened array.
    grid_dims = util.get_global_voxel_index(layer.max_block_index, layer.voxels_per_side, layer.voxels_per_side)
    size = grid_dims[0] * grid_dims[1] * grid_dims[2]

    # Create record array. Unitialized values are -inf
    values = np.zeros(size) - np.inf

    # Loop through blocks
    print('Copying values from blocks...')
    for block in tqdm(layer.blocks):
        # Handle the "inner" voxels.
        for k in range(block.voxels_per_side - 1):
            for j in range(block.voxels_per_side - 1):
                for i in range(block.voxels_per_side - 1):
                    voxel_index = np.array([i, j, k])

                    valid = layer.check_inner_voxel_validity(voxel_index, block)

                    # Get the voxel
                    voxel = block.voxel_data[util.get_linear_voxel_index(voxel_index, block.voxels_per_side)]

                    if not valid:
                        voxel.distance = -np.inf

                    # Get the global voxel index and convert to a global value.
                    global_index = util.get_global_voxel_index(block.block_index, voxel_index, block.voxels_per_side)
                    global_linear_index = util.get_global_linear_voxel_index(global_index, grid_dims)

                    # Place voxel data into the record at the global index.
                    values[global_linear_index] = voxel.distance / voxel.voxel_size

        # Now calculate validity on the max X plane.
        # Takes care of two edges: (x_max, y_max, z) & (x_max, y, z_max)
        for k in range(block.voxels_per_side):
            for j in range(block.voxels_per_side):
                voxel_index = np.array([block.voxels_per_side - 1, j, k])

                valid = layer.check_edge_voxel_validity(voxel_index, block)

                # Get the voxel
                voxel = block.voxel_data[util.get_linear_voxel_index(voxel_index, block.voxels_per_side)]

                if not valid:
                    voxel.distance = -np.inf

                # Get the global voxel index and convert to a global value.
                global_index = util.get_global_voxel_index(block.block_index,
                                                           voxel_index,
                                                           block.voxels_per_side)
                global_linear_index = util.get_global_linear_voxel_index(
                    global_index, grid_dims)

                # Place voxel data into the record at the global index.
                values[global_linear_index] = voxel.distance

        # Now calculate validity on the max Y plane.
        # Takes care of edge: (x, y_max, z_max) without corner (x_max, y_max, z_max)
        for k in range(block.voxels_per_side):
            for i in range(block.voxels_per_side - 1):
                voxel_index = np.array([i, block.voxels_per_side - 1, k])

                valid = layer.check_edge_voxel_validity(voxel_index,
                                                        block)

                # Get the voxel
                voxel = block.voxel_data[
                    util.get_linear_voxel_index(voxel_index,
                                                block.voxels_per_side)]

                if not valid:
                    voxel.distance = -np.inf

                # Get the global voxel index and convert to a global value.
                global_index = util.get_global_voxel_index(
                    block.block_index,
                    voxel_index,
                    block.voxels_per_side)
                global_linear_index = util.get_global_linear_voxel_index(
                    global_index, grid_dims)

                # Place voxel data into the record at the global index.
                values[global_linear_index] = voxel.distance

        # Now calculate validity on the max Z plane.
        for j in range(block.voxels_per_side - 1):
            for i in range(block.voxels_per_side - 1):
                voxel_index = np.array([i, j, block.voxels_per_side - 1])

                valid = layer.check_edge_voxel_validity(voxel_index, block)

                # Get the voxel
                voxel = block.voxel_data[
                    util.get_linear_voxel_index(voxel_index,
                                                block.voxels_per_side)]

                if not valid:
                    voxel.distance = -np.inf

                # Get the global voxel index and convert to a global value.
                global_index = util.get_global_voxel_index(block.block_index,
                                                           voxel_index,
                                                           block.voxels_per_side)
                global_linear_index = util.get_global_linear_voxel_index(
                    global_index, grid_dims)

                # Place voxel data into the record at the global index.
                values[global_linear_index] = voxel.distance

    # Rearrange grid dimensions for the record format.
    grid_dims[0], grid_dims[2] = grid_dims[2], grid_dims[0]

    create_tfrecord(filename, grid_dims, values)


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
            'input_sdf/dim': util.int64_feature(dims.tolist()),
            'input_sdf': util.float_feature(values.tolist())
        }

        print("Creating TF.Example.")
        example = tf.train.Example(features=tf.train.Features(feature=out_feature))

        print("Writing to file.")
        writer.write(example.SerializeToString())


def main():
    parser = make_parser()
    args = parser.parse_args()

    outfile = args.tsdf_pb.split('.')[0] + '.tfrecords'

    layer = import_pb(args.tsdf_pb)

    # Convert the layer to the TFRecord and write to file.
    convert_layer_to_record(outfile, layer)

    print('TF Record completed! It can be found at {}'.format(outfile))


if __name__ == '__main__':
    main()
