"""
python driver for nvdla-ila simulator
"""

import json
import math
import os
import numpy as np
import subprocess
import argparse
import timeit


class pdp_driver:
    def __init__(self, op_name, inp_shape, out_shape, pool_size, strides, dilation, padding, inp_data_format, out_format=None):
        self.op_name = op_name
        # 4d array - 1 at start because only batch number 1 supported
        self.orig_inp_shape = [1] + inp_shape
        self.orig_out_shape = out_shape
        self.inp_data_format = inp_data_format          # string
        self.pool_size = pool_size                      # 2d array
        self.strides = strides                          # 2d array
        self.dilation = dilation                        # 2d array
        self.padding = padding                          # 2d array
        self.inp_data_format = inp_data_format          # string
        self.out_format = out_format                    # string
        if self.out_format is None:
            self.out_format = self.inp_data_format
        # always convert to same internal format N = 1, then height, then width, then channels
        self.desired_inp_format = 'NHWC'
        self.inp_matrix = None
        self.ila_asm = []
        self.inp1_mem_length = 0                        # measured in ints

    def produce_write_asm(self):
        """
        Produce asm for writing data to the RDMA memory of the NVDLA
        """
        self.produce_write_asm_data_cube(
            self.inp_matrix.shape, self.inp_matrix, 0)

    def produce_write_asm_data_cube(self, cube_shape, cube_list, addr_offset):
        """
        Produce asm for writing given data cube
        """
        # dbbif configurable to width of 32, 64, 128, 256 or 512-bits
        # ie nb of ints is               2,   4,   8,  16 or  32
        dbbif_width = 512
        ints_per_line = int(dbbif_width / 16)
        n, h, w, c = cube_shape

        all_data_str = []
        full_data_str = ''
        numbers_per_block = 64
        for n_n in range(n):
            for n_c_large in range(0, math.ceil(c / numbers_per_block)):
                for n_h in range(h):
                    for n_w in range(w):
                        # 64 channels per atom for conv
                        # print(n_h, n_w)
                        for n_c_small in range(numbers_per_block):
                            # keep track of ints written
                            if addr_offset == 0:
                                self.inp1_mem_length += 1

                            # add on int or pad with 0s
                            ch_idx = n_c_large*numbers_per_block + n_c_small
                            if ch_idx < c:
                                num_str = cube_list[n_n][n_h][n_w][ch_idx].tobytes(
                                ).hex()
                                full_data_str = num_str + full_data_str
                            else:
                                full_data_str = '0000' + full_data_str

                            # purge line if full
                            if len(full_data_str) == ints_per_line * 4:
                                all_data_str.append(full_data_str)
                                full_data_str = ''

        for data_str_idx, full_data_str in enumerate(all_data_str):
            self.ila_asm.append({
                'name': 'VirMemWr',
                'addr': hex(data_str_idx * ints_per_line * 2 + addr_offset),
                'data': f"0x{full_data_str}",
            })

    def produce_main_asm(self):
        """
        produce asm fragment with reg configuration, enable and datapath 
        """
        # self.op_name = avg_pool2d or max_pool2d
        # --- PRE-SETUP ---
        self.ila_asm.append({
            'name': 'PDP_RDMA_S_POINTER',
            'NVDLA_PDP_RDMA_PRODUCER': 0,
            'NVDLA_PDP_RDMA_CONSUMER': 0
        })
        self.ila_asm.append({
            'name': 'PDP_S_POINTER',
            'NVDLA_PDP_PRODUCER': 0,
            'NVDLA_PDP_CONSUMER': 0
        })
        # --- SETUP ---
        # Set up PDP RDMA
        n, h, w, c = self.inp_matrix.shape
        # 0xc00c - input width
        self.ila_asm.append({
            'name': 'PDP_RDMA_D_DATA_CUBE_IN_WIDTH',
            'NVDLA_PDP_RDMA_CUBE_IN_WIDTH': w
        })
        # 0xc010 - input height
        self.ila_asm.append({
            'name': 'PDP_RDMA_D_DATA_CUBE_IN_HEIGHT',
            'NVDLA_PDP_RDMA_CUBE_IN_HEIGHT': h
        })
        # 0xc014 - input channels
        self.ila_asm.append({
            'name': 'PDP_RDMA_D_DATA_CUBE_IN_CHANNEL',
            'NVDLA_PDP_RDMA_CUBE_IN_CHANNEL': c
        })
        # 0xc018 - input data cube from RAM not CACC
        self.ila_asm.append({
            'name': 'PDP_RDMA_D_FLYING_MODE',
            'NVDLA_PDP_RDMA_FLYING_MODE': 0
        })
        # 0xc01c - base address of input data cube
        self.ila_asm.append({
            'name': 'PDP_RDMA_D_SRC_BASE_ADDR_LOW',
            'NVDLA_PDP_RDMA_SRC_BASE_ADDR_LOW': 0
        })
        # 0xc020
        self.ila_asm.append({
            'name': 'PDP_RDMA_D_SRC_BASE_ADDR_HIGH',
            'NVDLA_PDP_RDMA_SRC_BASE_ADDR_HIGH': 0
        })
        # 0xc024 - line stride of input data cube
        self.ila_asm.append({
            'name': 'PDP_RDMA_D_SRC_LINE_STRIDE',
            'NVDLA_PDP_RDMA_SRC_LINE_STRIDE': int(w*16*2/2**5)
        })
        # 0xc028 - surface stride of input data cube
        self.ila_asm.append({
            'name': 'PDP_RDMA_D_SRC_SURFACE_STRIDE',
            'NVDLA_PDP_RDMA_SRC_SURFACE_STRIDE': int(w*h*16*2/2**5)
        })
        # 0xc02c - source data cube from DRAM
        self.ila_asm.append({
            'name': 'PDP_RDMA_D_SRC_RAM_CFG',
            'NVDLA_PDP_RDMA_SRC_RAM_TYPE': 1
        })
        # 0xc030 - input data cube is int16
        self.ila_asm.append({
            'name': 'PDP_RDMA_D_DATA_FORMAT',
            'NVDLA_PDP_RDMA_DATA_FORMAT': 1
        })
        # 0xc034 - split mode is off - ie not split + from RAM
        self.ila_asm.append({
            'name': 'PDP_RDMA_D_OPERATION_MODE_CFG',
            'NVDLA_PDP_RDMA_SPLIT_NUM': 1
        })
        # 0xc038 - pooling kernel configs
        self.ila_asm.append({
            'name': 'PDP_RDMA_D_POOLING_KERNEL_CFG',
            'NVDLA_PDP_RDMA_KERNEL_WIDTH': self.pool_size[1],
            'NVDLA_PDP_RDMA_KERNEL_STRIDE_WIDTH': self.strides[1]
        })
        # 0xc03c - padding config
        self.ila_asm.append({
            'name': 'PDP_RDMA_D_POOLING_PADDING_CFG',
            'NVDLA_PDP_RDMA_PAD_WIDTH': self.padding[1]
        })
        # 0xc040 - unused partial width cfg if data cube is too big (assumed to not be)
        self.ila_asm.append({
            'name': 'PDP_RDMA_D_PARTIAL_WIDTH_IN',
            'NVDLA_PDP_RDMA_PARTIAL_WIDTH_IN_FIRST': 0,
            'NVDLA_PDP_RDMA_PARTIAL_WIDTH_IN_LAST': 0,
            'NVDLA_PDP_RDMA_PARTIAL_WIDTH_IN_MID': 0,
        })
        # 0xc044 - turn off performance counting registers
        self.ila_asm.append({
            'name': 'PDP_RDMA_D_PERF_ENABLE',
            'NVDLA_PDP_RDMA_DMA_EN': 0
        })

        # Set up PDP core
        # 0xd00c - input width
        self.ila_asm.append({
            'name': 'PDP_D_DATA_CUBE_WIDTH',
            'NVDLA_PDP_CUBE_IN_WIDTH': w
        })
        # 0xd010 - input height
        self.ila_asm.append({
            'name': 'PDP_D_DATA_CUBE_HEIGHT',
            'NVDLA_PDP_CUBE_IN_HEIGHT': h
        })
        # 0xd014 - input channels
        self.ila_asm.append({
            'name': 'PDP_D_DATA_CUBE_CHANNEL',
            'NVDLA_PDP_CUBE_IN_CHANNEL': c
        })
        out_n, out_h, out_w, out_c = self.orig_out_shape
        # 0xd018 - output cube width
        self.ila_asm.append({
            'name': 'PDP_D_DATA_CUBE_OUT_WIDTH',
            'NVDLA_PDP_CUBE_OUT_WIDTH': out_w
        })
        # 0xd01c - output cube height
        self.ila_asm.append({
            'name': 'PDP_D_DATA_CUBE_OUT_HEIGHT',
            'NVDLA_PDP_CUBE_OUT_HEIGHT': out_h
        })
        # 0xd020 - output cube channels
        self.ila_asm.append({
            'name': 'PDP_D_DATA_CUBE_OUT_CHANNEL',
            'NVDLA_PDP_CUBE_OUT_CHANNEL': out_c
        })
        # 0xd024 - configure:
        #   what type of pooling using (avg, min or max)
        #   fly (no - see above)
        #   split (not using so unused)
        pool_map = {
            'avg_pool2d': 0,
            'max_pool2d': 2
        }
        pooling_mode = pool_map[self.op_name]
        self.ila_asm.append({
            'name': 'PDP_D_OPERATION_MODE_CFG',
            'NVDLA_PDP_POOLING_MODE': pooling_mode,
            'NVDLA_PDP_FLYING_MODE': 0,
            'NVDLA_PDP_SPLIT_NUM': 0
        })
        # 0xd028 - floating point nan --> zero is off (unused as only doing int16)
        self.ila_asm.append({
            'name': 'PDP_D_NAN_FLUSH_TO_ZERO',
            'NVDLA_PDP_NAN_FLUSH_TO_ZERO': 0
        })
        # 0xd02c - input partial width (unused like earlier)
        self.ila_asm.append({
            'name': 'PDP_D_PARTIAL_WIDTH_IN',
            'NVDLA_PDP_PARTIAL_WIDTH_IN_FIRST': 0,
            'NVDLA_PDP_PARTIAL_WIDTH_IN_LAST': 0,
            'NVDLA_PDP_PARTIAL_WIDTH_IN_MID': 0
        })
        # 0xd030 - output partial width (unused like earlier)
        self.ila_asm.append({
            'name': 'PDP_D_PARTIAL_WIDTH_OUT',
            'NVDLA_PDP_PARTIAL_WIDTH_OUT_FIRST': 0,
            'NVDLA_PDP_PARTIAL_WIDTH_OUT_LAST': 0,
            'NVDLA_PDP_PARTIAL_WIDTH_OUT_MID': 0
        })
        # 0xd034 - configure pooling kernel
        self.ila_asm.append({
            'name': 'PDP_D_POOLING_KERNEL_CFG',
            'NVDLA_PDP_KERNEL_WIDTH': self.pool_size[1],
            'NVDLA_PDP_KERNEL_HEIGHT': self.pool_size[0],
            'NVDLA_PDP_KERNEL_STRIDE_WIDTH': self.strides[1],
            'NVDLA_PDP_KERNEL_STRIDE_HEIGHT': self.strides[0]
        })
        # 0xd038 - for average can't do divide so need to do reciprocal in driver
        self.ila_asm.append({
            'name': 'PDP_D_RECIP_KERNEL_WIDTH',
            'NVDLA_PDP_RECIP_KERNEL_WIDTH': int(2**16 / self.pool_size[1])
        })
        # 0xd03c - same as above
        self.ila_asm.append({
            'name': 'PDP_D_RECIP_KERNEL_HEIGHT',
            'NVDLA_PDP_RECIP_KERNEL_HEIGHT': int(2**16 / self.pool_size[0])
        })
        # 0xd040 - configure padding (up to 7)
        if self.padding[0] > 7 or self.padding[1] > 7:
            raise NotImplementedError('padding greater than 7 not supported')
        self.ila_asm.append({
            'name': 'PDP_D_POOLING_PADDING_CFG',
            'NVDLA_PDP_PAD_LEFT': self.padding[1],
            'NVDLA_PDP_PAD_TOP ': self.padding[0],
            'NVDLA_PDP_PAD_RIGHT': self.padding[1],
            'NVDLA_PDP_PAD_BOTTOM': self.padding[0]
        })
        # 0xd044 - value for padding in relay is always 0
        self.ila_asm.append({
            'name': 'PDP_D_POOLING_PADDING_VALUE_1_CFG',
            'NVDLA_PDP_PAD_VALUE_1X': 0
        })
        self.ila_asm.append({
            'name': 'PDP_D_POOLING_PADDING_VALUE_2_CFG',
            'NVDLA_PDP_PAD_VALUE_2X': 0
        })
        self.ila_asm.append({
            'name': 'PDP_D_POOLING_PADDING_VALUE_3_CFG',
            'NVDLA_PDP_PAD_VALUE_3X': 0
        })
        self.ila_asm.append({
            'name': 'PDP_D_POOLING_PADDING_VALUE_4_CFG',
            'NVDLA_PDP_PAD_VALUE_4X': 0
        })
        self.ila_asm.append({
            'name': 'PDP_D_POOLING_PADDING_VALUE_5_CFG',
            'NVDLA_PDP_PAD_VALUE_5X': 0
        })
        self.ila_asm.append({
            'name': 'PDP_D_POOLING_PADDING_VALUE_6_CFG',
            'NVDLA_PDP_PAD_VALUE_6X': 0
        })
        self.ila_asm.append({
            'name': 'PDP_D_POOLING_PADDING_VALUE_7_CFG',
            'NVDLA_PDP_PAD_VALUE_7X': 0
        })
        # 0xd060 - address of input data
        self.ila_asm.append({
            'name': 'PDP_D_SRC_BASE_ADDR_LOW',
            'NVDLA_PDP_SRC_BASE_ADDR_LOW': 0
        })
        # 0xd064 - address of input data
        self.ila_asm.append({
            'name': 'PDP_D_SRC_BASE_ADDR_HIGH',
            'NVDLA_PDP_SRC_BASE_ADDR_HIGH': 0
        })
        # 0xd068 - input data line stride
        self.ila_asm.append({
            'name': 'PDP_D_SRC_LINE_STRIDE',
            'NVDLA_PDP_SRC_LINE_STRIDE': int(w*16*2/2**5)
        })
        # 0xd06c - input data surface stride
        self.ila_asm.append({
            'name': 'PDP_D_SRC_SURFACE_STRIDE',
            'NVDLA_PDP_SRC_SURFACE_STRIDE': int(h*w*16*2/2**5)
        })
        # 0xd070 - address of output data
        self.ila_asm.append({
            'name': 'PDP_D_DST_BASE_ADDR_LOW',
            'NVDLA_PDP_DST_BASE_ADDR_LOW': int(512000/2**5)
        })
        # 0xd074 - address of output data
        self.ila_asm.append({
            'name': 'PDP_D_DST_BASE_ADDR_HIGH',
            'NVDLA_PDP_DST_BASE_ADDR_HIGH': 0
        })
        # 0xd078 - output data line stride
        self.ila_asm.append({
            'name': 'PDP_D_DST_LINE_STRIDE',
            'NVDLA_PDP_DST_LINE_STRIDE': int(out_w*16*2/2**5)
        })
        # 0xd07c - output data surface stride
        self.ila_asm.append({
            'name': 'PDP_D_DST_SURFACE_STRIDE',
            'NVDLA_PDP_DST_SURFACE_STRIDE': int(out_h*out_w*16*2/2**5)
        })
        # 0xd080 - use external DRAM for output
        self.ila_asm.append({
            'name': 'PDP_D_DST_RAM_CFG',
            'NVDLA_PDP_DST_RAM_TYPE': 1
        })
        # 0xd084 - int16 for input
        self.ila_asm.append({
            'name': 'PDP_D_DATA_FORMAT',
            'NVDLA_PDP_INPUT_DATA': 1
        })
        # 0xd094 - turn off perf counting
        self.ila_asm.append({
            'name': 'PDP_D_PERF_ENABLE',
            'NVDLA_PDP_PERF_EN': 0
        })

        # --- Enable PDP and PDP RDMA cores --- 
        self.ila_asm.append({
            'name': 'PDP_RDMA_D_OP_ENABLE',
            'NVDLA_PDP_RDMA_D_OP_ENABLE': 1
        })
        self.ila_asm.append({
            'name': 'PDP_D_OP_ENABLE',
            'NVDLA_PDP_D_OP_ENABLE': 1
        })

        # --- COMPUTATION ---
        # can process 4 int16s per call
        raise NotImplementedError('PDP simulator not implemented yet')

    def produce_read_asm(self):
        """
        produce asm for reading data from the RDMA memory of the NVDLA
        """
        # assumes other cores have run and that the result has been stored back in memory from those - incorrect currently
        dbbif_width = 512  # dbbif configurable to width of 32, 64, 128, 256 or 512-bits
        ints_per_line = int(dbbif_width / 16)
        n, h, w, c = self.orig_out_shape
        for i in range(int(np.ceil(h*w*16 / ints_per_line))):
            self.ila_asm.append({
                'name': 'VirMemRd',
                'addr': hex(i * ints_per_line * 2)
            })

    def produce_asm_all(self):
        self.produce_write_asm()
        self.produce_main_asm()
        self.produce_read_asm()
        self.ila_asm.append({
            'name': 'DONE'
        })
        self.ila_asm = {'asm': self.ila_asm}
        with open(f'./test/{self.op_name}/ila_asm.json', 'w') as fout:
            json.dump(self.ila_asm, fout, indent=2)

    def transpose_new_order(self, inp_format, out_format):
        new_order = []
        for c in out_format:
            if c in inp_format:
                new_order.append(inp_format.index(c))
            else:
                raise Exception(
                    f"Char {c} from {out_format} not found in {inp_format}. Cannot convert matrix to new form")
        return new_order

    def transform_matrix(self, inp_format, out_format, inp_matrix):
        """Reorder NCWH to NWHC and other orders"""
        # based on
        # https://stackoverflow.com/questions/49558818/change-image-channel-ordering-from-channel-first-to-last#:~:text=You%20need%20the%20np.transpose%20method%20like%20this%3A%20training_data,%280%2C%202%2C3%2C1%29%20The%20same%20for%20the%20other%20ones
        if len(inp_format) != len(inp_matrix.shape):
            raise Exception(
                f"Input format is wrong length ({len(out_format)}) vs matrix shape length ({len(inp_matrix.shape)})")

        if len(out_format) != len(inp_matrix.shape):
            raise Exception(
                f"Out format is wrong length ({len(out_format)}) vs matrix shape length ({len(inp_matrix.shape)})")

        new_order = self.transpose_new_order(inp_format, out_format)
        return np.transpose(inp_matrix, new_order)

    def collect_data_in(self):
        """
        collect relay data from files
        """
        print('\n--------------------------------------------------------------')
        print('\tcollecting input data')
        print('--------------------------------------------------------------\n')
        # input data
        with open(f'./data/{self.op_name}/inp.json', 'r') as fin:
            self.inp_matrix = np.array(json.load(fin)).astype(
                'int16').reshape(self.orig_inp_shape)
            self.inp_matrix = self.transform_matrix(
                self.inp_data_format, self.desired_inp_format, self.inp_matrix)
        print('Max of input:', np.max(self.inp_matrix))

        # output shape - ensure in expected order
        output_shape_reordering = self.transpose_new_order(
            self.out_format, self.desired_inp_format)
        self.orig_out_shape = [self.orig_out_shape[idx]
                               for idx in output_shape_reordering]

    def produce_prog_frag(self):
        print('\n--------------------------------------------------------------')
        print('\tgenerate prog_frag.json for ILA simulator')
        print('--------------------------------------------------------------\n')
        raise NotImplementedError(
            'PDP simulator and thus converter not implemented yet')
        # self.ila_cvtr = PDPConverter(
        #     f'./test/{self.op_name}/ila_asm.json', self.op_name)
        # self.ila_cvtr.dump_ila_prog_frag(
        #     f'./test/{self.op_name}/ila_prog_frag_input.json')

    def invoke_ila_simulator_and_collect_results(self):
        print('\n--------------------------------------------------------------')
        print('\tinvoking NVDLA ILA simulator and collecting result')
        print('--------------------------------------------------------------\n')
        raise NotImplementedError('PDP simulator not implemented yet')
        start_time = timeit.default_timer()
        cmd = [
            "pdp_sim_driver",
            f'./test/{self.op_name}/ila_prog_frag_input.json',
            f'./test/{self.op_name}/ila_prog_frag_out.json'
        ]
        print('Running command', " ".join(cmd))
        subprocess.run(cmd)

        sim_output = []
        with open(f'./test/{self.op_name}/ila_prog_frag_out.json', 'r') as fin:
            sim_output = []

        # iterate through output ... nb don't know output shape so have to assemble in known format then convert to correct
        n, h, w, k = self.orig_out_shape
        if n > 1:
            raise Exception('Batch sizes over 1 not supported')

        nhwk_out = np.zeros(self.orig_out_shape, 'int16')

        # reshape as desired
        nhwk_out = self.transform_matrix(
            self.desired_inp_format, self.out_format, nhwk_out)
        nhwk_out.tofile(f'./data/{self.op_name}/result.txt', sep='\n')
        print(f'result of {self.op_name} is:', nhwk_out)
        end_time = timeit.default_timer()
        print('\n********* ILA simulator performance ***********')
        print('ILA simulator execution time is {:04f}s'.format(
            end_time - start_time))

    def run(self):
        subprocess.run(['mkdir', '-p', 'test', 'data'])
        self.collect_data_in()
        self.produce_asm_all()
        self.produce_prog_frag()
        self.invoke_ila_simulator_and_collect_results()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add Parameters')
    parser.add_argument('--op_name', type=str)
    parser.add_argument('--inp_shape', nargs='+', type=int)
    parser.add_argument('--out_shape', nargs='+', type=int)

    # if pool size is an array in tvm and two params aren't equal throw err
    parser.add_argument('--pool_size', nargs='+', type=int)
    parser.add_argument('--strides', nargs='+', type=int)
    parser.add_argument('--dilation', nargs='+', type=int)
    parser.add_argument('--padding', nargs='+', type=int)
    parser.add_argument('--inp_data_format', type=str)
    parser.add_argument('--out_format', nargs='+', type=str, required=False)
    args = parser.parse_args()
    # Calculation conducted assuming:
    # ceil_mode is default value of False
    # count_include_pad is True (default value False)
    # Error thrown in c++ if this is not the case

    driver = pdp_driver(op_name=args.op_name,
                        inp_shape=args.inp_shape,
                        out_shape=args.out_shape,
                        pool_size=args.pool_size,
                        strides=args.strides,
                        dilation=args.dilation,
                        padding=args.padding,
                        inp_data_format=args.inp_data_format,
                        out_format=args.out_format
                        )
    driver.run()
