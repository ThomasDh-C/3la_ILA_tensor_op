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


class rubik_driver:

    def __init__(self, op_name, inp_shape):
        # op_name = merge or split
        # if merge doing NCHW to NHWC
        # if split doing NHWC to NCHW
        self.op_name = op_name
        # 3d array - 1 at start because only batch number 1 supported
        self.orig_inp_shape = [1] + inp_shape

        self.inp_format = "NCHW" if op_name == "merge" else "NHWC"
        self.desired_inp_format = 'NHWC'
        self.inp_matrix = None
        self.ila_asm = []
        self.inp1_mem_length = 0                        # measured in ints

    def produce_write_asm(self):
        """
        Produce asm for writing data to DRAM external to the NVDLA
        """
        self.produce_write_asm_data_cube(
            self.inp_matrix.shape, self.inp_matrix, 0)

    def produce_write_asm_data_cube(self, cube_shape, cube_list, addr_offset):
        """
        Produce asm for writing given data cube to DRAM external to the NVDLA
        """
        # dbbif configurable to width of 32, 64, 128, 256 or 512-bits
        # ie nb of ints is               2,   4,   8,  16 or  32
        dbbif_width = 512
        ints_per_line = int(dbbif_width / 16)
        n, h, w, c = cube_shape

        all_data_str = []
        full_data_str = ''
        numbers_per_block = 16
        for n_n in range(n):
            for n_c_large in range(0, math.ceil(c / numbers_per_block)):
                for n_h in range(h):
                    for n_w in range(w):
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
        # --- PRE-SETUP ---
        self.ila_asm.append({
            'name': 'RUBIK_S_POINTER',
            'NVDLA_RUBIK_PRODUCER': 0,
            'NVDLA_RUBIK_CONSUMER': 0
        })

        # --- SETUP RUBIK ---
        # (0=contract (unused), 1=split (nhwc to nchw), 2=merge (nchw to nhwc)
        self.ila_asm.append({
            'name': 'RUBIK_D_MISC_CFG',
            'NVDLA_RUBIK_RUBIK_MODE': 1 if self.op_name == 'contract' else 2,
            'NVDLA_RUBIK_IN_PRECISION': 1
        })
        # ram type - use external DRAM
        self.ila_asm.append({
            'name': 'RUBIK_D_DAIN_RAM_TYPE',
            'NVDLA_RUBIK_DATAIN_RAM_TYPE': 1
        })
        n, h, w, c = self.inp_matrix.shape
        # size of input
        self.ila_asm.append({
            'name': 'RUBIK_D_DATAIN_SIZE_0',
            'NVDLA_RUBIK_DATAIN_WIDTH': w,
            'NVDLA_RUBIK_DATAIN_HEIGHT': h
        })
        self.ila_asm.append({
            'name': 'RUBIK_D_DATAIN_SIZE_1',
            'NVDLA_RUBIK_DATAIN_CHANNEL': c,
        })
        # address of input
        self.ila_asm.append({
            'name': 'RUBIK_D_DAIN_ADDR_HIGH',
            'NVDLA_RUBIK_DAIN_ADDR_HIGH': 0
        })
        self.ila_asm.append({
            'name': 'RUBIK_D_DAIN_ADDR_LOW',
            'NVDLA_RUBIK_DAIN_ADDR_LOW': 0
        })
        # line stride of input
        self.ila_asm.append({
            'name': 'RUBIK_D_DAIN_LINE_STRIDE',
            'NVDLA_RUBIK_DAIN_LINE_STRIDE': int(w*16*2/2**5)
        })
        # surface stride of input
        self.ila_asm.append({
            'name': 'RUBIK_D_DAIN_SURF_STRIDE',
            'NVDLA_RUBIK_DAIN_SURF_STRIDE': int(w*h*16*2/2**5)
        })
        # planar stride
        self.ila_asm.append({
            'name': 'RUBIK_D_DAIN_PLANAR_STRIDE',
            'NVDLA_RUBIK_DAIN_PLANAR_STRIDE': int(w*h*16*2/2**5)
        })
        # output ram type is external DRAM
        self.ila_asm.append({
            'name': 'RUBIK_D_DAOUT_RAM_TYPE',
            'NVDLA_RUBIK_DATAOUT_RAM_TYPE': 1
        })
        # output size
        self.ila_asm.append({
            'name': 'RUBIK_D_DAOUT_SIZE_1',
            'NVDLA_RUBIK_DATAOUT_CHANNEL': c
        })
        # output address
        self.ila_asm.append({
            'name': 'RUBIK_D_DAOUT_ADDR_HIGH',
            'NVDLA_RUBIK_DAOUT_ADDR_HIGH': int(512000/2**5)
        })
        self.ila_asm.append({
            'name': 'RUBIK_D_DAOUT_ADDR_LOW',
            'NVDLA_RUBIK_DAOUT_ADDR_LOW': 0
        })
        # output line stride
        self.ila_asm.append({
            'name': 'RUBIK_D_DAOUT_LINE_STRIDE',
            'NVDLA_RUBIK_DAOUT_LINE_STRIDE': int(w*16*2/2**5)
        })
        # contract stride (unused)
        self.ila_asm.append({
            'name': 'RUBIK_D_CONTRACT_STRIDE_0',
            'NVDLA_RUBIK_CONTRACT_STRIDE_0': 0
        })
        self.ila_asm.append({
            'name': 'RUBIK_D_CONTRACT_STRIDE_1',
            'NVDLA_RUBIK_CONTRACT_STRIDE_1': 0
        })
        # surface stride
        self.ila_asm.append({
            'name': 'RUBIK_D_DAOUT_SURF_STRIDE',
            'NVDLA_RUBIK_DAOUT_SURF_STRIDE': int(w*h*16*2/2**5)
        })
        # planar stride
        self.ila_asm.append({
            'name': 'RUBIK_D_DAOUT_PLANAR_STRIDE',
            'NVDLA_RUBIK_DAOUT_PLANAR_STRIDE': int(w*h*16*2/2**5)
        })
        # deconv stide (unused as only do contract and merge)
        self.ila_asm.append({
            'name': 'RUBIK_D_DECONV_STRIDE',
            'NVDLA_RUBIK_DECONV_X_STRIDE': 0,
            'NVDLA_RUBIK_DECONV_Y_STRIDE': 0
        })
        # disable performance counters
        self.ila_asm.append({
            'name': 'RUBIK_D_PERF_ENABLE',
            'NVDLA_RUBIK_PERF_EN': 0
        })

        # --- Enable RUBIK ---
        self.ila_asm.append({
            'name': 'RUBIK_D_OP_ENABLE',
            'NVDLA_RUBIK_D_OP_ENABLE': 1
        })

        # --- COMPUTATION ---
        # can process 4 int16s per call
        raise NotImplementedError('RUBIK simulator not implemented yet')

    def produce_read_asm(self):
        """
        produce asm for reading data from the external DRAM into this program
        """
        # assumes other sub-units have run and that the result has been stored back in memory
        dbbif_width = 512  # dbbif configurable to width of 32, 64, 128, 256 or 512-bits
        ints_per_line = int(dbbif_width / 16)
        n, h, w, c = self.inp_matrix.shape
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
                self.inp_format, self.desired_inp_format, self.inp_matrix)
        print('Max of input:', np.max(self.inp_matrix))

    def produce_prog_frag(self):
        print('\n--------------------------------------------------------------')
        print('\tgenerate prog_frag.json for ILA simulator')
        print('--------------------------------------------------------------\n')
        raise NotImplementedError(
            'RUBIK simulator and thus converter not implemented yet')
        # self.ila_cvtr = RUBIKConverter(
        #     f'./test/{self.op_name}/ila_asm.json', self.op_name)
        # self.ila_cvtr.dump_ila_prog_frag(
        #     f'./test/{self.op_name}/ila_prog_frag_input.json')

    def invoke_ila_simulator_and_collect_results(self):
        print('\n--------------------------------------------------------------')
        print('\tinvoking NVDLA ILA simulator and collecting result')
        print('--------------------------------------------------------------\n')
        raise NotImplementedError('RUBIK simulator not implemented yet')
        start_time = timeit.default_timer()
        cmd = [
            "rubik_sim_driver",
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
            self.desired_inp_format, self.inp_format, nhwk_out)
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
    args = parser.parse_args()

    driver = rubik_driver(op_name=args.op_name,
                          inp_shape=args.inp_shape,
                          )
    driver.run()
