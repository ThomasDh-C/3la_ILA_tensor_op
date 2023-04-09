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
from conv_converter import ConvConverter


class conv_driver:
    def __init__(self, inp_shape, out_shape, kernels_shape, inp_weight_format,
                 kernels_weight_format, dilation, padding, strides):
        # 4d array - 1 at start because only batch number 1 supported
        self.orig_inp_shape = [1] + inp_shape
        self.orig_out_shape = out_shape
        self.orig_kernels_shape = kernels_shape  # 4d array
        self.inp_weight_format = inp_weight_format  # string
        self.kernels_weight_format = kernels_weight_format  # string
        # always convert to same internal format N = 1, then height, then width, then channels
        self.desired_inp_format = 'NHWC'
        # always convert to same internal format - same as for inp
        self.desired_kern_format = 'OHWI'
        self.dilation = dilation  # 2d array
        self.padding = padding  # 2d array
        self.strides = strides  # 2d array

        # Feature kernel
        self.inp_matrix = None
        # Collection of weight kernels
        self.kernels_matrix = None
        self.ila_asm = []
        self.op_name = "conv2d"
        self.inp1_mem_length = 0  # measured in ints

    def produce_write_asm(self):
        """
        Produce asm for writing data to the RDMA memory of the NVDLA
        """
        self.produce_write_asm_data_cube(
            self.inp_matrix.shape, self.inp_matrix, 0)
        self.produce_write_asm_data_cube(
            self.kernels_matrix.shape, self.kernels_matrix, 32000 * 2)

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
        # --- SETUP ---
        self.ila_asm.append({
            'name': 'CMAC_A_S_POINTER',
            'NVDLA_CMAC_A_PRODUCER': 0
        })
        # Conv mode does joining cores. In simulator CMAC_A_D_MISC_CFG currently unused
        # NVDLA_CMAC_A_PROC_PRECISION: 0 = 8 bit, 1 = 16 bit
        self.ila_asm.append({
            'name': 'CMAC_A_D_MISC_CFG',
            'NVDLA_CMAC_A_CONV_MODE': 0,
            'NVDLA_CMAC_A_PROC_PRECISION': 1
        })

        self.ila_asm.append({
            'name': 'CMAC_A_D_OP_ENABLE',
            'NVDLA_CMAC_A_D_OP_ENABLE': 1
        })

        # --- COMPUTATION ---
        # iterate through all kernels
        n, h, w, c = self.kernels_matrix.shape
        numbers_per_block = 64
        # Group operation below here
        for n_n_large in range(0, math.ceil(n / 16)):
            # Channel operation is below here --> after each chnnel op CACC must be unloaded to make room for next channel op
            for n_c_large in range(0, math.ceil(c / numbers_per_block)):
                # Block operation is below here
                for n_h in range(h):
                    for n_w in range(w):
                        # same kernel weights for all kernels in this stripe - 1 feature atom and 16 kernel atoms supported per os
                        # 64 channels
                        cache_weight_op = {
                            'name': 'CMAC_CACHE_WEIGHTS'
                        }
                        for n_n in range(16):
                            for n_c_small in range(numbers_per_block):
                                # add on int or pad with 0s
                                n_idx = n_n_large*16 + n_n
                                ch_idx = n_c_large*numbers_per_block + n_c_small
                                cache_weight_op[f'cmac_csc2cmac_wt_{n_n}_{n_c_small}'] = 0
                                if n_idx < n and ch_idx < c:
                                    cache_weight_op[f'cmac_csc2cmac_wt_{n_n}_{n_c_small}'] = int(self.kernels_matrix[
                                        n_idx][n_h][n_w][ch_idx])

                        self.produce_stripe_asm(
                            cache_weight_op, n_n_large, n_c_large, n_h, n_w)

    def produce_stripe_asm(self, cache_weight_op, n_n_large, n_c_large, n_h, n_w):
        """Iterate through all inputs this kernel atom can touch"""
        n, h, w, c = self.inp_matrix.shape
        kern_n, kern_h, kern_w, kern_c = self.kernels_matrix.shape
        stride_w, strid_h = self.strides
        dilation_w, dilation_h = self.dilation
        cmac_calls = 0  # min
        numbers_per_block = 64
        # if doing stride would throw in here in range the step size and do more math range(n_h, h-kern_h+2)
        _, _, out_h, out_w = self.orig_out_shape
        for n_h_inp in range(n_h*dilation_h, out_h+n_h*dilation_h):
            for n_w_inp in range(n_w*dilation_w, out_w+n_w*dilation_w):
                # same kernel weights for all kernels in this stripe - 1 feature atom and 16 kernel atoms supported per os
                # 64 channelscmac_conv_direct
                cmac_op = {'name': 'CMAC_CONV_DIRECT'}
                if cmac_calls == 0 or cmac_calls >= 32:
                    cmac_op = cache_weight_op.copy()
                    cmac_calls = 0
                cmac_calls += 1
                for n_c_small_inp in range(numbers_per_block):
                    ch_idx = n_c_large*numbers_per_block + n_c_small_inp
                    cmac_op[f'cmac_csc2cmac_ft_{n_c_small_inp}'] = 0
                    if ch_idx < c:
                        cmac_op[f'cmac_csc2cmac_ft_{n_c_small_inp}'] = int(
                            self.inp_matrix[0][n_h_inp][n_w_inp][ch_idx])
                # n_n_large = which set of 16 kernels working on
                # n_h_inp-n_h, n_w_inp-n_w are h, w position respectively in out matrix
                # don't include n_c_large as in the end will combine all n_c_large together
                # so don't need to distinguish them
                cmac_op['out_pos'] = f'{n_n_large}_{n_h_inp-n_h*dilation_h}_{n_w_inp-n_w*dilation_w}'
                self.ila_asm.append(cmac_op)

        # pad stripe with empty unused computations if necessary for CACC
        while cmac_calls < 16:
            cmac_op = {'name': 'CMAC_CONV_DIRECT'}
            if cmac_calls == 0:
                cmac_op = cache_weight_op
            cmac_calls += 1
            for n_c_small_inp in range(numbers_per_block):
                cmac_op[f'cmac_csc2cmac_ft_{n_c_small_inp}'] = 0
            cmac_op['out_pos'] = 'padding'
            self.ila_asm.append(cmac_op)

    def produce_read_asm(self):
        """
        produce asm for reading data from the RDMA memory of the NVDLA
        """
        # assumes other cores have run and that the result has been stored back in memory from those - incorrect currently
        dbbif_width = 512  # dbbif configurable to width of 32, 64, 128, 256 or 512-bits
        atoms_per_line = int(dbbif_width / 16)
        for i in range(int(np.ceil(np.prod(self.orig_out_shape) / atoms_per_line))):
            self.ila_asm.append({
                'name': 'VirMemRd',
                'addr': hex(i * atoms_per_line * 2)
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

        new_order = []
        for c in out_format:
            if c in inp_format:
                new_order.append(inp_format.index(c))
            else:
                raise Exception(
                    f"Char {c} from {out_format} not found in {inp_format}. Cannot convert matrix to new form")
        return np.transpose(inp_matrix, new_order)

    def collect_data_in(self):
        """
        collect relay data from files
        """
        print('\n--------------------------------------------------------------')
        print('\tcollecting input data')
        print('--------------------------------------------------------------\n')
        with open(f'./data/{self.op_name}/inp.json', 'r') as fin:
            self.inp_matrix = np.array(json.load(fin)).astype(
                'int16').reshape(self.orig_inp_shape)
            self.inp_matrix = self.transform_matrix(
                self.inp_weight_format, self.desired_inp_format, self.inp_matrix)
            # pad as necessary
            n, h, w, c = self.inp_matrix.shape
            pad_h, pad_w = self.padding
            temp_matrix = np.zeros(
                shape=(n, h+2*pad_h, w+2*pad_w, c), dtype='int16')
            temp_matrix[0, pad_h:h+pad_h,
                        pad_w:w+pad_w, :] = self.inp_matrix
            self.inp_matrix = temp_matrix
        print(self.inp_matrix)

        with open(f'./data/{self.op_name}/kernels.json', 'r') as fin:
            self.kernels_matrix = np.array(json.load(fin)).astype(
                'int16').reshape(self.orig_kernels_shape)
            self.kernels_matrix = self.transform_matrix(
                self.kernels_weight_format, self.desired_kern_format, self.kernels_matrix)
        print(self.kernels_matrix)

    def produce_prog_frag(self):
        print('\n--------------------------------------------------------------')
        print('\tgenerate prog_frag.json for ILA simulator')
        print('--------------------------------------------------------------\n')
        self.ila_cvtr = ConvConverter(
            f'./test/{self.op_name}/ila_asm.json', self.op_name)
        self.ila_cvtr.dump_ila_prog_frag(
            f'./test/{self.op_name}/ila_prog_frag_input.json')

    def invoke_ila_simulator_and_collect_results(self):
        print('\n--------------------------------------------------------------')
        print('\tinvoking NVDLA ILA simulator and collecting result')
        print('--------------------------------------------------------------\n')
        start_time = timeit.default_timer()
        cmd = [
            "cmac_sim_driver_v2_fast",
            f'./test/{self.op_name}/ila_prog_frag_input.json',
            f'./test/{self.op_name}/ila_prog_frag_out.json'
        ]
        print('Running command', " ".join(cmd))
        subprocess.run(cmd)

        sim_output = []
        with open(f'./test/{self.op_name}/ila_prog_frag_out.json', 'r') as fin:
            # sim_output = fin.readlines()[1::2]  # 1::2 skips datatype lines
            # sim_output = [l[:-1] if l[-1] == '\n' else l for l in sim_output]
            # sim_output = [l.strip().split(' ')[1:]
            #               for l in sim_output]  # remove line number
            # sim_output = [[int(num_str) for num_str in l]
            #               for l in sim_output]  # turn to ints
            temp = []
            for l_idx, line in enumerate(fin.readlines()):
                if line[:9] == 'instr No.':
                    if l_idx != 0:
                        sim_output.append(temp)
                    temp = []
                elif line[:3] == 'mac':
                    split_line = line.replace('\n', '').split(' ')
                    temp.append(int(split_line[1]))
            if len(temp) > 0:
                sim_output.append(temp)

        # iterate through output ... nb don't know output shape so have to assemble in known format then convert to correct
        n, k, h, w = self.orig_out_shape
        if self.inp_weight_format != 'NCHW':
            # don't know output format if this isn't NCHW
            # TODO: fix this in future
            raise Exception(
                f'Data collection for weight input format {self.inp_weight_format} not implemented')
        if n > 1:
            # TODO: fix this in future
            raise Exception(
                f'Batch sizes over 1 (currently {n} not implemented')

        nkhw_out = np.zeros(self.orig_out_shape, 'int16')
        for n_k_large in range(0, math.ceil(k/16)):
            for n_h in range(h):
                for n_w in range(w):
                    # get array of output indexes to sum
                    ids_to_sum = self.ila_cvtr.sim_to_out_cube[f'{n_k_large}_{n_h}_{n_w}']
                    for sim_output_idx in ids_to_sum:
                        # retrieve vals for that line using json_idx
                        line_vals = sim_output[sim_output_idx]
                        for n_k in range(16):
                            # add on int or pad with 0s
                            k_idx = n_k_large*16 + n_k
                            if k_idx < k:
                                nkhw_out[0][k_idx][n_h][n_w] += line_vals[n_k]

        nkhw_out.tofile(f'./data/{self.op_name}/result.txt', sep='\n')

        end_time = timeit.default_timer()
        print('\n********* ILA simulator performance ***********')
        print('ILA simulator execution time is {:04f}s'.format(
            end_time - start_time))

    def clean_root_dir(self, deep_clean):
        if deep_clean:
            for file_name in ['instr_log.txt', 'instr_update_log.txt']:
                os.remove(file_name)

    def run(self):
        subprocess.run(['mkdir', '-p', 'test', 'data'])

        self.collect_data_in()
        self.produce_asm_all()
        self.produce_prog_frag()
        self.invoke_ila_simulator_and_collect_results()
        self.clean_root_dir(deep_clean=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add Parameters')
    parser.add_argument('--inp_shape', nargs='+', type=int)
    parser.add_argument('--out_shape', nargs='+', type=int)
    parser.add_argument('--kernels_shape', nargs='+', type=int)
    parser.add_argument('--input_weight_format', type=str)
    parser.add_argument('--kernels_weight_format', type=str)
    parser.add_argument('--dilation', nargs='+', type=int)
    parser.add_argument('--padding', nargs='+', type=int)
    parser.add_argument('--strides', nargs='+', type=int)
    args = parser.parse_args()

    driver = conv_driver(inp_shape=args.inp_shape,
                         out_shape=args.out_shape,
                         kernels_shape=args.kernels_shape,
                         inp_weight_format=args.input_weight_format,
                         kernels_weight_format=args.kernels_weight_format,
                         dilation=args.dilation,
                         padding=args.padding,
                         strides=args.strides
                         )
    driver.run()
