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


class cdp_driver:

    def __init__(self, op_name, inp_shape, out_shape, size, axis, bias, alpha, beta):
        self.op_name = op_name
        # 3d array - 1 at start because only batch number 1 supported
        self.orig_inp_shape = [1] + inp_shape
        self.orig_out_shape = out_shape  # passed in as 4d
        self.size = size
        if self.size not in {3, 5, 7, 9}:
            raise Exception("NVDLA only supports sizes of 3, 5, 7 or 9")
        self.axis = axis
        self.bias = bias
        self.alpha = alpha
        self.beta = beta
        # always convert to same internal format N = 1, then height, then width, then channels
        self.inp_format = "NCHW"  # based on pytorch documentation as nothing on tvm site for it
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
        # self.op_name = lrn
        # --- PRE-SETUP ---
        self.ila_asm.append({
            'name': 'CDP_RDMA_S_POINTER',
            'NVDLA_CDP_RDMA_PRODUCER': 0,
            'NVDLA_CDP_RDMA_CONSUMER': 0
        })
        self.ila_asm.append({
            'name': 'CDP_S_POINTER',
            'NVDLA_CDP_PRODUCER': 0,
            'NVDLA_CDP_CONSUMER': 0
        })

        # --- SETUP CDP RDMA ---
        n, h, w, c = self.inp_matrix.shape
        # 0xe00c: input_width
        self.ila_asm.append({
            'name': 'CDP_RDMA_D_DATA_CUBE_WIDTH',
            'NVDLA_CDP_RDMA_WIDTH': w
        })
        # 0xe010: input_height
        self.ila_asm.append({
            'name': 'CDP_RDMA_D_DATA_CUBE_HEIGHT',
            'NVDLA_CDP_RDMA_HEIGHT': h
        })
        # 0xe014: input_channel
        self.ila_asm.append({
            'name': 'CDP_RDMA_D_DATA_CUBE_CHANNEL',
            'NVDLA_CDP_RDMA_CHANNEL': c
        })
        # 0xe018: input address
        self.ila_asm.append({
            'name': 'CDP_RDMA_D_SRC_BASE_ADDR_LOW',
            'NVDLA_CDP_RDMA_SRC_BASE_ADDR_LOW': 0
        })
        # 0xe01c: input address
        self.ila_asm.append({
            'name': 'CDP_RDMA_D_SRC_BASE_ADDR_HIGH',
            'NVDLA_CDP_RDMA_SRC_BASE_ADDR_HIGH': 0
        })
        # 0xe020: input stride
        self.ila_asm.append({
            'name': 'CDP_RDMA_D_SRC_LINE_STRIDE',
            'NVDLA_CDP_RDMA_SRC_LINE_STRIDE': int(w*16*2/2**5)
        })
        # 0xe024: input surface stride
        self.ila_asm.append({
            'name': 'CDP_RDMA_D_SRC_SURFACE_STRIDE',
            'NVDLA_CDP_RDMA_SRC_SURFACE_STRIDE': int(w*h*16*2/2**5)
        })
        # 0xe028: use external DRAM
        self.ila_asm.append({
            'name': 'CDP_RDMA_D_SRC_DMA_CFG',
            'NVDLA_CDP_RDMA_SRC_RAM_TYPE': 1
        })
        # 0xe02c: no compression used
        self.ila_asm.append({
            'name': 'CDP_RDMA_D_SRC_COMPRESSION_EN',
            'NVDLA_CDP_RDMA_SRC_COMPRESSION_EN': 0
        })
        # 0xe030: no split used
        self.ila_asm.append({
            'name': 'CDP_RDMA_D_OPERATION_MODE',
            'NVDLA_CDP_RDMA_SPLIT_NUM': 0
        })
        # 0xe034: input data format is int16
        self.ila_asm.append({
            'name': 'CDP_RDMA_D_DATA_FORMAT',
            'NVDLA_CDP_RDMA_INPUT_DATA': 1
        })
        # 0xe038: don't use performance registers
        self.ila_asm.append({
            'name': 'CDP_RDMA_D_PERF_ENABLE',
            'NVDLA_CDP_RDMA_DMA_EN': 0
        })

        # --- SETUP CDP ---
        # 0xf008: config how (write = 1) and which lut to address --> first 0 (LE), then 1 (LO)
        # setup lut LE/ Table X (exponential) - raw/ full range
        lut_table_idx = 0  # LE table
        self.ila_asm.append({
            'name': 'CDP_S_LUT_ACCESS_CFG',
            'NVDLA_CDP_LUT_TABLE_ID': lut_table_idx,
            'NVDLA_CDP_LUT_ACCESS_TYPE': 1
        })
        # 2^0 to 2^37
        for lut_table_entry_idx in range(38):
            lut_entry_data = (
                self.bias + (self.alpha * (2**lut_table_entry_idx) / self.size))**(-self.beta)
            self.ila_asm.append({
                'name': 'CDP_S_LUT_ACCESS_DATA',
                'NVDLA_CDP_LUT_DATA': int(lut_entry_data)
            })
        lut_table_idx = 1  # LO table
        self.ila_asm.append({
            'name': 'CDP_S_LUT_ACCESS_CFG',
            'NVDLA_CDP_LUT_TABLE_ID': lut_table_idx,
            'NVDLA_CDP_LUT_ACCESS_TYPE': 1
        })
        for lut_table_entry_idx in range(257):
            x = 2**20/(257-1) * lut_table_entry_idx
            lut_entry_data = (
                self.bias + (self.alpha * x / self.size))**(-self.beta)
            self.ila_asm.append({
                'name': 'CDP_S_LUT_ACCESS_DATA',
                'NVDLA_CDP_LUT_DATA': int(lut_entry_data)
            })
        # 0xf010: config lut overflow behaviour
        # a) NVDLA_CDP_LUT_LE_FUNCTION --> 0: exponential, 1: linear (yes counterintuitive)
        # b) NVDLA_CDP_LUT_UFLOW_PRIORITY --> unused as won't have an input x (sum of squares) that is under min(table Y) = 0
        # c) NVDLA_CDP_LUT_OFLOW_PRIORITY --> unused as raw handles values up to and over max(int16)
        # d) NVDLA_CDP_LUT_HYBRID_PRIORITY --> unused as never have case that one table overflows and other underflows
        self.ila_asm.append({
            'name': 'CDP_S_LUT_CFG',
            'NVDLA_CDP_LUT_LE_FUNCTION': 0,
            'NVDLA_CDP_LUT_UFLOW_PRIORITY': 0,
            'NVDLA_CDP_LUT_OFLOW_PRIORITY': 0,
            'NVDLA_CDP_LUT_HYBRID_PRIORITY': 0
        })
        # 0xf014: config LE and LO LUT index offset and selection
        # a) (ie X[0] =f(2^0) and scale in = 0)
        # b) NVDLA_CDP_LUT_LE_INDEX_OFFSET --> 0
        # c) NVDLA_CDP_LUT_LE_INDEX_SELECT --> -M where M is -12
        #       SF_lut = (257-1)/(2^20 * 1) = 2^-12 = 2^M
        self.ila_asm.append({
            'name': 'CDP_S_LUT_INFO',
            'NVDLA_CDP_LUT_LE_INDEX_OFFSET': 0,
            'NVDLA_CDP_LUT_LE_INDEX_SELECT': 0,
            'NVDLA_CDP_LUT_LO_INDEX_SELECT': 12,
        })
        # LE starts at 0
        # 0xf018: config LE start
        self.ila_asm.append({
            'name': 'CDP_S_LUT_LE_START_LOW',
            'NVDLA_CDP_LUT_LE_START_LOW': 0
        })
        # 0xf01c: config LE start
        self.ila_asm.append({
            'name': 'CDP_S_LUT_LE_START_HIGH',
            'NVDLA_CDP_LUT_LE_START_HIGH': 0
        })
        # LE ends at 2^(32+5)
        # 0xf020: config LE end
        self.ila_asm.append({
            'name': 'CDP_S_LUT_LE_END_LOW',
            'NVDLA_CDP_LUT_LE_END_LOW': 0
        })
        # 0xf024: config LE end (6 bits)
        self.ila_asm.append({
            'name': 'CDP_S_LUT_LE_END_HIGH',
            'NVDLA_CDP_LUT_LE_END_HIGH': int(2**5)
        })
        # LO starts at 0
        # 0xf028: config LO start - low_bits(O_lut) = 0
        self.ila_asm.append({
            'name': 'CDP_S_LUT_LO_START_LOW',
            'NVDLA_CDP_LUT_LO_START_LOW': 0
        })
        # 0xf02c: config LO start - high_bits(O_lut) = 0
        self.ila_asm.append({
            'name': 'CDP_S_LUT_LO_START_HIGH',
            'NVDLA_CDP_LUT_LO_START_HIGH': 0
        })
        # LO ends at 2^20 = 1048576 ~ 1,000,000
        # 0xf030: config LO end - low_bits(2^20) = 2^20
        self.ila_asm.append({
            'name': 'CDP_S_LUT_LO_END_LOW',
            'NVDLA_CDP_LUT_LO_END_LOW': 2**20
        })
        # 0xf034: config LO end - high_bits(2^20) = 0
        self.ila_asm.append({
            'name': 'CDP_S_LUT_LO_END_HIGH',
            'NVDLA_CDP_LUT_LO_END_HIGH': 0
        })
        # 0xf038: config LE scale - unused
        # a) unused as input never less than 0 and range of linear table Y covers 0-1
        # b) unused as Table X covers full int16 range
        self.ila_asm.append({
            'name': 'CDP_S_LUT_LE_SLOPE_SCALE',
            'NVDLA_CDP_LUT_LE_SLOPE_UFLOW_SCALE': 0,
            'NVDLA_CDP_LUT_LE_SLOPE_OFLOW_SCALE': 0
        })
        # 0xf03c: config LE shift - unused for same reasons as above
        self.ila_asm.append({
            'name': 'CDP_S_LUT_LE_SLOPE_SHIFT',
            'NVDLA_CDP_LUT_LE_SLOPE_UFLOW_SHIFT': 0,
            'NVDLA_CDP_LUT_LE_SLOPE_OFLOW_SHIFT': 0
        })
        # 0xf040: config LO scale - unused for same reasons as above
        self.ila_asm.append({
            'name': 'CDP_S_LUT_LO_SLOPE_SCALE',
            'NVDLA_CDP_LUT_LO_SLOPE_UFLOW_SCALE': 0,
            'NVDLA_CDP_LUT_LO_SLOPE_OFLOW_SCALE': 0
        })
        # 0xf044: config LO shift - unused for same reasons as above
        self.ila_asm.append({
            'name': 'CDP_S_LUT_LO_SLOPE_SHIFT',
            'NVDLA_CDP_LUT_LO_SLOPE_UFLOW_SHIFT': 0,
            'NVDLA_CDP_LUT_LO_SLOPE_OFLOW_SHIFT': 0
        })
        # 0xf04c: don't bypass mul and sum(square(x)) parts of CDP
        self.ila_asm.append({
            'name': 'CDP_D_FUNC_BYPASS',
            'NVDLA_CDP_SQSUM_BYPASS': 0,
            'NVDLA_CDP_MUL_BYPASS': 0,
        })
        # 0xf050: destination address in DRAM
        self.ila_asm.append({
            'name': 'CDP_D_DST_BASE_ADDR_LOW',
            'NVDLA_CDP_DST_BASE_ADDR_LOW': int(512000/(2**5))
        })
        # 0xf054: destination address in DRAM
        self.ila_asm.append({
            'name': 'CDP_D_DST_BASE_ADDR_HIGH',
            'NVDLA_CDP_DST_BASE_ADDR_HIGH': 0
        })
        out_n, out_h, out_w, out_c = self.orig_out_shape
        # 0xf058: destination line stride
        self.ila_asm.append({
            'name': 'CDP_D_DST_LINE_STRIDE',
            'NVDLA_CDP_DST_LINE_STRIDE': int(out_w*16*2/(2**5))
        })
        # 0xf05c: destination surface stride
        self.ila_asm.append({
            'name': 'CDP_D_DST_SURFACE_STRIDE',
            'NVDLA_CDP_DST_SURFACE_STRIDE': int(out_h*out_w*16*2/(2**5))
        })
        # 0xf060: use DRAM as output
        self.ila_asm.append({
            'name': 'CDP_D_DST_DMA_CFG',
            'NVDLA_CDP_DST_RAM_TYPE': 1
        })
        # 0xf064: don't use compression for output
        self.ila_asm.append({
            'name': 'CDP_D_DST_COMPRESSION_EN',
            'NVDLA_CDP_COMPRESSION_EN': 0
        })
        # 0xf068: data format is int16
        self.ila_asm.append({
            'name': 'CDP_D_DATA_FORMAT',
            'NVDLA_CDP_INPUT_DATA_TYPE': 1
        })
        # 0xf06c: unused as won't have na's but if did don't flush to 0
        self.ila_asm.append({
            'name': 'CDP_D_NAN_FLUSH_TO_ZERO',
            'NVDLA_CDP_NAN_TO_ZERO': 0
        })
        # 0xf070: set size variable
        self.ila_asm.append({
            'name': 'CDP_D_LRN_CFG',
            'NVDLA_CDP_NORMALZ_LEN': self.size
        })
        # 0xf074: set data in offset
        self.ila_asm.append({
            'name': 'CDP_D_DATIN_OFFSET',
            'NVDLA_CDP_DATIN_OFFSET': 0
        })
        # 0xf078: set data in scale
        self.ila_asm.append({
            'name': 'CDP_D_DATIN_SCALE',
            'NVDLA_CDP_DATIN_SCALE': 1
        })
        # 0xf07c: set data in shifter
        self.ila_asm.append({
            'name': 'CDP_D_DATIN_SHIFTER',
            'NVDLA_CDP_DATIN_SHIFTER': 0
        })
        # 0xf080: set data out offset
        self.ila_asm.append({
            'name': 'CDP_D_DATOUT_OFFSET',
            'NVDLA_CDP_DATOUT_OFFSET': 0
        })
        # 0xf084: set data out scale
        self.ila_asm.append({
            'name': 'CDP_D_DATOUT_SCALE',
            'NVDLA_CDP_DATOUT_SCALE': 1
        })
        # 0xf088: set data out shifter
        self.ila_asm.append({
            'name': 'CDP_D_DATOUT_SHIFTER',
            'NVDLA_CDP_DATOUT_SHIFTER': 0
        })
        # 0xf09c: don't use performance metrics (assume perfectly programmed LUT)
        self.ila_asm.append({
            'name': 'CDP_D_PERF_ENABLE',
            'NVDLA_CDP_DMA_EN': 0,
            'NVDLA_CDP_LUT_EN': 0
        })

        # --- Enable CDP and CDP RDMA sub-units ---
        self.ila_asm.append({
            'name': 'CDP_RDMA_D_OP_ENABLE',
            'NVDLA_CDP_RDMA_D_OP_ENABLE': 1
        })
        self.ila_asm.append({
            'name': 'CDP_D_OP_ENABLE',
            'NVDLA_CDP_D_OP_ENABLE': 1
        })

        # --- COMPUTATION ---
        # can process 4 int16s per call
        raise NotImplementedError('CDP simulator not implemented yet')

    def produce_read_asm(self):
        """
        produce asm for reading data from the external DRAM into this program
        """
        # assumes other sub-units have run and that the result has been stored back in memory
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
                self.inp_format, self.desired_inp_format, self.inp_matrix)
        print('Max of input:', np.max(self.inp_matrix))

        # output shape - ensure transform from NCHW to NHWC
        output_shape_reordering = self.transpose_new_order(
            self.inp_format, self.desired_inp_format)
        self.orig_out_shape = [self.orig_out_shape[idx]
                               for idx in output_shape_reordering]

    def produce_prog_frag(self):
        print('\n--------------------------------------------------------------')
        print('\tgenerate prog_frag.json for ILA simulator')
        print('--------------------------------------------------------------\n')
        raise NotImplementedError(
            'CDP simulator and thus converter not implemented yet')
        # self.ila_cvtr = CDPConverter(
        #     f'./test/{self.op_name}/ila_asm.json', self.op_name)
        # self.ila_cvtr.dump_ila_prog_frag(
        #     f'./test/{self.op_name}/ila_prog_frag_input.json')

    def invoke_ila_simulator_and_collect_results(self):
        print('\n--------------------------------------------------------------')
        print('\tinvoking NVDLA ILA simulator and collecting result')
        print('--------------------------------------------------------------\n')
        raise NotImplementedError('CDP simulator not implemented yet')
        start_time = timeit.default_timer()
        cmd = [
            "cdp_sim_driver",
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
    parser.add_argument('--out_shape', nargs='+', type=int)

    # if pool size is an array in tvm and two params aren't equal throw err
    parser.add_argument('--size', type=int)
    parser.add_argument('--axis', type=int)
    parser.add_argument('--bias',  type=float)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--beta', type=float)
    args = parser.parse_args()

    driver = cdp_driver(op_name=args.op_name,
                        inp_shape=args.inp_shape,
                        out_shape=args.out_shape,
                        size=args.size,
                        axis=args.axis,
                        bias=args.bias,
                        alpha=args.alpha,
                        beta=args.beta
                        )
    driver.run()
