"""
python driver for nvdla-ila simulator
"""

import json
import os
import math
import numpy as np
import subprocess
import argparse
import timeit
from sdp_converter import SDPConverter


class sdp_driver:
    def __init__(self, inp_shape, inp2_shape=[], axis=None,
                 op_name="relu"):
        self.inp1_shape = inp_shape
        self.inp2_shape = inp2_shape
        if self.inp2_shape == None:
            self.inp2_shape = []
        self.input1 = []
        self.input2 = []
        self.input3 = []
        self.ila_asm = []
        self.axis = axis
        self.op_name = op_name
        self.inp1_mem_length = 0  # measured in ints

    def produce_write_asm(self):
        """
        Produce asm for writing data to the RDMA memory of the NVDLA
        """
        self.produce_write_asm_data_cube(self.inp1_shape, self.input1, None, 0)

        if self.op_name[:5] != 'layer':
            # TODO: might need to rejig input for bias add
            if self.op_name == 'channel_batch_norm':
                self.produce_write_asm_data_cube(
                    self.inp2_shape, self.input2, self.input3, 128000 * 2)
            else:
                self.produce_write_asm_data_cube(
                    self.inp2_shape, self.input2, None, 128000 * 2)

    def produce_write_asm_data_cube(self, cube_shape, cube_list, cube_list2, addr_offset):
        """
        Produce asm for writing given data cube
        """
        # dbbif configurable to width of 32, 64, 128, 256 or 512-bits
        # ie nb of ints is               2,   4,   8,  16 or  32
        dbbif_width = 512
        ints_per_atom = 16
        # ints_per_line = int(dbbif_width / 16)
        # 4 bits in byte, 2 bytes in int16
        ints_per_line = int(dbbif_width / (4*2))
        h, w, c = cube_shape

        all_data_str = []
        full_data_str = ''
        for n_c_large in range(0, math.ceil(c / ints_per_atom)):
            for n_h in range(h):
                for n_w in range(w):
                    # 16 channels per atom --> 32 bytes
                    for n_c_small in range(ints_per_atom):
                        # keep track of ints written
                        if addr_offset == 0:
                            self.inp1_mem_length += 1

                        # add on int or pad with 0s
                        ch_idx = n_c_large*ints_per_atom + n_c_small
                        if ch_idx < c:
                            num_str = cube_list[n_h][n_w][ch_idx].tobytes(
                            ).hex()
                            full_data_str = num_str + full_data_str
                            # only ever used for 1/sqrt(variance + eps) in case of batch_norm
                            if cube_list2 is not None:
                                num_str = cube_list2[n_h][n_w][ch_idx].tobytes(
                                ).hex()
                                full_data_str = num_str + full_data_str
                        else:
                            full_data_str = '0000' + full_data_str

                        # purge line if full - note each int takes up 4 chars as 2 bytes
                        if len(full_data_str) == ints_per_line * 4:
                            all_data_str.append(full_data_str)
                            full_data_str = ''

        if len(full_data_str) > 0:
            full_data_str = '0'*(ints_per_line * 4 -
                                 len(full_data_str)) + full_data_str
            all_data_str.append(full_data_str)

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
        # --- Pre-setup for SDP RDMA ---
        self.ila_asm.append({
            'name': 'SDP_RDMA_S_POINTER',
            'NVDLA_SDP_RDMA_PRODUCER': 0,
            'NVDLA_SDP_RDMA_CONSUMER': 0
        })
        # --- Set up SDP RDMA ---
        # set input data cube dimensions
        h, w, c = self.input1.shape
        self.ila_asm.append({
            'name': 'SDP_RDMA_D_DATA_CUBE_WIDTH',
            'NVDLA_SDP_RDMA_WIDTH': w
        })
        self.ila_asm.append({
            'name': 'SDP_RDMA_D_DATA_CUBE_HEIGHT',
            'NVDLA_SDP_RDMA_HEIGHT': h
        })
        self.ila_asm.append({
            'name': 'SDP_RDMA_D_DATA_CUBE_CHANNEL',
            'NVDLA_SDP_RDMA_CHANNEL': c
        })
        # address of input data cube
        self.ila_asm.append({
            'name': 'SDP_RDMA_D_SRC_BASE_ADDR_LOW',
            'NVDLA_SDP_RDMA_SRC_BASE_ADDR_LOW': 0
        })
        self.ila_asm.append({
            'name': 'SDP_RDMA_D_SRC_BASE_ADDR_HIGH',
            'NVDLA_SDP_RDMA_SRC_BASE_ADDR_HIGH': 0
        })
        # line stride of input data cube
        self.ila_asm.append({
            'name': 'SDP_RDMA_D_SRC_LINE_STRIDE',
            'NVDLA_SDP_RDMA_SRC_LINE_STRIDE': int(w*16*2/2**5)
        })
        # surface stride of input data cube
        self.ila_asm.append({
            'name': 'SDP_RDMA_D_SRC_SURFACE_STRIDE',
            'NVDLA_SDP_RDMA_SRC_SURFACE_STRIDE': int(w*h*16*2/2**5)
        })
        # brdma cfg (for sub-module of SDP that use for all ops currently)
        self.ila_asm.append({
            'name': 'SDP_RDMA_D_BRDMA_CFG',
            'NVDLA_SDP_RDMA_BRDMA_DISABLE': 0
        })
        # alu_mul brdma
        alu_or_mul = 0
        if 'mul' in self.op_name or 'prelu' in self.op_name:
            alu_or_mul = 1
        if 'channel_batch_norm' == self.op_name:
            alu_or_mul = 2
        self.ila_asm.append({
            'name': 'SDP_RDMA_D_BRDMA_CFG',
            'NVDLA_SDP_RDMA_BRDMA_DISABLE': 1,
            'NVDLA_SDP_RDMA_BRDMA_DATA_USE': alu_or_mul,
            'NVDLA_SDP_RDMA_BRDMA_DATA_SIZE': 1,
            'NVDLA_SDP_RDMA_BRDMA_DATA_MODE': 1 if 'elemwise' in self.op_name else 0,
            'NVDLA_SDP_RDMA_BRDMA_RAM_TYPE': 1
        })
        # bias addresses
        self.ila_asm.append({
            'name': 'SDP_RDMA_D_BS_BASE_ADDR_LOW',
            'NVDLA_SDP_RDMA_BS_BASE_ADDR_LOW': int(256000/2**5)
        })
        self.ila_asm.append({
            'name': 'SDP_RDMA_D_BS_BASE_ADDR_HIGH',
            'NVDLA_SDP_RDMA_BS_BASE_ADDR_HIGH': 0
        })
        # line stride of bias data
        self.ila_asm.append({
            'name': 'SDP_RDMA_D_BS_LINE_STRIDE',
            'NVDLA_SDP_RDMA_BS_LINE_STRIDE': int(w*16*2/2**5)
        })
        self.ila_asm.append({
            'name': 'SDP_RDMA_D_BS_SURFACE_STRIDE',
            'NVDLA_SDP_RDMA_BS_SURFACE_STRIDE': int(w*h*16*2/2**5)
        })
        # only batches of 1 supported for now
        self.ila_asm.append({
            'name': 'SDP_RDMA_D_BS_BATCH_STRIDE',
            'NVDLA_SDP_RDMA_BS_BATCH_STRIDE': int(w*h*16*2/2**5)
        })
        # ... unused below here until otherwise marked (but needed to disable) ...
        # nrdma (input second sub-unit of SDP) cfg (unused)
        self.ila_asm.append({
            'name': 'SDP_RDMA_D_NRDMA_CFG',
            'NVDLA_SDP_RDMA_NRDMA_DISABLE': 1,
            'NVDLA_SDP_RDMA_NRDMA_DATA_USE': 0,
            'NVDLA_SDP_RDMA_NRDMA_DATA_SIZE': 0,
            'NVDLA_SDP_RDMA_NRDMA_DATA_MODE': 0,
            'NVDLA_SDP_RDMA_NRDMA_RAM_TYPE': 0
        })
        # bias addresses (unused)
        self.ila_asm.append({
            'name': 'SDP_RDMA_D_BN_BASE_ADDR_LOW',
            'NVDLA_SDP_RDMA_BN_BASE_ADDR_LOW': 0
        })
        self.ila_asm.append({
            'name': 'SDP_RDMA_D_BN_BASE_ADDR_HIGH',
            'NVDLA_SDP_RDMA_BN_BASE_ADDR_HIGH': 0
        })
        # line stride of bias data (unused)
        self.ila_asm.append({
            'name': 'SDP_RDMA_D_BN_LINE_STRIDE',
            'NVDLA_SDP_RDMA_BN_LINE_STRIDE': 0
        })
        self.ila_asm.append({
            'name': 'SDP_RDMA_D_BN_SURFACE_STRIDE',
            'NVDLA_SDP_RDMA_BN_SURFACE_STRIDE': 0
        })
        # only batches of 1 supported for now (unused)
        self.ila_asm.append({
            'name': 'SDP_RDMA_D_BN_BATCH_STRIDE',
            'NVDLA_SDP_RDMA_BN_BATCH_STRIDE': 0
        })
        # erdma (input third sub-unit of SDP) cfg (unused)
        self.ila_asm.append({
            'name': 'SDP_RDMA_D_ERDMA_CFG',
            'NVDLA_SDP_RDMA_ERDMA_DISABLE': 1,
            'NVDLA_SDP_RDMA_ERDMA_DATA_USE': 0,
            'NVDLA_SDP_RDMA_ERDMA_DATA_SIZE': 0,
            'NVDLA_SDP_RDMA_ERDMA_DATA_MODE': 0,
            'NVDLA_SDP_RDMA_ERDMA_RAM_TYPE': 0
        })
        # bias addresses (unused)
        self.ila_asm.append({
            'name': 'SDP_RDMA_D_EW_BASE_ADDR_LOW',
            'NVDLA_SDP_RDMA_EW_BASE_ADDR_LOW': 0
        })
        self.ila_asm.append({
            'name': 'SDP_RDMA_D_EW_BASE_ADDR_HIGH',
            'NVDLA_SDP_RDMA_EW_BASE_ADDR_HIGH': 0
        })
        # line stride of bias data (unused)
        self.ila_asm.append({
            'name': 'SDP_RDMA_D_EW_LINE_STRIDE',
            'NVDLA_SDP_RDMA_EW_LINE_STRIDE': 0
        })
        self.ila_asm.append({
            'name': 'SDP_RDMA_D_EW_SURFACE_STRIDE',
            'NVDLA_SDP_RDMA_EW_SURFACE_STRIDE': 0
        })
        # only batches of 1 supported for now (unused)
        self.ila_asm.append({
            'name': 'SDP_RDMA_D_EW_BATCH_STRIDE',
            'NVDLA_SDP_RDMA_EW_BATCH_STRIDE': 0
        })
        # ... used again below here ...
        # configure features
        self.ila_asm.append({
            'name': 'SDP_RDMA_D_FEATURE_MODE_CFG',
            'NVDLA_SDP_RDMA_FLYING_MODE': 0,
            'NVDLA_SDP_RDMA_WINOGRAD': 0,
            'NVDLA_SDP_RDMA_IN_PRECISION': 1,
            'NVDLA_SDP_RDMA_PROC_PRECISION': 1,
            'NVDLA_SDP_RDMA_OUT_PRECISION': 1,
            'NVDLA_SDP_RDMA_BATCH_NUMBER': 1
        })
        # use external DRAM for input
        self.ila_asm.append({
            'name': 'SDP_RDMA_D_SRC_DMA_CFG',
            'NVDLA_SDP_RDMA_SRC_RAM_TYPE': 1
        })
        # don't use perf counters and no floats so don't worry about nan
        self.ila_asm.append({
            'name': 'SDP_RDMA_D_PERF_ENABLE',
            'NVDLA_SDP_RDMA_PERF_EN': 0,
            'NVDLA_SDP_RDMA_PERF_NAN_INF_COUNT_EN': 0
        })

        # --- ENABLE SDP RDMA ---
        self.ila_asm.append({
            'name': 'SDP_RDMA_D_OP_ENABLE',
            'NVDLA_SDP_RDMA_D_OP_ENABLE': 1
        })

        # --- Pre-setup for SDP ---
        self.ila_asm.append({
            'name': 'SDP_S_POINTER',
            'NVDLA_SDP_PRODUCER': 0,
            'NVDLA_SDP_CONSUMER': 0
        })

        # --- Set up SDP (SDP and SDP RDMA (TODO: rdma enable) enable occurs in the various produce_ functions)
        # Luts aren't used but have to set them up as such
        # self.produce_lut_asm()

        # dimensions of input
        self.ila_asm.append({
            'name': 'SDP_D_DATA_CUBE_WIDTH',
            'NVDLA_SDP_WIDTH': w
        })
        self.ila_asm.append({
            'name': 'SDP_D_DATA_CUBE_HEIGHT',
            'NVDLA_SDP_HEIGHT': h
        })
        self.ila_asm.append({
            'name': 'SDP_D_DATA_CUBE_CHANNEL',
            'NVDLA_SDP_CHANNEL': c
        })
        # output memory address
        self.ila_asm.append({
            'name': 'SDP_D_DST_BASE_ADDR_LOW',
            'NVDLA_SDP_DST_BASE_ADDR_LOW': int(512000/2**5)
        })
        self.ila_asm.append({
            'name': 'SDP_D_DST_BASE_ADDR_HIGH',
            'NVDLA_SDP_DST_BASE_ADDR_HIGH': 0
        })
        # line stride of output data (output dims same as input 0 dims)
        self.ila_asm.append({
            'name': 'SDP_D_DST_LINE_STRIDE',
            'NVDLA_SDP_DST_LINE_STRIDE': int(w*16*2/2**5)
        })
        # surface stride of output data (output dims same as input 0 dims)
        self.ila_asm.append({
            'name': 'SDP_D_DST_SURFACE_STRIDE',
            'NVDLA_SDP_DST_SURFACE_STRIDE': int(w*h*16*2/2**5)
        })

        # SDP_D_DP_BS_CFG, SDP_D_DP_BN_CFG and SDP_D_DP_EW_CFG all configured in each produce function
        # Same for SDP_D_DP_BS_ALU_CFG and SDP_D_DP_BS_MUL_CFG
        # SDP_D_DP_BS_ALU_SRC_VALUE and SDP_D_DP_BS_MUL_SRC_VALUE only used by batch_norm per layer
        # but this is never used by relay so never called --> can safely set it arbitrarly to 0 as unused
        self.ila_asm.append({
            'name': 'SDP_D_DP_BS_ALU_SRC_VALUE',
            'NVDLA_SDP_BS_ALU_OPERAND': 0
        })
        self.ila_asm.append({
            'name': 'SDP_D_DP_BS_MUL_SRC_VALUE',
            'NVDLA_SDP_BS_MUL_OPERAND': 0
        })
        # following are never used as only use first module of SDP currently
        self.ila_asm.append({
            'name': 'SDP_D_DP_BN_ALU_CFG',
            'NVDLA_SDP_BN_ALU_SRC': 0,
            'NVDLA_SDP_BN_ALU_SHIFT_VALUE': 0
        })
        self.ila_asm.append({
            'name': 'SDP_D_DP_BN_ALU_SRC_VALUE',
            'NVDLA_SDP_BN_ALU_OPERAND': 0
        })
        self.ila_asm.append({
            'name': 'SDP_D_DP_BN_MUL_CFG',
            'NVDLA_SDP_BN_MUL_SRC': 0,
            'NVDLA_SDP_BN_MUL_SHIFT_VALUE': 0
        })
        self.ila_asm.append({
            'name': 'SDP_D_DP_BN_MUL_SRC_VALUE',
            'NVDLA_SDP_BN_MUL_OPERAND': 0
        })
        self.ila_asm.append({
            'name': 'SDP_D_DP_EW_ALU_CFG',
            'NVDLA_SDP_EW_ALU_SRC': 0,
            'NVDLA_SDP_EW_ALU_CVT_BYPASS': 0
        })
        self.ila_asm.append({
            'name': 'SDP_D_DP_EW_ALU_SRC_VALUE',
            'NVDLA_SDP_EW_ALU_OPERAND': 0
        })
        self.ila_asm.append({
            'name': 'SDP_D_DP_EW_ALU_CVT_OFFSET_VALUE',
            'NVDLA_SDP_EW_ALU_CVT_OFFSET': 0
        })
        self.ila_asm.append({
            'name': 'SDP_D_DP_EW_ALU_CVT_SCALE_VALUE',
            'NVDLA_SDP_EW_ALU_CVT_SCALE': 0
        })
        self.ila_asm.append({
            'name': 'SDP_D_DP_EW_ALU_CVT_TRUNCATE_VALUE',
            'NVDLA_SDP_EW_ALU_CVT_TRUNCATE': 0
        })
        self.ila_asm.append({
            'name': 'SDP_D_DP_EW_MUL_CFG',
            'NVDLA_SDP_EW_MUL_SRC': 0,
            'NVDLA_SDP_EW_MUL_CVT_BYPASS': 0
        })
        self.ila_asm.append({
            'name': 'SDP_D_DP_EW_MUL_SRC_VALUE',
            'NVDLA_SDP_EW_MUL_OPERAND': 0
        })
        self.ila_asm.append({
            'name': 'SDP_D_DP_EW_MUL_CVT_OFFSET_VALUE',
            'NVDLA_SDP_EW_MUL_CVT_OFFSET': 0
        })
        self.ila_asm.append({
            'name': 'SDP_D_DP_EW_MUL_CVT_SCALE_VALUE',
            'NVDLA_SDP_EW_MUL_CVT_SCALE': 0
        })
        self.ila_asm.append({
            'name': 'SDP_D_DP_EW_MUL_CVT_TRUNCATE_VALUE',
            'NVDLA_SDP_EW_MUL_CVT_TRUNCATE': 0
        })
        self.ila_asm.append({
            'name': 'SDP_D_DP_EW_TRUNCATE_VALUE',
            'NVDLA_SDP_EW_TRUNCATE': 0
        })
        # SDP_D_FEATURE_MODE_CFG is in every produce function
        # use external DRAM
        self.ila_asm.append({
            'name': 'SDP_D_DST_DMA_CFG',
            'NVDLA_SDP_DST_RAM_TYPE': 1
        })
        self.ila_asm.append({
            'name': 'SDP_D_DST_BATCH_STRIDE',
            'NVDLA_SDP_DST_BATCH_STRIDE': int(w*h*16*2/2**5)
        })
        # turn off perf counting
        self.ila_asm.append({
            'name': 'SDP_D_PERF_ENABLE',
            'NVDLA_SDP_PERF_DMA_EN': 0,
            'NVDLA_SDP_PERF_LUT_EN': 0,
            'NVDLA_SDP_PERF_SAT_EN': 0,
            'NVDLA_SDP_PERF_NAN_INF_COUNT_EN': 0
        })

        if self.op_name == 'layer_relu':
            self.produce_relu_asm()
        if self.op_name == 'channel_bias_add':
            self.produce_bias_add_asm()
        if self.op_name == 'elemwise_max':
            self.produce_elemwise_max_min_add_equal_asm(op='max')
        if self.op_name == 'elemwise_min':
            self.produce_elemwise_max_min_add_equal_asm(op='min')
        if self.op_name == 'elemwise_add':
            self.produce_elemwise_max_min_add_equal_asm(op='add')
        if self.op_name == 'elemwise_equal':
            self.produce_elemwise_max_min_add_equal_asm(op='equal')
        if self.op_name == 'elemwise_mul' or self.op_name == 'channel_mul':
            self.produce_mul_prelu_asm(op='mul')
        if self.op_name == 'channel_prelu':
            self.produce_mul_prelu_asm(op='prelu')
        if self.op_name == 'channel_batch_norm':
            self.produce_batch_norm(per='channel')

    def produce_lut_asm(self):
        """Currently unused but this function would set up any used functions"""
        # write to table 0 (table X)
        self.ila_asm.append({
            'name': 'SDP_S_LUT_ACCESS_CFG',
            'NVDLA_SDP_LUT_TABLE_ID': 0,
            'NVDLA_SDP_LUT_ACCESS_TYPE': 1
        })
        # write some data to the table (repeat for all data and for both tables)
        self.ila_asm.append({
            'name': 'SDP_S_LUT_ACCESS_DATA',
            'NVDLA_SDP_LUT_DATA': 0
        })
        self.ila_asm.append({
            'name': 'SDP_S_LUT_CFG',
            'NVDLA_SDP_LUT_LE_FUNCTION': 0,
            'NVDLA_SDP_LUT_UFLOW_PRIORITY': 0,
            'NVDLA_SDP_LUT_OFLOW_PRIORITY': 0,
            'NVDLA_SDP_LUT_HYBRID_PRIORITY': 0
        })
        self.ila_asm.append({
            'name': 'SDP_S_LUT_INFO',
            'NVDLA_SDP_LUT_LE_INDEX_OFFSET': 0,
            'NVDLA_SDP_LUT_LE_INDEX_SELECT': 0,
            'NVDLA_SDP_LUT_LO_INDEX_SELECT': 0
        })
        self.ila_asm.append({
            'name': 'SDP_S_LUT_LE_START',
            'NVDLA_SDP_LUT_LE_START': 0
        })
        self.ila_asm.append({
            'name': 'SDP_S_LUT_LE_END',
            'NVDLA_SDP_LUT_LE_END': 0
        })
        self.ila_asm.append({
            'name': 'SDP_S_LUT_LO_START',
            'NVDLA_SDP_LUT_LO_START': 0
        })
        self.ila_asm.append({
            'name': 'SDP_S_LUT_LO_END',
            'NVDLA_SDP_LUT_LO_END': 0
        })
        self.ila_asm.append({
            'name': 'SDP_S_LUT_LE_SLOPE_SCALE',
            'NVDLA_SDP_LUT_LE_SLOPE_UFLOW_SCALE': 0,
            'NVDLA_SDP_LUT_LE_SLOPE_OFLOW_SCALE': 0
        })
        self.ila_asm.append({
            'name': 'SDP_S_LUT_LE_SLOPE_SHIFT',
            'NVDLA_SDP_LUT_LE_SLOPE_UFLOW_SHIFT': 0,
            'NVDLA_SDP_LUT_LE_SLOPE_OFLOW_SHIFT': 0
        })
        self.ila_asm.append({
            'name': 'SDP_S_LUT_LO_SLOPE_SCALE',
            'NVDLA_SDP_LUT_LO_SLOPE_UFLOW_SCALE': 0,
            'NVDLA_SDP_LUT_LO_SLOPE_OFLOW_SCALE': 0
        })
        self.ila_asm.append({
            'name': 'SDP_S_LUT_LO_SLOPE_SHIFT',
            'NVDLA_SDP_LUT_LO_SLOPE_UFLOW_SHIFT': 0,
            'NVDLA_SDP_LUT_LO_SLOPE_OFLOW_SHIFT': 0
        })

    def produce_relu_asm(self):
        """
        produce asm fragment specifics needed for relu
        """
        # bypass alu and mul, but leave relu on
        self.ila_asm.append({
            'name': 'SDP_D_DP_BS_CFG',
            'NVDLA_SDP_BS_BYPASS': 0,
            'NVDLA_SDP_BS_ALU_BYPASS': 1,
            'NVDLA_SDP_BS_ALU_ALGO': 0,
            'NVDLA_SDP_BS_MUL_BYPASS': 1,
            'NVDLA_SDP_BS_MUL_PRELU': 0,
            'NVDLA_SDP_BS_RELU_BYPASS': 0
        })
        # bypass x2 and y
        self.ila_asm.append({
            'name': 'SDP_D_DP_BN_CFG',
            'NVDLA_SDP_BN_BYPASS': 1,
            'NVDLA_SDP_BN_ALU_BYPASS': 0,
            'NVDLA_SDP_BN_ALU_ALGO': 0,
            'NVDLA_SDP_BN_MUL_BYPASS': 0,
            'NVDLA_SDP_BN_MUL_PRELU': 0,
            'NVDLA_SDP_BN_RELU_BYPASS': 0
        })
        self.ila_asm.append({
            'name': 'SDP_D_DP_EW_CFG',
            'NVDLA_SDP_EW_BYPASS': 1,
            'NVDLA_SDP_EW_ALU_BYPASS': 0,
            'NVDLA_SDP_EW_ALU_ALGO': 0,
            'NVDLA_SDP_EW_MUL_BYPASS': 0,
            'NVDLA_SDP_EW_MUL_PRELU': 0,
            'NVDLA_SDP_EW_LUT_BYPASS': 1
        })
        # if flying false (0), data from mrdma_data
        # if flying true (1), data from cacc_data (prev sub-unit)
        # here use false
        self.ila_asm.append({
            'name': 'SDP_D_FEATURE_MODE_CFG',
            'NVDLA_SDP_FLYING_MODE': 0,
            'NVDLA_SDP_OUTPUT_DST': 0,
            'NVDLA_SDP_WINOGRAD': 0,
            'NVDLA_SDP_NAN_TO_ZERO': 0,
            'NVDLA_SDP_BATCH_NUMBER': 0
        })
        # unused but include for completeness
        self.ila_asm.append({
            'name': 'SDP_D_DP_BS_ALU_CFG',
            'NVDLA_SDP_BS_ALU_SRC': 0,
            'NVDLA_SDP_BS_ALU_SHIFT_VALUE': 0,
        })
        # unused but include for completeness
        self.ila_asm.append({
            'name': 'SDP_D_DP_BS_MUL_CFG',
            'NVDLA_SDP_BS_MUL_SRC': 0,
            'NVDLA_SDP_BS_MUL_SHIFT_VALUE': 0,
        })
        # unused but include for completeness
        self.ila_asm.append({
            'name': 'SDP_D_CVT_OFFSET',
            'NVDLA_SDP_CVT_OFFSET': 0
        })
        # unused but include for completeness
        self.ila_asm.append({
            'name': 'SDP_D_CVT_SCALE',
            'NVDLA_SDP_CVT_SCALE': 1
        })
        # unused but include for completeness
        self.ila_asm.append({
            'name': 'SDP_D_CVT_SHIFT',
            'NVDLA_SDP_CVT_SHIFT': 0
        })
        # NVDLA_SDP_PROC_PRECISION unused in sim
        # NVDLA_SDP_OUT_PRECISION: 0 = 8 bit, 1 = 16 bit
        self.ila_asm.append({
            'name': 'SDP_D_DATA_FORMAT',
            'NVDLA_SDP_PROC_PRECISION': 1,
            'NVDLA_SDP_OUT_PRECISION': 1,
        })

        # enable means ready for incoming data
        self.ila_asm.append({
            'name': 'SDP_D_OP_ENABLE',
            'NVDLA_SDP_D_OP_ENABLE': 1
        })
        # 16 atoms per computation for relu
        for _ in range(int(self.inp1_mem_length / 16)):
            self.ila_asm.append({
                'name': 'ReLU_Compute'
            })

    def produce_bias_add_asm(self):
        """
        produce asm fragment specifics needed for bias add
        """
        # bypass relu, don't bypass alu and set to add
        self.ila_asm.append({
            'name': 'SDP_D_DP_BS_CFG',
            'NVDLA_SDP_BS_BYPASS': 0,
            'NVDLA_SDP_BS_ALU_BYPASS': 0,
            'NVDLA_SDP_BS_ALU_ALGO': 2,
            'NVDLA_SDP_BS_MUL_BYPASS': 1,
            'NVDLA_SDP_BS_MUL_PRELU': 0,
            'NVDLA_SDP_BS_RELU_BYPASS': 1
        })
        # bypass x2 and y
        self.ila_asm.append({
            'name': 'SDP_D_DP_BN_CFG',
            'NVDLA_SDP_BN_BYPASS': 1,
            'NVDLA_SDP_BN_ALU_BYPASS': 0,
            'NVDLA_SDP_BN_ALU_ALGO': 0,
            'NVDLA_SDP_BN_MUL_BYPASS': 0,
            'NVDLA_SDP_BN_MUL_PRELU': 0,
            'NVDLA_SDP_BN_RELU_BYPASS': 0
        })
        self.ila_asm.append({
            'name': 'SDP_D_DP_EW_CFG',
            'NVDLA_SDP_EW_BYPASS': 1,
            'NVDLA_SDP_EW_ALU_BYPASS': 0,
            'NVDLA_SDP_EW_ALU_ALGO': 0,
            'NVDLA_SDP_EW_MUL_BYPASS': 0,
            'NVDLA_SDP_EW_MUL_PRELU': 0,
            'NVDLA_SDP_EW_LUT_BYPASS': 1
        })
        # if flying false (0), data from mrdma_data
        # if flying true (1), data from cacc_data (prev sub-unit)
        # here use false
        self.ila_asm.append({
            'name': 'SDP_D_FEATURE_MODE_CFG',
            'NVDLA_SDP_FLYING_MODE': 0,
            'NVDLA_SDP_OUTPUT_DST': 0,
            'NVDLA_SDP_WINOGRAD': 0,
            'NVDLA_SDP_NAN_TO_ZERO': 0,
            'NVDLA_SDP_BATCH_NUMBER': 0
        })
        # BS_ALU_SRC
        #   0 = data from regs_data_alu_ --> constant input
        #   1 = data from dma_data_alu_ --> matrix of input
        self.ila_asm.append({
            'name': 'SDP_D_DP_BS_ALU_CFG',
            'NVDLA_SDP_BS_ALU_SRC': 1,
            'NVDLA_SDP_BS_ALU_SHIFT_VALUE': 0,
        })
        # even if not using mul, output of alu still shifted by the shift value
        self.ila_asm.append({
            'name': 'SDP_D_DP_BS_MUL_CFG',
            'NVDLA_SDP_BS_MUL_SRC': 0,
            'NVDLA_SDP_BS_MUL_SHIFT_VALUE': 0,
        })
        # the simulator still does stuff with CVT before storing
        # so config everything to 0
        self.ila_asm.append({
            'name': 'SDP_D_CVT_OFFSET',
            'NVDLA_SDP_CVT_OFFSET': 0
        })
        self.ila_asm.append({
            'name': 'SDP_D_CVT_SCALE',
            'NVDLA_SDP_CVT_SCALE': 1
        })
        self.ila_asm.append({
            'name': 'SDP_D_CVT_SHIFT',
            'NVDLA_SDP_CVT_SHIFT': 0
        })
        # NVDLA_SDP_PROC_PRECISION unused in sim
        # NVDLA_SDP_OUT_PRECISION: 0 = 8 bit, 1 = 16 bit
        self.ila_asm.append({
            'name': 'SDP_D_DATA_FORMAT',
            'NVDLA_SDP_PROC_PRECISION': 1,
            'NVDLA_SDP_OUT_PRECISION': 1,
        })

        # enable means ready for incoming data
        self.ila_asm.append({
            'name': 'SDP_D_OP_ENABLE',
            'NVDLA_SDP_D_OP_ENABLE': 1
        })
        # 16 atoms per computation for relu
        for _ in range(int(self.inp1_mem_length / 16)):
            self.ila_asm.append({
                'name': 'ALU_Compute'
            })

    def produce_elemwise_max_min_add_equal_asm(self, op='add'):
        """
        produce asm fragment specifics needed for elemwise max, min, add or equal
        of two matrices
        NB equal expected to return matrix of boolean in relay ... currently returns
        0 = True (equal), 1 = False (not equal)
        """
        alu_op_convert = {
            'max': 0,
            'min': 1,
            'add': 2,
            'equal': 3
        }

        # bypass relu, don't bypass alu and set to add
        self.ila_asm.append({
            'name': 'SDP_D_DP_BS_CFG',
            'NVDLA_SDP_BS_BYPASS': 0,
            'NVDLA_SDP_BS_ALU_BYPASS': 0,
            'NVDLA_SDP_BS_ALU_ALGO': alu_op_convert[op],
            'NVDLA_SDP_BS_MUL_BYPASS': 1,
            'NVDLA_SDP_BS_MUL_PRELU': 0,
            'NVDLA_SDP_BS_RELU_BYPASS': 1
        })
        # bypass x2 and y
        self.ila_asm.append({
            'name': 'SDP_D_DP_BN_CFG',
            'NVDLA_SDP_BN_BYPASS': 1,
            'NVDLA_SDP_BN_ALU_BYPASS': 0,
            'NVDLA_SDP_BN_ALU_ALGO': 0,
            'NVDLA_SDP_BN_MUL_BYPASS': 0,
            'NVDLA_SDP_BN_MUL_PRELU': 0,
            'NVDLA_SDP_BN_RELU_BYPASS': 0
        })
        self.ila_asm.append({
            'name': 'SDP_D_DP_EW_CFG',
            'NVDLA_SDP_EW_BYPASS': 1,
            'NVDLA_SDP_EW_ALU_BYPASS': 0,
            'NVDLA_SDP_EW_ALU_ALGO': 0,
            'NVDLA_SDP_EW_MUL_BYPASS': 0,
            'NVDLA_SDP_EW_MUL_PRELU': 0,
            'NVDLA_SDP_EW_LUT_BYPASS': 1
        })
        # if flying false (0), data from mrdma_data
        # if flying true (1), data from cacc_data (prev sub-unit)
        # here use false
        self.ila_asm.append({
            'name': 'SDP_D_FEATURE_MODE_CFG',
            'NVDLA_SDP_FLYING_MODE': 0,
            'NVDLA_SDP_OUTPUT_DST': 0,
            'NVDLA_SDP_WINOGRAD': 0,
            'NVDLA_SDP_NAN_TO_ZERO': 0,
            'NVDLA_SDP_BATCH_NUMBER': 0
        })
        # BS_ALU_SRC
        #   0 = data from regs_data_alu_ --> constant input
        #   1 = data from dma_data_alu_ --> matrix of input
        self.ila_asm.append({
            'name': 'SDP_D_DP_BS_ALU_CFG',
            'NVDLA_SDP_BS_ALU_SRC': 1,
            'NVDLA_SDP_BS_ALU_SHIFT_VALUE': 0,
        })
        # even if not using mul, output of alu still shifted by the shift value
        self.ila_asm.append({
            'name': 'SDP_D_DP_BS_MUL_CFG',
            'NVDLA_SDP_BS_MUL_SRC': 0,
            'NVDLA_SDP_BS_MUL_SHIFT_VALUE': 0,
        })
        # the simulator still does stuff with CVT before storing
        # so config everything to 0
        self.ila_asm.append({
            'name': 'SDP_D_CVT_OFFSET',
            'NVDLA_SDP_CVT_OFFSET': 0
        })
        self.ila_asm.append({
            'name': 'SDP_D_CVT_SCALE',
            'NVDLA_SDP_CVT_SCALE': 1
        })
        self.ila_asm.append({
            'name': 'SDP_D_CVT_SHIFT',
            'NVDLA_SDP_CVT_SHIFT': 0
        })
        # NVDLA_SDP_PROC_PRECISION unused in sim
        # NVDLA_SDP_OUT_PRECISION: 0 = 8 bit, 1 = 16 bit
        self.ila_asm.append({
            'name': 'SDP_D_DATA_FORMAT',
            'NVDLA_SDP_PROC_PRECISION': 1,
            'NVDLA_SDP_OUT_PRECISION': 1,
        })
        # enable means ready for incoming data
        self.ila_asm.append({
            'name': 'SDP_D_OP_ENABLE',
            'NVDLA_SDP_D_OP_ENABLE': 1
        })
        # 16 atoms per computation for relu
        for _ in range(int(self.inp1_mem_length / 16)):
            self.ila_asm.append({
                'name': 'ALU_Compute'
            })

    def produce_mul_prelu_asm(self, op='mul'):
        """
        produce asm fragment specifics needed for elemwise mul of two matrices
        or prelu per channel (both cases use dma and converter handles elemwise vs per channel)
        """
        # should be 0 for mul, 1 for prelu
        mul_op_convert = {
            'mul': 0,
            'prelu': 1
        }
        # bypass relu, don't bypass alu and set to mul (prelu = 0)
        self.ila_asm.append({
            'name': 'SDP_D_DP_BS_CFG',
            'NVDLA_SDP_BS_BYPASS': 0,
            'NVDLA_SDP_BS_ALU_BYPASS': 1,
            'NVDLA_SDP_BS_ALU_ALGO': 0,
            'NVDLA_SDP_BS_MUL_BYPASS': 0,
            'NVDLA_SDP_BS_MUL_PRELU': mul_op_convert[op],
            'NVDLA_SDP_BS_RELU_BYPASS': 1
        })
        # bypass x2 and y
        self.ila_asm.append({
            'name': 'SDP_D_DP_BN_CFG',
            'NVDLA_SDP_BN_BYPASS': 1,
            'NVDLA_SDP_BN_ALU_BYPASS': 0,
            'NVDLA_SDP_BN_ALU_ALGO': 0,
            'NVDLA_SDP_BN_MUL_BYPASS': 0,
            'NVDLA_SDP_BN_MUL_PRELU': 0,
            'NVDLA_SDP_BN_RELU_BYPASS': 0
        })
        self.ila_asm.append({
            'name': 'SDP_D_DP_EW_CFG',
            'NVDLA_SDP_EW_BYPASS': 1,
            'NVDLA_SDP_EW_ALU_BYPASS': 0,
            'NVDLA_SDP_EW_ALU_ALGO': 0,
            'NVDLA_SDP_EW_MUL_BYPASS': 0,
            'NVDLA_SDP_EW_MUL_PRELU': 0,
            'NVDLA_SDP_EW_LUT_BYPASS': 1
        })
        # if flying false (0), data from mrdma_data
        # if flying true (1), data from cacc_data (prev sub-unit)
        # here use false
        self.ila_asm.append({
            'name': 'SDP_D_FEATURE_MODE_CFG',
            'NVDLA_SDP_FLYING_MODE': 0,
            'NVDLA_SDP_OUTPUT_DST': 0,
            'NVDLA_SDP_WINOGRAD': 0,
            'NVDLA_SDP_NAN_TO_ZERO': 0,
            'NVDLA_SDP_BATCH_NUMBER': 0
        })
        # BS_ALU_SRC
        # not using but have to make sure shift is 0
        self.ila_asm.append({
            'name': 'SDP_D_DP_BS_ALU_CFG',
            'NVDLA_SDP_BS_ALU_SRC': 0,
            'NVDLA_SDP_BS_ALU_SHIFT_VALUE': 0,
        })
        # get input from dma
        self.ila_asm.append({
            'name': 'SDP_D_DP_BS_MUL_CFG',
            'NVDLA_SDP_BS_MUL_SRC': 1,
            'NVDLA_SDP_BS_MUL_SHIFT_VALUE': 0,
        })
        # the simulator still does stuff with CVT before storing
        # so config everything to 0
        self.ila_asm.append({
            'name': 'SDP_D_CVT_OFFSET',
            'NVDLA_SDP_CVT_OFFSET': 0
        })
        self.ila_asm.append({
            'name': 'SDP_D_CVT_SCALE',
            'NVDLA_SDP_CVT_SCALE': 1
        })
        self.ila_asm.append({
            'name': 'SDP_D_CVT_SHIFT',
            'NVDLA_SDP_CVT_SHIFT': 0
        })
        # NVDLA_SDP_PROC_PRECISION unused in sim
        # NVDLA_SDP_OUT_PRECISION: 0 = 8 bit, 1 = 16 bit
        self.ila_asm.append({
            'name': 'SDP_D_DATA_FORMAT',
            'NVDLA_SDP_PROC_PRECISION': 1,
            'NVDLA_SDP_OUT_PRECISION': 1,
        })
        # enable means ready for incoming data
        self.ila_asm.append({
            'name': 'SDP_D_OP_ENABLE',
            'NVDLA_SDP_D_OP_ENABLE': 1
        })
        # 16 atoms per computation for relu
        for _ in range(int(self.inp1_mem_length / 16)):
            self.ila_asm.append({
                'name': 'Mult_Compute'
            })

    def produce_batch_norm(self, per='layer'):
        """
        produce asm fragment for per layer or channel batch norm
        """
        #  0 = 2 operands stored in regs, 1 = operand stored in mem (more than 2)
        per_convert = {
            'layer': 0,
            'channel': 1
        }
        # bypass relu, alu to add (alu_algo=2) and mul to mul (prelu=0)
        self.ila_asm.append({
            'name': 'SDP_D_DP_BS_CFG',
            'NVDLA_SDP_BS_BYPASS': 0,
            'NVDLA_SDP_BS_ALU_BYPASS': 0,
            'NVDLA_SDP_BS_ALU_ALGO': 2,
            'NVDLA_SDP_BS_MUL_BYPASS': 0,
            'NVDLA_SDP_BS_MUL_PRELU': 0,
            'NVDLA_SDP_BS_RELU_BYPASS': 1
        })
        # bypass x2 and y
        self.ila_asm.append({
            'name': 'SDP_D_DP_BN_CFG',
            'NVDLA_SDP_BN_BYPASS': 1,
            'NVDLA_SDP_BN_ALU_BYPASS': 0,
            'NVDLA_SDP_BN_ALU_ALGO': 0,
            'NVDLA_SDP_BN_MUL_BYPASS': 0,
            'NVDLA_SDP_BN_MUL_PRELU': 0,
            'NVDLA_SDP_BN_RELU_BYPASS': 0
        })
        self.ila_asm.append({
            'name': 'SDP_D_DP_EW_CFG',
            'NVDLA_SDP_EW_BYPASS': 1,
            'NVDLA_SDP_EW_ALU_BYPASS': 0,
            'NVDLA_SDP_EW_ALU_ALGO': 0,
            'NVDLA_SDP_EW_MUL_BYPASS': 0,
            'NVDLA_SDP_EW_MUL_PRELU': 0,
            'NVDLA_SDP_EW_LUT_BYPASS': 1
        })
        # if flying false (0), data from mrdma_data
        # if flying true (1), data from cacc_data (prev sub-unit)
        # here use false
        self.ila_asm.append({
            'name': 'SDP_D_FEATURE_MODE_CFG',
            'NVDLA_SDP_FLYING_MODE': 0,
            'NVDLA_SDP_OUTPUT_DST': 0,
            'NVDLA_SDP_WINOGRAD': 0,
            'NVDLA_SDP_NAN_TO_ZERO': 0,
            'NVDLA_SDP_BATCH_NUMBER': 0
        })
        # BS_ALU_SRC
        #   0 = data from regs_data_alu_ --> constant input
        #   1 = data from dma_data_alu_ --> matrix of input
        self.ila_asm.append({
            'name': 'SDP_D_DP_BS_ALU_CFG',
            'NVDLA_SDP_BS_ALU_SRC': per_convert[per],
            'NVDLA_SDP_BS_ALU_SHIFT_VALUE': 0,
        })
        # even if not using mul, output of alu still shifted by the shift value
        self.ila_asm.append({
            'name': 'SDP_D_DP_BS_MUL_CFG',
            'NVDLA_SDP_BS_MUL_SRC': per_convert[per],
            'NVDLA_SDP_BS_MUL_SHIFT_VALUE': 0,
        })
        # the simulator still does stuff with CVT before storing
        # so config everything to 0
        self.ila_asm.append({
            'name': 'SDP_D_CVT_OFFSET',
            'NVDLA_SDP_CVT_OFFSET': 0
        })
        self.ila_asm.append({
            'name': 'SDP_D_CVT_SCALE',
            'NVDLA_SDP_CVT_SCALE': 1
        })
        self.ila_asm.append({
            'name': 'SDP_D_CVT_SHIFT',
            'NVDLA_SDP_CVT_SHIFT': 0
        })
        # NVDLA_SDP_PROC_PRECISION unused in sim
        # NVDLA_SDP_OUT_PRECISION: 0 = 8 bit, 1 = 16 bit
        self.ila_asm.append({
            'name': 'SDP_D_DATA_FORMAT',
            'NVDLA_SDP_PROC_PRECISION': 1,
            'NVDLA_SDP_OUT_PRECISION': 1,
        })
        # enable means ready for incoming data
        self.ila_asm.append({
            'name': 'SDP_D_OP_ENABLE',
            'NVDLA_SDP_D_OP_ENABLE': 1
        })
        # 16 atoms per computation for relu
        for _ in range(int(self.inp1_mem_length / 16)):
            self.ila_asm.append({
                'name': 'Mult_Compute'
            })

    def produce_read_asm(self):
        """
        produce asm for reading data from the RDMA memory of the NVDLA
        """
        dbbif_width = 64  # dbbif configurable to width of 32, 64, 128, 256 or 512-bits
        atoms_per_line = int(dbbif_width / 32)
        for i in range(int(np.ceil(np.prod(self.inp1_shape) / atoms_per_line))):
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

    def collect_data_in(self):
        """
        collect relay data from files
        """
        print('\n--------------------------------------------------------------')
        print('\tcollecting input data')
        print('--------------------------------------------------------------\n')
        # don't forget to reshape input 1 NCHW to NHWC if needed by axis=1 on batch norm
        with open(f'./data/{self.op_name}/inp.json', 'r') as fin:
            self.input1 = np.array(json.load(fin)).astype(
                'int16').reshape(self.inp1_shape)
            if self.axis != None:
                # axis is 0,1,2   ... if 0 then chw to hwc (1,2,0)
                #                     if 1 then hcw to hwc (0,2,1)
                #                     if 2 then hwc to hwc (0,1,2)
                reshaper = [i for i in range(3) if i != self.axis]
                reshaper.append(self.axis)
                self.input1 = np.transpose(self.input1, reshaper)
                self.inp1_shape = self.input1.shape

        if len(self.inp2_shape) > 0:
            with open(f'./data/{self.op_name}/inp2.json', 'r') as fin:
                self.input2 = np.array(json.load(fin)).astype(
                    'int16').reshape(self.inp2_shape)
        if self.op_name == 'channel_batch_norm':
            with open(f'./data/{self.op_name}/inp3.json', 'r') as fin:
                self.input3 = np.array(json.load(fin)).astype(
                    'int16').reshape(self.inp2_shape)

    def produce_prog_frag(self):
        print('\n--------------------------------------------------------------')
        print('\tgenerate prog_frag.json for ILA simulator')
        print('--------------------------------------------------------------\n')
        self.ila_cvtr = SDPConverter(
            f'./test/{self.op_name}/ila_asm.json', self.op_name)
        self.ila_cvtr.dump_ila_prog_frag(
            f'./test/{self.op_name}/ila_prog_frag_input.json')

    def invoke_ila_simulator_and_collect_results(self):
        print('\n--------------------------------------------------------------')
        print(
            f'\tinvoking NVDLA ILA simulator and collecting result')
        print('--------------------------------------------------------------\n')
        start_time = timeit.default_timer()
        result_unshaped = []
        start_dp, end_dp = min(self.ila_cvtr.dp_indices), max(
            self.ila_cvtr.dp_indices)
        cmd = [
            "sdp_sim_driver_fast",
            f'./test/{self.op_name}/ila_prog_frag',
        ]
        subprocess.run(cmd)
        with open(f'./test/{self.op_name}/ila_prog_frag_out.json', 'r') as fin:
            ila_out = json.load(fin)
            for instr_idx in self.ila_cvtr.dp_indices:
                result_unshaped.extend(ila_out[instr_idx])

        # datapath emits int16 so save as int too
        h, w, c = self.inp1_shape
        result_np = np.zeros((1, h, w, c), 'int16')
        ints_per_atom = 16
        idx = 0
        for n_c_large in range(0, math.ceil(c / ints_per_atom)):
            for n_h in range(h):
                for n_w in range(w):
                    # 16 channels per atom
                    for n_c_small in range(ints_per_atom):
                        ch_idx = n_c_large*ints_per_atom + n_c_small
                        if ch_idx < c:
                            if self.op_name == 'elemwise_equal':
                                result_unshaped[idx] = (
                                    1, 0)[result_unshaped[idx] >= 1]
                            result_np[0][n_h][n_w][ch_idx] = result_unshaped[idx]
                        idx += 1
        if self.axis != None:
            # axis is 0,1,2   ... if 0 then nhwc to nchw (0,3,1,2)
            #                     if 1 then nhwc to nhcw (0,1,3,2)
            #                     if 2 then nhwc to nhwc (0,1,2,3)
            reshaper = [i for i in range(3)]
            reshaper = reshaper[:self.axis+1] + [3] + reshaper[self.axis+1:]
            result_np = np.transpose(result_np, reshaper)
        result_np.tofile(f'./data/{self.op_name}/result.txt', sep='\n')
        print(f'result of {self.op_name} is:\n', result_np)

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
        # self.clean_root_dir(deep_clean=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add Parameters')
    parser.add_argument('--inp_shape', nargs='+', type=int)
    parser.add_argument('--inp2_shape', nargs='+', type=int, required=False)
    parser.add_argument('--axis', type=int, required=False)
    parser.add_argument("--op_name", default="relu")
    args = parser.parse_args()

    driver = sdp_driver(inp_shape=args.inp_shape,
                        inp2_shape=args.inp2_shape, op_name=args.op_name, axis=args.axis)
    driver.run()
