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
                 kernels_weight_format, dilation, padding, strides, out_format=None):
        # 4d array - 1 at start because only batch number 1 supported
        self.orig_inp_shape = [1] + inp_shape
        self.orig_out_shape = out_shape
        self.orig_kernels_shape = kernels_shape             # 4d array
        self.inp_weight_format = inp_weight_format          # string
        self.kernels_weight_format = kernels_weight_format  # string
        self.out_format = out_format                        # string
        if self.out_format is None:
            self.out_format = self.inp_weight_format
        # always convert to same internal format N = 1, then height, then width, then channels
        self.desired_inp_format = 'NHWC'
        # always convert to same internal format - same as for inp
        self.desired_kern_format = 'OHWI'
        self.desired_out_internal_format = 'NCHW'
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
            self.kernels_matrix.shape, self.kernels_matrix, 128000 * 2)

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
        numbers_per_block = 16
        for n_n in range(n):
            for n_c_large in range(0, math.ceil(c / numbers_per_block)):
                for n_h in range(h):
                    for n_w in range(w):
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
        # --- PRE-SETUP CDMA, CSC, CMAC_A, CMAC_B, CACC ---
        self.ila_asm.append({
            'name': 'CDMA_S_POINTER',
            'NVDLA_CDMA_PRODUCER': 0,
            'NVDLA_CDMA_CONSUMER': 0
        })
        self.ila_asm.append({
            'name': 'CSC_S_POINTER',
            'NVDLA_CSC_PRODUCER': 0,
            'NVDLA_CSC_CONSUMER': 0
        })
        self.ila_asm.append({
            'name': 'CMAC_A_S_POINTER',
            'NVDLA_CMAC_A_PRODUCER': 0,
            'NVDLA_CMAC_A_CONSUMER': 0
        })
        self.ila_asm.append({
            'name': 'CMAC_B_S_POINTER',
            'NVDLA_CMAC_B_PRODUCER': 0,
            'NVDLA_CMAC_B_CONSUMER': 0
        })
        self.ila_asm.append({
            'name': 'CACC_S_POINTER',
            'NVDLA_CACC_PRODUCER': 0,
            'NVDLA_CACC_CONSUMER': 0
        })

        # --- SETUP CDMA, CSC, CMAC_A, CMAC_B, CACC ---
        # --- CDMA sub-unit configuration ---
        # adjust whether weights or WMB compression gets more priority in external memory access
        self.ila_asm.append({
            'name': 'CDMA_S_ARBITER',
            'NVDLA_CDMA_ARB_WEIGHT': 0,
            'NVDLA_CDMA_ARB_WMB': 0
        })
        # config conv mode, data type, data reuse, weight reuse,
        # if reset nvdla should we worry about releasing SBUF
        # (yes skip release as assume general reset between
        # NN layers due to curr implementation)
        self.ila_asm.append({
            'name': 'CDMA_D_MISC_CFG',
            'NVDLA_CDMA_CONV_MODE': 0,
            'NVDLA_CDMA_IN_PRECISION': 1,
            'NVDLA_CDMA_PROC_PRECISION': 1,
            'NVDLA_CDMA_DATA_REUSE': 0,
            'NVDLA_CDMA_WEIGHT_REUSE': 0,
            'NVDLA_CDMA_SKIP_DATA_RLS': 1
        })
        # a) 0 = feature not image
        # b) unused (see table 33 for formats)
        # c) unused
        # d) unused
        self.ila_asm.append({
            'name': 'CDMA_D_DATAIN_FORMAT',
            'NVDLA_CDMA_DATAIN_FORMAT': 0,
            'NVDLA_CDMA_PIXEL_FORMAT': 0,
            'NVDLA_CDMA_PIXEL_MAPPING ': 0,
            'NVDLA_CDMA_PIXEL_SIGN_OVERRIDE': 0
        })
        # input data dimensions
        n, h, w, c = self.inp_matrix.shape
        self.ila_asm.append({
            'name': 'CDMA_D_DATAIN_SIZE_0',
            'NVDLA_CDMA_DATAIN_WIDTH': w,
            'NVDLA_CDMA_DATAIN_HEIGHT': h
        })
        # input data channels
        self.ila_asm.append({
            'name': 'CDMA_D_DATAIN_SIZE_1',
            'NVDLA_CDMA_DATAIN_CHANNEL': c
        })
        # width and height after extension
        self.ila_asm.append({
            'name': 'CDMA_D_DATAIN_SIZE_EXT_0',
            'NVDLA_CDMA_DATAIN_WIDTH_EXT': w,
            'NVDLA_CDMA_DATAIN_HEIGHT_EXT': h
        })
        # unused pixel offset
        self.ila_asm.append({
            'name': 'CDMA_D_PIXEL_OFFSET',
            'NVDLA_CDMA_PIXEL_X_OFFSET': 0,
            'NVDLA_CDMA_PIXEL_Y_OFFSET': 0
        })
        # input memory is RAM (use MCIF)
        self.ila_asm.append({
            'name': 'CDMA_D_DAIN_RAM_TYPE',
            'NVDLA_CDMA_DATAIN_RAM_TYPE': 1
        })
        # input memory address
        self.ila_asm.append({
            'name': 'CDMA_D_DAIN_ADDR_HIGH_0',
            'NVDLA_CDMA_DATAIN_ADDR_HIGH_0': 0
        })
        self.ila_asm.append({
            'name': 'CDMA_D_DAIN_ADDR_LOW_0',
            'NVDLA_CDMA_DATAIN_ADDR_LOW_0': 0
        })
        # uv plane address for input data (unused)
        self.ila_asm.append({
            'name': 'CDMA_D_DAIN_ADDR_HIGH_1',
            'NVDLA_CDMA_DATAIN_ADDR_HIGH_1': 0
        })
        self.ila_asm.append({
            'name': 'CDMA_D_DAIN_ADDR_LOW_1',
            'NVDLA_CDMA_DATAIN_ADDR_LOW_1': 0
        })
        # input line stride
        self.ila_asm.append({
            'name': 'CDMA_D_LINE_STRIDE',
            'NVDLA_CDMA_LINE_STRIDE': int(w*16*2/2**5)
        })
        # uv line stride (unused)
        self.ila_asm.append({
            'name': 'CDMA_D_LINE_UV_STRIDE',
            'NVDLA_CDMA_UV_LINE_STRIDE': 0
        })
        # input surface stride
        self.ila_asm.append({
            'name': 'CDMA_D_SURF_STRIDE',
            'NVDLA_CDMA_SURF_STRIDE': int(w*h*16*2/2**5)
        })
        # is the data cube as tightly packed as possible (probably not)
        self.ila_asm.append({
            'name': 'CDMA_D_DAIN_MAP',
            'NVDLA_CDMA_LINE_PACKED': 0,
            'NVDLA_CDMA_SURF_PACKED': 0
        })
        # only batch of 1 supported
        self.ila_asm.append({
            'name': 'CDMA_D_BATCH_NUMBER',
            'NVDLA_CDMA_BATCHES': 1
        })
        # batch stride is almost hence same as surface stride but for whole cube
        channel_layers = np.ceil(c/16)
        self.ila_asm.append({
            'name': 'CDMA_D_BATCH_STRIDE',
            'NVDLA_CDMA_BATCH_STRIDE': int(w*h*channel_layers*16*2/2**5)
        })
        # entry per slice
        # based on nvdlahw/cmod/csc/NV_NVDLA_cdma.cpp
        cbuf_entry_per_slice = (channel_layers / 4) * w
        if ((channel_layers % 4) == 3):
            cbuf_entry_per_slice += w
        elif ((channel_layers % 4) == 2):
            cbuf_entry_per_slice += (w + 1)/2
        elif ((channel_layers % 4) == 1):
            cbuf_entry_per_slice += (w + 3)/4
        self.ila_asm.append({
            'name': 'CDMA_D_ENTRY_PER_SLICE',
            'NVDLA_CDMA_ENTRIES': int(cbuf_entry_per_slice)
        })
        # when space for new slice send new slice
        self.ila_asm.append({
            'name': 'CDMA_D_FETCH_GRAIN',
            'NVDLA_CDMA_FETCH_GRAIN': 1
        })
        # uncompressed weight (kernels) data
        self.ila_asm.append({
            'name': 'CDMA_D_WEIGHT_FORMAT',
            'NVDLA_CDMA_WEIGHT_FORMAT': 0
        })
        # bytes per kernel
        k_n, k_h, k_w, k_c = self.kernels_matrix.shape
        kernel_channel_layers = np.ceil(k_c/16)
        self.ila_asm.append({
            'name': 'CDMA_D_WEIGHT_SIZE_0',
            'NVDLA_CDMA_BYTE_PER_KERNEL': int(k_w*k_h*kernel_channel_layers*16*2)
        })
        # number of kernels
        self.ila_asm.append({
            'name': 'CDMA_D_WEIGHT_SIZE_1',
            'NVDLA_CDMA_WEIGHT_KERNEL': int(k_n)
        })
        # stored in external ram
        self.ila_asm.append({
            'name': 'CDMA_D_WEIGHT_RAM_TYPE',
            'NVDLA_CDMA_WEIGHT_RAM_TYPE': 1
        })
        # address of kernels
        self.ila_asm.append({
            'name': 'CDMA_D_WEIGHT_ADDR_HIGH',
            'NVDLA_CDMA_WEIGHT_ADDR_HIGH': 0
        })
        self.ila_asm.append({
            'name': 'CDMA_D_WEIGHT_ADDR_LOW',
            'NVDLA_CDMA_WEIGHT_ADDR_LOW': int(256000/2**5)
        })
        # kernel bytes total
        self.ila_asm.append({
            'name': 'CDMA_D_WEIGHT_BYTES',
            'NVDLA_CDMA_WEIGHT_BYTES': int(k_n*k_w*k_h*kernel_channel_layers*16*2/2**7)
        })
        # compression address high and low (unused)
        self.ila_asm.append({
            'name': 'CDMA_D_WGS_ADDR_HIGH',
            'NVDLA_CDMA_WGS_ADDR_HIGH': 0
        })
        self.ila_asm.append({
            'name': 'CDMA_D_WGS_ADDR_LOW',
            'NVDLA_CDMA_WGS_ADDR_LOW': 0
        })
        self.ila_asm.append({
            'name': 'CDMA_D_WMB_ADDR_HIGH',
            'NVDLA_CDMA_WMB_ADDR_HIGH': 0
        })
        self.ila_asm.append({
            'name': 'CDMA_D_WMB_ADDR_LOW',
            'NVDLA_CDMA_WMB_ADDR_LOW': 0
        })
        # compression bytes (unused)
        self.ila_asm.append({
            'name': 'CDMA_D_WMB_BYTES',
            'NVDLA_CDMA_WMB_BYTES': 0
        })
        # don't use mean registers (unused)
        self.ila_asm.append({
            'name': 'CDMA_D_MEAN_FORMAT',
            'NVDLA_CDMA_MEAN_FORMAT': 0
        })
        # mean registers (unused)
        self.ila_asm.append({
            'name': 'CDMA_D_MEAN_GLOBAL_0',
            'NVDLA_CDMA_MEAN_RY': 0,
            'NVDLA_CDMA_MEAN_GU': 0
        })
        self.ila_asm.append({
            'name': 'CDMA_D_MEAN_GLOBAL_1',
            'NVDLA_CDMA_MEAN_BV': 0,
            'NVDLA_CDMA_MEAN_AX': 0
        })
        # disable cvt scaling
        self.ila_asm.append({
            'name': 'CDMA_D_CVT_CFG',
            'NVDLA_CDMA_CVT_EN': 0,
            'NVDLA_CDMA_CVT_TRUNCATE': 0
        })
        # cvt params (unused)
        self.ila_asm.append({
            'name': 'CDMA_D_CVT_OFFSET',
            'NVDLA_CDMA_CVT_OFFSET': 0
        })
        self.ila_asm.append({
            'name': 'CDMA_D_CVT_SCALE',
            'NVDLA_CDMA_CVT_SCALE': 0
        })
        # conv stride params
        self.ila_asm.append({
            'name': 'CDMA_D_CONV_STRIDE',
            'NVDLA_CDMA_CONV_X_STRIDE': self.strides[1],
            'NVDLA_CDMA_CONV_Y_STRIDE': self.strides[0]
        })
        # padding .... simulator didn't have padding so driver does padding
        # so here we set padding to 0 ... in future set to padding[1] and padding[0]
        self.ila_asm.append({
            'name': 'CDMA_D_ZERO_PADDING',
            'NVDLA_CDMA_PAD_LEFT': 0,
            'NVDLA_CDMA_PAD_RIGHT': 0,
            'NVDLA_CDMA_PAD_TOP': 0,
            'NVDLA_CDMA_PAD_BOTTOM': 0
        })
        self.ila_asm.append({
            'name': 'CDMA_D_ZERO_PADDING_VALUE',
            'NVDLA_CDMA_PAD_VALUE': 0
        })
        # how to alocate 16 banks in CBUF to input matrix vs kernel
        self.ila_asm.append({
            'name': 'CDMA_D_BANK',
            'NVDLA_CDMA_BANK': 6,
            'NVDLA_CDMA_WEIGHT_BANK': 10
        })
        # enable flush nan to zero but won't have with int16 (unused)
        self.ila_asm.append({
            'name': 'CDMA_D_NAN_FLUSH_TO_ZERO',
            'NVDLA_CDMA_NAN_TO_ZERO': 1
        })
        # turn off performance counters
        self.ila_asm.append({
            'name': 'CDMA_D_PERF_ENABLE',
            'NVDLA_CDMA_DMA_EN': 0
        })

        # --- CSC sub-unit configuration ---
        # misc cfg - mode = direct not winograd
        self.ila_asm.append({
            'name': 'CSC_D_MISC_CFG',
            'NVDLA_CSC_CONV_MODE': 0,
            'NVDLA_CSC_IN_PRECISION': 1,
            'NVDLA_CSC_PROC_PRECISION': 1,
            'NVDLA_CSC_DATA_REUSE': 0,
            'NVDLA_CSC_WEIGHT_REUSE': 0,
            'NVDLA_CSC_SKIP_DATA_RLS': 1,
            'NVDLA_CSC_SKIP_WEIGHT_RLS': 1
        })
        # feature not image input data
        self.ila_asm.append({
            'name': 'CSC_D_DATAIN_FORMAT',
            'NVDLA_CSC_DATAIN_FORMAT': 0
        })
        # dimensions input data post extension (but not doing any)
        self.ila_asm.append({
            'name': 'CSC_D_DATAIN_SIZE_EXT_0',
            'NVDLA_CSC_DATAIN_WIDTH_EXT': w,
            'NVDLA_CSC_DATAIN_HEIGHT_EXT': h
        })
        self.ila_asm.append({
            'name': 'CSC_D_DATAIN_SIZE_EXT_1',
            'NVDLA_CSC_DATAIN_CHANNEL_EXT': c
        })
        assert n == 1, "Batch size over 1 not supported by driver currently"
        self.ila_asm.append({
            'name': 'CSC_D_BATCH_NUMBER',
            'NVDLA_CSC_BATCH_NUMBER': 1
        })
        # post extension params (unused)
        self.ila_asm.append({
            'name': 'CSC_D_POST_Y_EXTENSION',
            'NVDLA_CSC_Y_EXTENSION': 0
        })
        # entry per slice (same calc as above)
        self.ila_asm.append({
            'name': 'CSC_D_ENTRY_PER_SLICE',
            'NVDLA_CSC_ENTRIES': int(cbuf_entry_per_slice)
        })
        # uncompressed weights
        self.ila_asm.append({
            'name': 'CSC_D_WEIGHT_FORMAT',
            'NVDLA_CSC_WEIGHT_FORMAT': 0
        })
        # dimensions of kernel
        self.ila_asm.append({
            'name': 'CSC_D_WEIGHT_SIZE_EXT_0',
            'NVDLA_CSC_WEIGHT_WIDTH_EXT': k_w,
            'NVDLA_CSC_WEIGHT_HEIGHT_EXT': k_h
        })
        self.ila_asm.append({
            'name': 'CSC_D_WEIGHT_SIZE_EXT_1',
            'NVDLA_CSC_WEIGHT_CHANNEL_EXT': k_c,
            'NVDLA_CSC_WEIGHT_KERNEL': k_n
        })
        # kernel and compression data (none) size
        self.ila_asm.append({
            'name': 'CSC_D_WEIGHT_BYTES',
            'NVDLA_CSC_WEIGHT_BYTES': int(k_n*k_w*k_h*kernel_channel_layers*16*2/2**5)
        })
        self.ila_asm.append({
            'name': 'CSC_D_WMB_BYTES',
            'NVDLA_CSC_WMB_BYTES': 0
        })
        # output dimensions
        out_n, out_c, out_h, out_w = self.orig_out_shape
        self.ila_asm.append({
            'name': 'CSC_D_DATAOUT_SIZE_0',
            'NVDLA_CSC_DATAOUT_WIDTH': out_w,
            'NVDLA_CSC_DATAOUT_HEIGHT': out_h
        })
        self.ila_asm.append({
            'name': 'CSC_D_DATAOUT_SIZE_1',
            'NVDLA_CSC_DATAOUT_CHANNEL': out_c
        })
        # atomics ... in future check if should be out_w*(out_h -1)
        self.ila_asm.append({
            'name': 'CSC_D_ATOMICS',
            'NVDLA_CSC_ATOMICS': out_w*out_h - 1
        })
        # release 1 slice of cbuf after current layer
        self.ila_asm.append({
            'name': 'CSC_D_RELEASE',
            'NVDLA_CSC_RELEASE': 1
        })
        # conv stride and dilation after extension is same as before as no extension
        self.ila_asm.append({
            'name': 'CSC_D_CONV_STRIDE_EXT',
            'NVDLA_CSC_CONV_X_STRIDE_EXT': self.strides[1],
            'NVDLA_CSC_CONV_Y_STRIDE_EXT': self.strides[0]
        })
        self.ila_asm.append({
            'name': 'CSC_D_DILATION_EXT',
            'NVDLA_CSC_X_DILATION_EXT': self.dilation[1],
            'NVDLA_CSC_Y_DILATION_EXT': self.dilation[0]
        })
        # padding is still 0 due to it being done in the driver
        self.ila_asm.append({
            'name': 'CSC_D_ZERO_PADDING',
            'NVDLA_CSC_PAD_LEFT': 0,
            'NVDLA_CSC_PAD_TOP': 0
        })
        self.ila_asm.append({
            'name': 'CSC_D_ZERO_PADDING_VALUE',
            'NVDLA_CSC_PAD_VALUE': 0
        })
        # cbuf banks setup as before
        self.ila_asm.append({
            'name': 'CSC_D_BANK',
            'NVDLA_CSC_DATA_BANK': 6,
            'NVDLA_CSC_WEIGHT_BANK': 10
        })
        # pra trunctate for winograd (unused as not using winograd)
        self.ila_asm.append({
            'name': 'CSC_D_PRA_CFG',
            'NVDLA_CSC_PRA_TRUNCATE': 0
        })

        # --- CMAC_A config ---
        # Conv mode does joining sub-units. In simulator CMAC_A_D_MISC_CFG currently unused
        # NVDLA_CMAC_A_PROC_PRECISION: 0 = 8 bit, 1 = 16 bit
        self.ila_asm.append({
            'name': 'CMAC_A_D_MISC_CFG',
            'NVDLA_CMAC_A_CONV_MODE': 0,
            'NVDLA_CMAC_A_PROC_PRECISION': 1
        })
        # --- CMAC_B config ---
        # Same as above but simulator just does both A and B so only
        # CMAC_A config commands are used for the simulator
        self.ila_asm.append({
            'name': 'CMAC_B_D_MISC_CFG',
            'NVDLA_CMAC_B_CONV_MODE': 0,
            'NVDLA_CMAC_B_PROC_PRECISION': 1
        })

        # --- CACC config ---
        # same as above
        self.ila_asm.append({
            'name': 'CACC_D_MISC_CFG',
            'NVDLA_CACC_CONV_MODE': 0,
            'NVDLA_CACC_PROC_PRECISION': 1
        })
        # output dimensions
        self.ila_asm.append({
            'name': 'CACC_D_DATAOUT_SIZE_0',
            'NVDLA_CACC_DATAOUT_WIDTH': out_w,
            'NVDLA_CACC_DATAOUT_HEIGHT': out_h
        })
        self.ila_asm.append({
            'name': 'CACC_D_DATAOUT_SIZE_1',
            'NVDLA_CACC_DATAOUT_CHANNEL': out_c
        })
        # out addr
        # TOOD: this address is silly - fix it across the board
        self.ila_asm.append({
            'name': 'CACC_D_DATAOUT_ADDR',
            'NVDLA_CACC_DATAOUT_ADDR': int(512000/2**5)
        })
        # batch number
        assert out_n == 1, "Output batch number not 1"
        self.ila_asm.append({
            'name': 'CACC_D_BATCH_NUMBER',
            'NVDLA_CACC_BATCH_NUMBER': out_n
        })
        # output line stride
        self.ila_asm.append({
            'name': 'CACC_D_LINE_STRIDE',
            'NVDLA_CACC_LINE_STRIDE': int(out_w*16*2/2**5)
        })
        # output surface stride
        self.ila_asm.append({
            'name': 'CACC_D_SURF_STRIDE',
            'NVDLA_CACC_SURF_STRIDE': int(out_w*out_h*16*2/2**5)
        })
        # not packed
        self.ila_asm.append({
            'name': 'CACC_D_DATAOUT_MAP',
            'NVDLA_CACC_LINE_PACKED': 0,
            'NVDLA_CACC_LINE_PACKED': 0
        })
        # clip down to 32 bit for sdp from 48 bit
        self.ila_asm.append({
            'name': 'CACC_D_CLIP_CFG',
            'NVDLA_CACC_CLIP_TRUNCATE': 16
        })
        # saturation
        self.ila_asm.append({
            'name': 'CACC_D_OUT_SATURATION',
            'NVDLA_CACC_SAT_COUNT': 16
        })

        # --- Enable CDMA, CSC, CMAC_A, CMAC_B, CACC sub-units ---
        self.ila_asm.append({
            'name': 'CDMA_D_OP_ENABLE',
            'NVDLA_CDMA_D_OP_ENABLE': 1
        })
        self.ila_asm.append({
            'name': 'CSC_D_OP_ENABLE',
            'NVDLA_CSC_D_OP_ENABLE': 1
        })
        self.ila_asm.append({
            'name': 'CMAC_A_D_OP_ENABLE',
            'NVDLA_CMAC_A_D_OP_ENABLE': 1
        })
        self.ila_asm.append({
            'name': 'CMAC_B_D_OP_ENABLE',
            'NVDLA_CMAC_B_D_OP_ENABLE': 1
        })
        self.ila_asm.append({
            'name': 'CACC_D_OP_ENABLE',
            'NVDLA_CACC_D_OP_ENABLE': 1
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
        """Iterate through all inputs this kernel atom can touch
        n_h and n_w is positon in kernel matrix that is being processed"""
        n, h, w, c = self.inp_matrix.shape
        kern_n, kern_h, kern_w, kern_c = self.kernels_matrix.shape
        stride_w, stride_h = self.strides
        dilation_w, dilation_h = self.dilation
        cmac_calls = 0  # min
        numbers_per_block = 64
        _, _, out_h, out_w = self.orig_out_shape
        for n_h_inp in range(n_h*dilation_h, stride_h*out_h+n_h*dilation_h, stride_h):
            for n_w_inp in range(n_w*dilation_w, stride_w*out_w+n_w*dilation_w, stride_w):
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
                # don't include n_c_large as in the end will combine all n_c_large together
                # so don't need to distinguish them
                out_pos_h = int((n_h_inp-n_h*dilation_h)/stride_h)
                out_pos_w = int((n_w_inp-n_w*dilation_w)/stride_w)
                cmac_op['out_pos'] = f'{n_n_large}_{out_pos_h}_{out_pos_w}'
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
        # assumes other sub-units have run and that the result has been stored back in memory
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
                self.inp_weight_format, self.desired_inp_format, self.inp_matrix)
            # pad as necessary
            n, h, w, c = self.inp_matrix.shape
            pad_h, pad_w = self.padding
            temp_matrix = np.zeros(
                shape=(n, h+2*pad_h, w+2*pad_w, c), dtype='int16')
            temp_matrix[0, pad_h:h+pad_h,
                        pad_w:w+pad_w, :] = self.inp_matrix
            self.inp_matrix = temp_matrix
        print('Max of input:', np.max(self.inp_matrix))

        # kernels
        with open(f'./data/{self.op_name}/kernels.json', 'r') as fin:
            self.kernels_matrix = np.array(json.load(fin)).astype(
                'int16').reshape(self.orig_kernels_shape)
            self.kernels_matrix = self.transform_matrix(
                self.kernels_weight_format, self.desired_kern_format, self.kernels_matrix)
        print('Max of kernels:', np.max(self.kernels_matrix))

        # output shape - ensure in expected order
        output_shape_reordering = self.transpose_new_order(
            self.out_format, self.desired_out_internal_format)
        self.orig_out_shape = [self.orig_out_shape[idx]
                               for idx in output_shape_reordering]

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
                        # retrieve vals for that line using sim_output_idx
                        line_vals = sim_output[sim_output_idx]
                        for n_k in range(16):
                            k_idx = n_k_large*16 + n_k
                            if k_idx < k:
                                # allow overflow of int 16 to negative as default relay behaviour
                                nkhw_out[0][k_idx][n_h][n_w] += line_vals[n_k]
        # reshape as desired
        nkhw_out = self.transform_matrix(
            self.desired_out_internal_format, self.out_format, nkhw_out)
        nkhw_out.tofile(f'./data/{self.op_name}/result.txt', sep='\n')
        print(f'result of {self.op_name} is:', nkhw_out)
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
    parser.add_argument('--out_format', type=str, required=False)
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
                         strides=args.strides,
                         out_format=args.out_format
                         )
    driver.run()
