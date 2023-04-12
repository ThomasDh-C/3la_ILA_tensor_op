import json
import os
import numpy as np


class SDPConverter:
    def __init__(self, asm_path, op_name):
        # load asm and data_lib from files
        with open(asm_path, 'r') as fin:
            self.asm_list = json.load(fin)['asm']
        self.op_name = op_name
        self.prog_frag = None
        self.inp_addr = 0
        self.inp_addr_end = 0
        self.inp1_data = ''     # main data
        self.inp1_shape = [0, 0, 0]  # w, h, c
        self.inp2_data = ''     # operand
        self.dp_indices = []    # prog frag indices where datapath called

        # operand info (second number in operation of alu or mul)
        self.bs_alu_src = None  # operand is from ... 0 = reg, 1 = dma
        self.bn_alu_src = None  # operand is from ... 0 = reg, 1 = dma
        self.ew_alu_src = None  # operand is from ... 0 = reg, 1 = dma
        self.bs_mul_src = None  # operand is from ... 0 = reg, 1 = dma
        self.bn_mul_src = None  # operand is from ... 0 = reg, 1 = dma
        self.ew_mul_src = None  # operand is from ... 0 = reg, 1 = dma
        self.bs_reg_alu_operand = None
        self.bn_reg_alu_operand = None
        self.ew_reg_alu_operand = None
        self.bs_reg_mul_operand = None
        self.bn_reg_mul_operand = None
        self.ew_reg_mul_operand = None

        # ops can be elemwise (dma wxhxc), per channel (dma 1x1xc or 1x1x2c if bias_add)
        # or per layer (reg) --> used to correctly generate dp instr
        self.elemwise_op = self.op_name[:8] == 'elemwise'

        path = os.path.dirname(os.path.abspath(__file__))
        with open(path + '/all_param_shifts_for_csb.json', 'r') as fin:
            self.sdp_param_shifts_for_csb = json.load(fin)

    def to_ila_prog_frag(self):
        """
        convert ILA assembly to program fragment if not done already
        """
        dp_instrs = {'ReLU_Compute', 'ALU_Compute',
                     'Mult_Compute', 'BatchNorm_Compute'}
        dp_instrs_not_supported = {'LUT_Write', 'LUT_Read'}
        nb_vals_in_layer_seen = 0  # used for per channel
        alu_op_num, mul_op_num = 0, 0
        alu_op_nums, mul_op_nums = [0 for i in range(16)], [
            0 for i in range(16)]
        self.prog_frag = []
        for asm_line in self.asm_list:
            if asm_line['name'] in self.sdp_param_shifts_for_csb:
                if 'SDP' not in asm_line['name']:
                    print('ERROR!!!!', asm_line['name'])
                    continue
                # append to prog frag based on that and find inp_addr_end
                instr = {"instr No.": len(self.prog_frag)}
                for channel in ['cacc_data', 'mrdma_data', 'regs_data_alu', 'regs_data_mult', 'dma_data_alu', 'dma_data_mult']:
                    for channel_idx in range(16):
                        instr[f'{channel}_{channel_idx}'] = 0
                name_conversion_dict = self.sdp_param_shifts_for_csb[asm_line['name']]
                # first part of address as simulator can't understand it
                instr['csb_addr'] = '0x' + name_conversion_dict['addr'][3:]
                # create csb data
                data = ''
                all_params = list(name_conversion_dict['shifts'].items())
                all_params.sort(key=lambda x: x[1][0])
                for param, shift in all_params:
                    # if gaps between params pad with 0s
                    start_shift, end_shift = shift
                    if len(data) < start_shift:
                        data = '0'*(start_shift - len(data)) + data
                    # convert param to binary
                    param_desired_length = end_shift - start_shift + 1
                    param_binary = bin(asm_line[param])[2:]
                    if len(param_binary) < param_desired_length:
                        param_binary = '0' * \
                            (param_desired_length - len(param_binary)) + param_binary
                    # add onto data
                    data = param_binary + data
                # data for registers is put in int format in example
                instr['csb_data'] = int(data, 2)
                instr['csb_write'] = 1
                instr['csb_vld'] = 1
                instr['csb_rdy'] = 1
                instr['fifo_clr'] = 0
                instr['done'] = 0
                self.prog_frag.append(instr)

                # unique instructions that set registers important to later datapath instructions
                if asm_line['name'] == 'SDP_D_DP_BS_ALU_CFG':
                    self.bs_alu_src = asm_line['NVDLA_SDP_BS_ALU_SRC']
                if asm_line['name'] == 'SDP_D_DP_BN_ALU_CFG':
                    self.bn_alu_src = asm_line['NVDLA_SDP_BN_ALU_SRC']
                if asm_line['name'] == 'SDP_D_DP_EW_ALU_CFG':
                    self.ew_alu_src = asm_line['NVDLA_SDP_EW_ALU_SRC']
                if asm_line['name'] == 'SDP_D_DP_BS_MUL_CFG':
                    self.bs_mul_src = asm_line['NVDLA_SDP_BS_MUL_SRC']
                if asm_line['name'] == 'SDP_D_DP_BN_MUL_CFG':
                    self.bn_mul_src = asm_line['NVDLA_SDP_BN_MUL_SRC']
                if asm_line['name'] == 'SDP_D_DP_EW_MUL_CFG':
                    self.ew_mul_src = asm_line['NVDLA_SDP_EW_MUL_SRC']
                if asm_line['name'] == 'SDP_D_DP_BS_ALU_SRC_VALUE':
                    self.bs_reg_alu_operand = asm_line['NVDLA_SDP_DP_BS_ALU_SRC_VALUE']
                if asm_line['name'] == 'SDP_D_DP_BS_MUL_SRC_VALUE':
                    self.bs_reg_mul_operand = asm_line['NVDLA_SDP_DP_BS_MUL_SRC_VALUE']
                if asm_line['name'] == 'SDP_D_DP_BN_ALU_SRC_VALUE':
                    self.bn_reg_alu_operand = asm_line['NVDLA_SDP_DP_BN_ALU_SRC_VALUE']
                if asm_line['name'] == 'SDP_D_DP_BN_MUL_SRC_VALUE':
                    self.bn_reg_mul_operand = asm_line['NVDLA_SDP_DP_BN_MUL_SRC_VALUE']
                if asm_line['name'] == 'SDP_D_DP_EW_ALU_SRC_VALUE':
                    self.ew_reg_alu_operand = asm_line['NVDLA_SDP_DP_EW_ALU_SRC_VALUE']
                if asm_line['name'] == 'SDP_D_DP_EW_ALU_SRC_VALUE':
                    self.ew_reg_mul_operand = asm_line['NVDLA_SDP_DP_EW_MUL_SRC_VALUE']

                if asm_line['name'] == 'SDP_D_DATA_CUBE_WIDTH':
                    self.inp1_shape[0] = asm_line['NVDLA_SDP_WIDTH']
                if asm_line['name'] == 'SDP_D_DATA_CUBE_HEIGHT':
                    self.inp1_shape[1] = asm_line['NVDLA_SDP_HEIGHT']
                if asm_line['name'] == 'SDP_D_DATA_CUBE_CHANNEL':
                    self.inp1_shape[2] = asm_line['NVDLA_SDP_CHANNEL']
            elif asm_line['name'] == 'VirMemWr':
                # max size of main input is 128000 2 byte atoms
                if int(asm_line['addr'], 16) < 128000 * 2:
                    # take off 0x and append all together so end ---> start, each entry is 4 hex dig as int16
                    self.inp1_data = asm_line['data'][2:] + self.inp1_data
                # operand
                else:
                    self.inp2_data = asm_line['data'][2:] + self.inp2_data
            elif asm_line['name'] == 'VirMemRd':
                continue
            elif asm_line['name'] in dp_instrs:
                # every w*h (16 int per atom, 16 per dp call)
                collect_dma_ops_flag = len(self.dp_indices) % (
                    self.inp1_shape[0] * self.inp1_shape[1]) == 0

                self.dp_indices.append(len(self.prog_frag))

                instr = {"instr No.": len(self.prog_frag)}
                for channel in ['cacc_data', 'mrdma_data', 'regs_data_alu', 'regs_data_mult', 'dma_data_alu', 'dma_data_mult']:
                    for channel_idx in range(16):
                        instr[f'{channel}_{channel_idx}'] = 0  # set default

                        # always override for main data (sdp_driver always uses mrdma_data currently)
                        if channel == 'mrdma_data' and len(self.inp1_data) > 0:
                            if nb_vals_in_layer_seen >= self.inp1_shape[0] * self.inp1_shape[1] * 32:
                                nb_vals_in_layer_seen = 0
                            num_str = self.inp1_data[-4:]
                            num_np = np.frombuffer(bytes.fromhex(
                                num_str), dtype=np.int16, count=1)[0]
                            self.inp1_data = self.inp1_data[:-4]
                            # convert np int16/8 to python int so easy to make JSON
                            instr[f'{channel}_{channel_idx}'] = int(num_np)

                            nb_vals_in_layer_seen += 1

                        # override param value if datapath op is per channel or per elem
                        # Note order is important - batchnorm packed [(alu,mul), (alu,mul) ...]
                        if channel == 'dma_data_alu' and (self.elemwise_op or self.op_name[:7] == 'channel'):
                            if self.elemwise_op and (self.bs_alu_src == 1 or self.bn_alu_src == 1 or self.ew_alu_src == 1):
                                num_str = self.inp2_data[-4:]
                                # convert np int16/8 to python int so easy to make JSON
                                alu_op_num = int(np.frombuffer(bytes.fromhex(
                                    num_str), dtype=np.int16, count=1)[0])
                                self.inp2_data = self.inp2_data[:-4]
                                instr[f'{channel}_{channel_idx}'] = alu_op_num
                            # per channel (1 elem per channel) + start of new layer
                            if self.op_name[:7] == 'channel':
                                if collect_dma_ops_flag and self.op_name != 'channel_batch_norm':
                                    collect_dma_ops_flag = False
                                    for op_idx_temp in range(16):
                                        num_str = self.inp2_data[-4:]
                                        if self.op_name == 'channel_mul' or self.op_name == 'channel_prelu':
                                            mul_op_nums[op_idx_temp] = int(np.frombuffer(bytes.fromhex(
                                                num_str), dtype=np.int16, count=1)[0])
                                        else:
                                            alu_op_nums[op_idx_temp] = int(np.frombuffer(bytes.fromhex(
                                                num_str), dtype=np.int16, count=1)[0])
                                        self.inp2_data = self.inp2_data[:-4]
                                elif collect_dma_ops_flag and self.op_name == 'channel_batch_norm':
                                    collect_dma_ops_flag = False
                                    for op_idx_temp in range(16):
                                        # alu first
                                        num_str = self.inp2_data[-4:]
                                        alu_op_nums[op_idx_temp] = int(np.frombuffer(bytes.fromhex(
                                            num_str), dtype=np.int16, count=1)[0])
                                        self.inp2_data = self.inp2_data[:-4]
                                        # then mul
                                        num_str = self.inp2_data[-4:]
                                        mul_op_nums[op_idx_temp] = int(np.frombuffer(bytes.fromhex(
                                            num_str), dtype=np.int16, count=1)[0])
                                        self.inp2_data = self.inp2_data[:-4]
                                alu_op_nums_idx = channel_idx
                                instr[f'{channel}_{channel_idx}'] = alu_op_nums[alu_op_nums_idx]

                        elif channel == 'dma_data_mult' and (self.elemwise_op or self.op_name[:7] == 'channel'):
                            if self.elemwise_op and (self.bs_mul_src == 1 or self.bn_mul_src == 1 or self.ew_mul_src == 1):
                                num_str = self.inp2_data[-4:]
                                # convert np int16/8 to python int so easy to make JSON
                                mul_op_num = int(np.frombuffer(bytes.fromhex(
                                    num_str), dtype=np.int16, count=1)[0])
                                self.inp2_data = self.inp2_data[:-4]
                                instr[f'{channel}_{channel_idx}'] = mul_op_num
                            # per channel (1 elem per channel) + start of new layer
                            if self.op_name[:7] == 'channel':
                                if collect_dma_ops_flag and self.op_name != 'channel_batch_norm':
                                    collect_dma_ops_flag = False
                                    for op_idx_temp in range(16):
                                        num_str = self.inp2_data[-4:]
                                        mul_op_nums[op_idx_temp] = int(np.frombuffer(bytes.fromhex(
                                            num_str), dtype=np.int16, count=1)[0])
                                        self.inp2_data = self.inp2_data[:-4]
                                mul_op_nums_idx = channel_idx
                                # + 8 * ((len(self.dp_indices)-1) % 2)
                                instr[f'{channel}_{channel_idx}'] = mul_op_nums[mul_op_nums_idx]

                        # override operand value if has been set in a register
                        if channel == 'regs_data_alu':
                            if self.bs_alu_src == 0 and self.bs_reg_alu_operand != None:
                                instr[f'{channel}_{channel_idx}'] = self.bs_reg_alu_operand
                            if self.bn_alu_src == 0 and self.bn_reg_alu_operand != None:
                                instr[f'{channel}_{channel_idx}'] = self.bn_reg_alu_operand
                            if self.ew_alu_src == 0 and self.ew_reg_alu_operand != None:
                                instr[f'{channel}_{channel_idx}'] = self.ew_reg_alu_operand
                        if channel == 'regs_data_mult':
                            if self.bs_mul_src == 0 and self.bs_reg_mul_operand != None:
                                instr[f'{channel}_{channel_idx}'] = self.bs_reg_mul_operand
                            if self.bn_mul_src == 0 and self.bn_reg_mul_operand != None:
                                instr[f'{channel}_{channel_idx}'] = self.bn_reg_mul_operand
                            if self.ew_mul_src == 0 and self.ew_reg_mul_operand != None:
                                instr[f'{channel}_{channel_idx}'] = self.ew_reg_mul_operand

                instr['csb_addr'] = '0x000'
                instr['csb_data'] = 0
                instr['csb_write'] = 0
                instr['csb_vld'] = 0
                instr['csb_rdy'] = 0
                instr['fifo_clr'] = 0
                instr['done'] = 0
                self.prog_frag.append(instr)
            elif asm_line['name'] == 'DONE':
                # process done
                instr = {"instr No.": len(self.prog_frag)}
                for channel in ['cacc_data', 'mrdma_data', 'regs_data_alu', 'regs_data_mult', 'dma_data_alu', 'dma_data_mult']:
                    for channel_idx in range(16):
                        instr[f'{channel}_{channel_idx}'] = 0
                instr['csb_addr'] = '0x000'
                instr['csb_data'] = 0
                instr['csb_write'] = 0
                instr['csb_vld'] = 0
                instr['csb_rdy'] = 0
                instr['fifo_clr'] = 0
                instr['done'] = 1
                self.prog_frag.append(instr)
            else:
                raise Exception(
                    "Driver generated an ILA instuction that doesn't exist")

    def dump_ila_prog_frag(self, out_path):
        """
        convert and dump program fragment
        """
        if self.prog_frag == None:
            self.to_ila_prog_frag()
        with open(out_path, 'w') as fout:
            json.dump({'program fragment': self.prog_frag}, fout, indent=4)
