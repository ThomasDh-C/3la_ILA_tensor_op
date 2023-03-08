import json
import os
from collections import defaultdict
import numpy as np


class PDPConverter:
    def __init__(self, asm_path, op_name):
        # load asm and data_lib from files
        with open(asm_path, 'r') as fin:
            self.asm_list = json.load(fin)['asm']
        self.op_name = op_name
        self.prog_frag = None
        self.sim_to_out_cube = defaultdict(list)

        path = os.path.dirname(os.path.abspath(__file__))
        with open(path + '/wrong_all_param_shifts_for_csb.json', 'r') as fin:
            self.param_shifts_for_csb = json.load(fin)

    def to_ila_prog_frag(self):
        """
        convert ILA assembly to program fragment if not done already
        """
        dp_instrs = {'CMAC_CACHE_WEIGHTS', 'CMAC_COMPUTE_DOT_PRODUCT'}
        nb_vals_in_layer_seen = 0  # used for per channel
        alu_op_num, mul_op_num = 0, 0
        alu_op_nums, mul_op_nums = [0 for i in range(32)], [
            0 for i in range(32)]
        self.prog_frag = []
        for asm_line in self.asm_list:
            if asm_line['name'] in self.param_shifts_for_csb:
                # append to prog frag based on that and find inp_addr_end
                instr = {"instr No.": len(self.prog_frag)}

                name_conversion_dict = self.param_shifts_for_csb[asm_line['name']]
                # simulator only looks at lowest 12 bits so can put real address
                instr['cmac_csb2cmac_addr'] = name_conversion_dict['addr']
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
                instr['cmac_csb2cmac_data'] = int(data, 2)
                instr['cmac_csb2cmac_write'] = 1
                instr['cmac_csb2cmac_vld'] = 1
                self.prog_frag.append(instr)
            elif asm_line['name'] == 'VirMemWr':
                # ignore for moment
                continue
            elif asm_line['name'] == 'VirMemRd':
                # ignore for the moment
                continue
            elif asm_line['name'] in dp_instrs:
                curr_instr_nb = len(self.prog_frag)
                instr = {"instr No.": curr_instr_nb}
                pos_in_out_cube = asm_line['out_pos']
                if pos_in_out_cube != 'padding':
                    self.sim_to_out_cube[pos_in_out_cube].append(curr_instr_nb)
                instr['cmac_csc2cmac_vld'] = True
                instr['cmac_csc2cmac_sending_last_batch'] = False
                instr['cmac_csc2cmac_reuse_weights'] = True
                if asm_line['name'] == 'CMAC_CACHE_WEIGHTS':
                    instr['cmac_csc2cmac_reuse_weights'] = False
                for key, val in asm_line.items():
                    if key not in {'name', 'out_pos'}:
                        instr[key] = val
                self.prog_frag.append(instr)
            elif asm_line['name'] == 'DONE':
                # process done
                instr = {"instr No.": len(self.prog_frag)}
                instr['cmac_csc2cmac_sending_last_batch'] = True
                self.prog_frag.append(instr)
            else:
                raise Exception(
                    f"Driver generated an ILA instuction '{asm_line['name']}' that doesn't exist")

    def dump_ila_prog_frag(self, out_path):
        """
        convert and dump program fragment
        """
        if self.prog_frag == None:
            self.to_ila_prog_frag()
        with open(out_path, 'w') as fout:
            json.dump({'program fragment': self.prog_frag}, fout, indent=4)
