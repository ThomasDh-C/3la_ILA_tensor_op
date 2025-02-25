import json
import sys
import numpy as np
import subprocess
import os

from src.utils import tool
from src.converter import Converter as cvtr

class layernorm_driver:
  def __init__(self, num_v, num_ts):
    """
    for layernorm, num_v_in == num_v_out
    """
    self.num_v = num_v
    self.num_ts = num_ts
    self.tl = tool()

  def produce_asm(self):
    ila_asm = []
    ila_asm.append({
      'name' : 'store_beta',
      'idx' : 'beta'
    })
    ila_asm.append({
      'name' : 'store_gamma',
      'idx' : 'gamma'
    })
    for i in range(self.num_ts):
      ila_asm.append({
        'name' : 'store_act',
        'timestep_idx' : 'ts_' + str(i),
        'idx' : i
      })  
    ila_asm.append({
      'name' : 'layernorm',
      'num_ts' : self.num_ts
    })
    ila_asm.append({
      'name' : 'wait_irq'
    })
    for i in range(self.num_ts):
      ila_asm.append({
        'name' : 'load_act',
        'mem_idx' : 0,
        'ts_idx' : i
      })
  
    ila_asm = {'asm' : ila_asm}
    with open('./test/layernorm_asm.json', 'w') as fout:
      json.dump(ila_asm, fout, indent=4)
    print('*** ILA tensor assmbly has been dumped to ./test/layernorm_asm.json ***')
  
  def produce_data_lib(self):
    """
    produce data_lib for layernorm
    """
    inp_q, bias_inp = self.tl.get_adpfloat_bias(self.inp)
    beta_q, bias_beta = self.tl.get_adpfloat_bias(self.beta)
    gamma_q, bias_gamma = self.tl.get_adpfloat_bias(self.gamma)
    
    self.adpbias_inp = int(bias_inp + 10)
    self.adpbias_beta = int(bias_beta + 10)
    self.adpbias_gamma = int(bias_gamma + 10)

       
    param = {
      'gb_num_vector_in' : self.num_v,
      'gb_num_vector_out' : self.num_v,
      'adpbias_inp' : self.adpbias_inp,
      'adpbias_beta' : self.adpbias_beta,
      'adpbias_gamma' : self.adpbias_gamma
    }

    print('\n--------------------------------------------------------------')
    print('\tinvoking float to adpfloat converter')
    print('--------------------------------------------------------------\n')
    inp_q.tofile('./test/inp_q.tmp', sep = '\n')
    self.tl.call_float_adpt_v_cvtr('./test/inp_q.tmp', self.adpbias_inp, './test/inp_q_av.tmp')
    beta_q.tofile('./test/beta_q.tmp', sep = '\n')
    self.tl.call_float_adpt_v_cvtr('./test/beta_q.tmp', self.adpbias_beta, './test/beta_q_av.tmp')
    gamma_q.tofile('./test/gamma_q.tmp', sep = '\n')
    self.tl.call_float_adpt_v_cvtr('./test/gamma_q.tmp', self.adpbias_gamma, './test/gamma_q_av.tmp')

    self.gen_data_lib_helper(param, './test/layernorm_data_lib.json')
    """
    store the quantized values for reference calculation
    """
    self.inp = inp_q
    self.beta = beta_q
    self.gamma = gamma_q

  def gen_data_lib_helper(self, param, out_path):
    """
    produce data lib
    """
    self.data_lib = param
    # put adpfloat type input data into data-lib
    with open('./test/inp_q_av.tmp', 'r') as fin:
      inp_v_list = fin.read().splitlines()
    assert len(inp_v_list) == self.num_v * self.num_ts
    for t in range(self.num_ts):
      self.data_lib = \
        self.tl.vector_to_data_lib(inp_v_list[t*self.num_v : (t+1)*self.num_v], 
                                   'ts_{}'.format(t), self.num_v, self.data_lib)
    # put adpfloat type beta data into data_lib
    with open('./test/beta_q_av.tmp', 'r') as fin:
      beta_v_list = fin.read().splitlines()
    assert len(beta_v_list) == self.num_v
    self.data_lib = self.tl.vector_to_data_lib(beta_v_list, 'beta',
                                               self.num_v, self.data_lib)
    # put adpfloat type gamma data into data_lib
    with open('./test/gamma_q_av.tmp', 'r') as fin:
      gamma_v_list = fin.read().splitlines()
    assert len(gamma_v_list) == self.num_v
    self.data_lib = self.tl.vector_to_data_lib(gamma_v_list, 'gamma',
                                               self.num_v, self.data_lib)
    with open(out_path, 'w') as fout:
      json.dump(self.data_lib, fout, indent=4)
    print('\n*** data_lib has been dumped to {} ***\n'.format(out_path))
  
  # --------------------------------
  # invoke ILA simulation
  # --------------------------------
  def gen_prog_frag(self):
    print('\n--------------------------------------------------------------')
    print('\tgenerate prog_frag.json for ILA simulator')
    print('--------------------------------------------------------------\n')
    self.ila_cvtr = cvtr('./test/layernorm_asm.json', './test/layernorm_data_lib.json')
    self.ila_cvtr.dump_ila_asm('./test/layernorm_ila_asm.json')
    self.ila_cvtr.dump_ila_prog_frag('./test/layernorm_prog_frag_in.json')
    print('*** ILA program fragment has been dumped to ./test/layernorm_prog_frag_in.json***\n')
  
  def collect_ila_result(self):
    """
    run ila simulation and collect the result
    """
    self.result_ila = self.tl.collect_ila_result(in_path='./test/layernorm_prog_frag_in.json',
                      mem_idx=0, num_ts=self.num_ts, 
                      num_vi=self.num_v, num_vo=self.num_v, bias=self.adpbias_inp)

  # --------------------------------------
  # dump axi commands
  # --------------------------------------
  def gen_axi_cmds(self, base_addr = '0x33000000'):
    print('\n--------------------------------------------------------------')
    print('\tgenerate axi commands for FlexNLP')
    print('--------------------------------------------------------------\n')
    if not self.ila_cvtr:
      self.ila_cvtr = cvtr('./test/layernorm_asm.json', './test/layernorm_data_lib.json')
    self.ila_cvtr.dump_axi_cmds('./test/layernorm_axi_cmd.csv', base_addr)
    print('*** axi commands has been dumped to ./test/layernorm_axi_cmd.csv ***')
  
  def run_test(self, verbose_analysis=0):
    subprocess.run(['mkdir', '-p', 'npy', 'test', 'data'])
    self.produce_asm()
    self.produce_random_test_data()
    self.produce_data_lib()
    self.gen_prog_frag()
    self.collect_ila_result()
    self.gen_axi_cmds('0xA0000000')
    self.produce_ref_result()
    err_out_list = self.result_analysis(verbose_analysis)
    return err_out_list


  ##########################################
  ##  Test tool
  ##########################################
  # ----------------------------------------
  # produce layernorm input data
  # ----------------------------------------
  def produce_random_test_data(self):
    """
    produce random data for testing
    """
    print('\n--------------------------------------------------------------')
    print('\tproducing random input data')
    print('--------------------------------------------------------------\n')
    coef = 1
    # because FlexNLP use the same adpbias of input for the output, thus the
    # value range for output is limited! gamma & beta has to be small
    self.inp = \
      coef * np.random.uniform(-1, 1, (self.num_ts * 16 * self.num_v)).astype(np.float32)
    self.beta = \
      0.2 * np.random.uniform(-1, 1, (self.num_v * 16)).astype(np.float32)
    self.gamma = \
      0.2 * np.random.uniform(-1, 1, (self.num_v * 16)).astype(np.float32)
  
  def produce_ref_result(self):
    """
    produce relay reference result
    """
    ref = self.tl.get_relay_layernorm_ref(self.num_v, self.inp, self.beta, self.gamma)
    self.ref_out = ref.reshape(self.num_ts, -1)

  def result_analysis(self, is_verbose):
    print('\n--------------------------------------------------------------')
    print('\tanalyze ILA simulation result')
    print('--------------------------------------------------------------\n')
    err_out_list = []
    for i in range(self.num_ts):
      if not os.environ.get('USE_3LA_FPGA'):
        result_ts = self.result_ila[self.num_v*16*i : self.num_v*16*(i+1)]
      ref = self.tl.get_adpfloat_bias(self.ref_out[i])[0]
      err_out, err_ref = self.tl.cal_error(result_ts, ref)
      print("result timestep No.{} --- relative error (vs. sim_out): {:5.5%}\
            relative error (vs. ref): {:5.5%}\n".format(i, err_out, err_ref))
      if is_verbose:
        print("reference output: \n{}\nresult: \n{}\n".format(ref, result_ts))
      err_out_list.append(err_out)
    return err_out_list

  def clean_up(self):
    for file in os.listdir('./test'):
      if '.tmp' in file:
        subprocess.run(['rm', './test/'+file])

if __name__ == '__main__':
  assert len(sys.argv) == 3, \
    "Usage: python3 layernorm [num_vector_in] [num_timestep]"
  num_v = int(sys.argv[1])
  num_ts = int(sys.argv[2])
  test_driver = layernorm_driver(num_v, num_ts)
  test_driver.run_test(verbose_analysis=0)
  test_driver.clean_up()