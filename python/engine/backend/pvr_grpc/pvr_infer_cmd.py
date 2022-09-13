# Copyright (c) 2022 Imagination Technologies Ltd. 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np


class PowerVR_Infer_Cmdline(object):
    def __init__(self, pvr_config):
        #pdb.set_trace()
        self.vm = pvr_config['base_name']
        #self.input_name = pvr_config['input_name']
        #self.batch_size = pvr_config['batch_size']
        #self.out_file_name = pvr_config['output_file_name']
        #self.out_shape = pvr_config['output_shape']
        # del reference output files
        self.output_number = None

    def check_output_number(self):
        return 1

    def __call__(self, inputs):
        assert isinstance(inputs, dict)

        cmd = 'powervr_execute -c '
        cmd += self.vm
        cmd += ' -i '

        for i, x_name in enumerate(inputs):
            if i > 0:
                cmd += ','
            x_file_name = "data_{}.f32".format(i)
            if os.path.exists(x_file_name):
                os.system('rm ' + x_file_name)
            inputs[x_name].tofile(x_file_name)
            cmd += x_name + ':'
            cmd += x_file_name

        cmd += ' >> powervr_execute.log 2>&1'
        print(cmd)
        os.system(cmd)

        if self.output_number == None:
            self.output_number = self.check_output_number()

        outputs = list()
        for i in range(self.output_number):
            out_name = "tvm_infer_0_out_{}.bin".format(i)
            out = np.fromfile(out_name)
            outputs.append(out)

        return outputs
