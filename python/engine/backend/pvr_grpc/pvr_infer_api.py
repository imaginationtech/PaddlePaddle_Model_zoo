#import powervr_runtime as pvr_rt
class PowerVR_RT(object):
    def __init__(self, base_name):
        print("init vm")

    def set_input(name, data):
        print("set input tensor to the named input")

    def invoke(func_name):
        # return list of output tensors
        print("run the named func, default is 'main'")

class PowerVRInference_2(object):
    def __init(self, pvr_config):
        self.vm = pvr_config['base_name']
        self.input_name = pvr_config['input_name']
        self.batch_size = pvr_config['batch_size']
        self.out_shape = pvr_config['output_shape']

        self.pvr_rt = PowerVR_RT(self.vm)

    def __call__(self, x):
        self.pvr_rt.set_input(self.input_name, x)
        outputs = self.pvr_rt.invoke()

