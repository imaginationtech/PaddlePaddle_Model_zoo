import os
import numpy as np
import grpc

from .pvr_grpc.generated import pvr_infer_pb2
from .pvr_grpc.generated import pvr_infer_pb2_grpc
from .pvr_grpc.pvr_infer_cmd import PowerVR_Infer_Cmdline

class PowerVR_Infer(object):
    def __init__(self, pvr_config):
        self.vm = pvr_config['base_name']
        self.input_name = pvr_config['input_name']
        self.batch_size = pvr_config['batch_size']
        #self.out_file_name = pvr_config['output_file_name']
        self.out_shape = pvr_config['output_shape']

        self.pvr_infer = PowerVR_Infer_Cmdline(pvr_config)

    def __call__(self, x):
        inputs = dict()
        inputs[self.input_name] = x
        outputs = self.pvr_infer(inputs)
        out = outputout[0].astype(np.float32)
        out.shape = [self.batch_size] + self.out_shape
        return out


class PowerVR_Infer_gRPC(object):
    def __init__(self, pvr_grpc_config):
        self.input_name = pvr_grpc_config['input_name']
        self.batch_size = pvr_grpc_config['batch_size']
        self.out_shape = pvr_grpc_config['output_shape']

        server = pvr_grpc_config['pvr_server']
        server += ":50051"
        #with grpc.insecure_channel(server) as channel:
        channel =  grpc.insecure_channel(server)
        self.stub = pvr_infer_pb2_grpc.PVRInferStub(channel)
 
    def __call__(self, x):
        request = pvr_infer_pb2.InferRequest()
        request.input.name = self.input_name
        request.input.data = x.tobytes()

        response = self.stub.Inference(request)
        outputs = list()
        for out in response.outputs:
            data = np.frombuffer(out.data, dtype=np.float32)
            outputs.append(data)

        out = outputs[0]
        out.shape = [self.batch_size] + self.out_shape
        return out
