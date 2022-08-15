from __future__ import print_function

import logging

import grpc
from generated import pvr_infer_pb2
from generated import pvr_infer_pb2_grpc

import numpy as np
from io import BytesIO

def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    #with grpc.insecure_channel('localhost:50051') as channel:
    with grpc.insecure_channel('10.80.49.42:50051') as channel:
        stub = pvr_infer_pb2_grpc.PVRInferStub(channel)

        request = pvr_infer_pb2.FindServiceRequest()
        request.service_name = "ImageClassification"
        response = stub.FindService(request)
        print("PVRInfer client received: " + response.service_ack)


        rng = np.random.default_rng(12345)
        #x = rng.integers(low=0, high=255, size=(3,224,224), dtype=np.uint8, endpoint=False)
        x = rng.random(size=(3,224,224), dtype=np.float32)
        request = pvr_infer_pb2.InferRequest()
        request.input.name = 'x'
        request.input.data = x.tobytes()

        response = stub.Inference(request)
        for out in response.outputs:
            print("output name:{}".format(out.name))
            data = np.frombuffer(out.data, dtype=np.float32)
            print(data.shape)
        #print("Greeter client received: " + response)

if __name__ == '__main__':
    logging.basicConfig()
    run()
