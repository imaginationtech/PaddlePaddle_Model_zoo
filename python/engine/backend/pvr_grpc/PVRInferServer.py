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

"""The Python implementation of the GRPC pvr.PVRInfer server."""

from concurrent import futures
import logging
import yaml

import grpc
import generated.pvr_infer_pb2 as pvr_infer_pb2
import generated.pvr_infer_pb2_grpc as pvr_infer_pb2_grpc
from pvr_infer_cmd import PowerVR_Infer_Cmdline

import numpy as np

MAX_MESSAGE_LENGTH = 32*1024*1024

class PVRInfer(pvr_infer_pb2_grpc.PVRInferServicer):

    def __init__(self, pvr_service_config):
        self.pvr_infer = PowerVR_Infer_Cmdline(pvr_service_config)

    def FindService(self, request, context):
        print("Receive service request with name {}".format(request.service_name))
        response = pvr_infer_pb2.FindServiceResponse()
        response.service_ack = request.service_name + "_ready"
        return response

    def Inference(self, request, context):
        print("input name:{}".format(request.input.name))
        infer_in = dict()
        infer_in[request.input.name] = np.frombuffer(request.input.data)

        infer_out = self.pvr_infer(infer_in)

        response = pvr_infer_pb2.InferResponse()
        out = response.outputs.add()
        out.name = "output_0"
        out.data = infer_out[0].tobytes()
        return response

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=[
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH), 
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH)
            ])

    # get config
    f = open('pvr_service_config.yml', 'r')
    pvr_config = yaml.load(f, Loader=yaml.SafeLoader)

    # create PVRInfer with config
    pvr_service = PVRInfer(pvr_config)

    # add service to server
    pvr_infer_pb2_grpc.add_PVRInferServicer_to_server(pvr_service, server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    serve()
