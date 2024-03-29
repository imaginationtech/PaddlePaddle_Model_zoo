# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from . import pvr_infer_pb2 as engine_dot_backend_dot_pvr__grpc_dot_generated_dot_pvr__infer__pb2


class PVRInferStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.FindService = channel.unary_unary(
                '/pvr.PVRInfer/FindService',
                request_serializer=engine_dot_backend_dot_pvr__grpc_dot_generated_dot_pvr__infer__pb2.FindServiceRequest.SerializeToString,
                response_deserializer=engine_dot_backend_dot_pvr__grpc_dot_generated_dot_pvr__infer__pb2.FindServiceResponse.FromString,
                )
        self.Inference = channel.unary_unary(
                '/pvr.PVRInfer/Inference',
                request_serializer=engine_dot_backend_dot_pvr__grpc_dot_generated_dot_pvr__infer__pb2.InferRequest.SerializeToString,
                response_deserializer=engine_dot_backend_dot_pvr__grpc_dot_generated_dot_pvr__infer__pb2.InferResponse.FromString,
                )


class PVRInferServicer(object):
    """Missing associated documentation comment in .proto file."""

    def FindService(self, request, context):
        """detect the server
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Inference(self, request, context):
        """invoke inference after server found
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_PVRInferServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'FindService': grpc.unary_unary_rpc_method_handler(
                    servicer.FindService,
                    request_deserializer=engine_dot_backend_dot_pvr__grpc_dot_generated_dot_pvr__infer__pb2.FindServiceRequest.FromString,
                    response_serializer=engine_dot_backend_dot_pvr__grpc_dot_generated_dot_pvr__infer__pb2.FindServiceResponse.SerializeToString,
            ),
            'Inference': grpc.unary_unary_rpc_method_handler(
                    servicer.Inference,
                    request_deserializer=engine_dot_backend_dot_pvr__grpc_dot_generated_dot_pvr__infer__pb2.InferRequest.FromString,
                    response_serializer=engine_dot_backend_dot_pvr__grpc_dot_generated_dot_pvr__infer__pb2.InferResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'pvr.PVRInfer', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class PVRInfer(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def FindService(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/pvr.PVRInfer/FindService',
            engine_dot_backend_dot_pvr__grpc_dot_generated_dot_pvr__infer__pb2.FindServiceRequest.SerializeToString,
            engine_dot_backend_dot_pvr__grpc_dot_generated_dot_pvr__infer__pb2.FindServiceResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Inference(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/pvr.PVRInfer/Inference',
            engine_dot_backend_dot_pvr__grpc_dot_generated_dot_pvr__infer__pb2.InferRequest.SerializeToString,
            engine_dot_backend_dot_pvr__grpc_dot_generated_dot_pvr__infer__pb2.InferResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
