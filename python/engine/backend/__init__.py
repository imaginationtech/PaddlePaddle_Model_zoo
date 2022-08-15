from utils import logger

def build_inference(config):
    backend_type = config['Model']['backend']
    assert backend_type in ['powervr', 'paddle', 'clas_sim', 'powervr_grpc'
            ],"backend type should be 'clas_sim', 'powervr', 'paddle'"
    mode = config['Global']['mode']
    if mode == 'evaluation':
        batch_size = config['DataLoader']['Eval']['sampler']['batch_size']
    elif mode == 'inference':
        batch_size = config['Infer']['batch_size']
    else:
        logger.error("Invalid mode set in configuration")

    if backend_type == 'powervr':
        from .powervr import PowerVR_Infer
        powervr_config = config['Model']['PowerVR']
        powervr_config['batch_size'] = batch_size
        return PowerVR_Infer(powervr_config)
    if backend_type == 'powervr_grpc':
        from .powervr import PowerVR_Infer_gRPC
        pvr_grpc_config = config['Model']['PowerVR_gRPC']
        pvr_grpc_config['batch_size'] = batch_size
        return PowerVR_Infer_gRPC(pvr_grpc_config)
    elif backend_type == 'paddle':
        from .paddle_engine import PaddleInference
        paddle_config = config['Model']['Paddle']
        return PaddleInference(paddle_config)
    elif backend_type == 'clas_sim':
        from .simulator import SimClasRuntime
        sim_config = config['Model']['ClasSim']
        return SimClasRuntime(sim_config)
        

