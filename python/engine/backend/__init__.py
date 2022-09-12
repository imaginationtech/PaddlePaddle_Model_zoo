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
        

