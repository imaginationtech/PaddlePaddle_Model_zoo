[简体中文](README_cn.md) | English

# powervr_paddle_model

## Introduction
This is an Imagination NCSDK enabled PaddlePaddle model zoo. The source model is from PadlePaddle model zoo. Utilize Imagination NCSDK, PaddlePaddle models could be compiled and deployed on device embedded with Imagination computing IPs, such as NNA and GPU.

On the repository, we will get models and test program to evaluate the inference performance.

## System Overview
This repository could be used as an Evaluation/Inference framework based on different inference backend.

<div align=center>
![Local](./docs/images/local_infer.png)  
<p align="center">Fig.1 Local inference</p>
</div>

<img src="https://gitee.com/jiansowa/powervr_paddle_model/docs/images/local_infer.png" width="614" height="428")  

<p align="center">Fig.1 Local inference</p>

![gRPC](./docs/images/grpc_infer.png)  
<center>Fig.2 Remote inference</center>

### NCSDK TVM runtime backend
There are two way to run evaluation on Imagination IPs.
1. Local inference
   All code is running on ROC1, see Fig.1
2. Remote inference
   The runtime system is a distributed system. Dataloader, preprocess, postprocess run on Host. ROC1 only run the NN inference. See Fig.2

### PaddlePaddle inference backend
Non-quantized performance data is obtained by running evaluation with exported Paddle inference model on Machine with PaddlePaddle installed.




## Imagination SW and HW
HW: Unisoc ROC1  
SW: NCSDK 2.8, NNA DDK 3.4, GPU DDK 1.17  

## Download models
sftp server: transfer.imgtec.com  
http client: https://transfer.imgtec.com  
user_name:imgchinalab_public  
password: public  


## Running evaluation and inference
### Setup
#### Inference board
Install NNA, GPU DDK  
Install NCSDK TVM Runtime  
Configure and run innference server  
The compiled model  

#### Evaluation machine
Clone this repo  

### Evaluatin
Set mode to 'Eval'  
Run run_test_egret.sh  

### Inference
Set mode to 'Infer'  
Run run_test_egret.sh  

## Performance
### Image Classification



| Model | top-1 | top-5 | time(ms)<br>bs=1 | time(ms)<br>bs=4 | Download<br>Adress |
|:----:|:----:|:----:|:----:|:----:|:----:|
|EfficientNetB0<br>(d16-w16-b16)|75.4|93.2|null|null|[link](sftp://transfer.imgtec.com/paddle_models/EfficientNetB0-AX2185-d16w16b16-ncsdk-2_8_deploy.tar.bz2)|
|EfficientNetB0<br>(non-quant)|75.9|93.7|null|null|[link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/EfficientNetB0_infer.tar)|
|EfficientNetB0<br>(d8-w8-b16)|null|null|null|null|[link](sftp://transfer.imgtec.com/paddle_models/EfficientNetB0-AX2185-d8w8b16-ncsdk-2_8_deploy.tar.bz2)|

## Contribution
Contributions are highly welcomed and we would really appreciate your feedback!!

