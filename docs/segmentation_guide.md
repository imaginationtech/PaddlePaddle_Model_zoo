# Deployment and Test on models in PaddleSeg 

## Introduction
We start Deployment and Test from [HRNet_Contrast_W48](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/configs/hrnet_w48_contrast), a semantic segmentaion model in [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg).

## Running evaluation and inference

### Preparation
Setup development host and deployment target device according to NCSDK documents. Make sure basic IMGDNN test pass and tutorial example could be deployed successfully.

#### Development Host
1. Install PaddlePaddle
2. Compile PaddlePaddle prediction model into PowerVR deployment packages.
3. Clone this repository.
4. Download cityspace evaluate dataset from sftp server to EvalDatasetPath directory. The structure of the evaluation dataset is as follows:
```
    cityscapes
    |
    |--leftImg8bit
    |  |--val
    |
    |--gtFine
    |  |--val
```

#### Target device
1. Copy python/engine/backend/pvr_grpc/* to GRPCServerPath directory on target devices;  
2. Copy PowerVR deployment package to DeploymentModelPath;  
3. Set base_name field in $GRPCServerPath/pvr_service_config.yml, e.g

|field|description|values|
|:---:|:---------:|:-----:|
|base_name|path to vm file|$DeploymentModelPath/HRNet_Contrast_W48-AX2185-d16b16w16-ncsdk_2_8-arrch64_linux_gnu.ro|
4. Launch the gRPC server
```
python PVRInferServer.py
```

### Running Test
#### Config File
Create a config file in Yaml, some fields are described below. Also refer to configs/image_segmentation/HRNet_Contrast_W48.yaml

|field|description|values|
|:---:|:---------:|:-----:|
|Global.mode|test mode|evaluation,inference|
|Dataloader.dataset.name|dataset class name to be instantialize|CityScapesDataset|
|Dataloader.dataset.image_root|directory of input mages of CityScapes dataset|string|
|Dataloader.dataset.label_path|directory of ground truth of CityScapes dataset|string|
|Dataloader.sampler.batch_size|set the batch size of inupt|integer|
|Infer.infer_imgs|path to test image or directory|string|
|Infer.batch_size|batch size of inference|interger|
|Infer.PostProcess.save_path|path to inference results of test image or directory|string|
|Metric.Eval.mIou.num_classes|number of semantic categories|interger|


##### Config PaddlePaddle backend
To inference with Paddle backend, the [Pretrained Model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/HRNet_W48_contrast_cityscapes_1024x512_60k/model.pdparams) need to be downloaded and [exported](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/docs/model_export.md) to a prediction model before deployment. Besides, fields below need to be set.

|field|description|values|
|:---:|:---------:|:-----:|
|Model.backend|backend used to inference|paddle|
|Model.Paddle.path|path to Paddle model|string|
|Model.Paddle.base_name|base name of model files|string|

##### Config PowerVR standalone backend
To run all test code on target device, set the fields below

|field|description|values|
|:---:|:---------:|:-----:|
|Model.backend|backend used to inference|powervr|
|Model.PowerVR.base_name|path to vm file|string|
|Model.PowerVR.input_name|network input name|string|
|Model.PowerVR.output_shape|shape of output tensor|list|

##### Config PowerVR Distributed backend
To run test code on host and run inference on target device, set the fields below

|field|description|values|
|:---:|:---------:|:-----:|
|Model.backend|backend used to inference|powervr_grpc|
|Model.PowerVR_gRPC.pvr_server|IP of gRPC server|IP address|
|Model.PowerVR.input_name|network input name|string|
|Model.PowerVR.output_shape|shape of output tensor|list|

#### Running test scripts
```
python tools/test_egret.py -c ./configs/image_segmentation/HRNet_Contrast_W48.yaml
```
Some field could be override at command line, e.g. to override the batch_size
```
python tools/test_egret.py -c ./configs/image_segmentation/HRNet_Contrast_W48.yaml \
-o DataLoader.Eval.sampler.batch_size=1
```

## Performance
### Image Sematic Segmentation
| Model | mIoU (with 5 images) | time(ms)<br>bs=1 | time(ms)<br>bs=4 | Download<br>Address |
|:----:|:----:|:----:|:----:|:----|
|HRNet_Contrast_W48<br>(d16-w16-b16)|0.593|null|null|sftp://transfer.imgtec.com/paddle_models/paddle_segmentation/HRNet_Contrast_W48-AX2185-d16b16w16-ncsdk_2_8-arrch64_linux_gnu.ro|
|HRNet_Contrast_W48<br>(non-quant)|0.593|null|null|[link](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/HRNet_W48_contrast_cityscapes_1024x512_60k/model.pdparams)|



