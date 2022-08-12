#!/bin/bash
# run at PaddleClas

#ImageNetValRoot=/home/jiansowa/data/imagenet_aiia/valset
#ImageNetValLabel=/home/jiansowa/data/imagenet_aiia/valset/val.txt
ImageNetValRoot=/home/jasonwang/data/imagenet_aiia/valset
ImageNetValLabel=/home/jasonwang/data/imagenet_aiia/valset/val.txt


Model=EfficientNetB0
#Model=MobileNetV1_ssld
#Model=MobileNetV2_ssld
#Model=MobileNetV3_large_ssld
#Model=MobileNetV3_small_ssld
#Model=ResNet50_vd
#Model=HRNet_W48_C_ssld

## 'eval': network(in framework), pretrain parameter; 'infer': inference model
Mode='run'	
#Mode='debug'

if [ $Model == 'EfficientNetB0' ]; then
  if [ $Mode = 'debug' ]
  then
      printf "Debug mode"
      python -m pdb tools/test_egret.py \
      -c ./configs/image_classification/EfficientNetB0.yaml \
      -o DataLoader.Eval.dataset.data_root=${ImageNetValRoot}	\
      -o DataLoader.Eval.dataset.label_path=${ImageNetValLabel} \
      -o DataLoader.Eval.sampler.batch_size=1
  else
      printf "Run mode"
      python tools/test_egret.py \
      -c ./configs/image_classification/EfficientNetB0.yaml \
      -o DataLoader.Eval.dataset.data_root=${ImageNetValRoot}	\
      -o DataLoader.Eval.dataset.label_path=${ImageNetValLabel} \
      -o DataLoader.Eval.sampler.batch_size=1

      #-o Global.mode=	\
  fi
fi
