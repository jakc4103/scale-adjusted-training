# scale-adjusted-training

PyTorch implementation of [Towards Efficient Training for Neural Network Quantization](https://arxiv.org/abs/1912.10207)

## Introduction
This repo implement the Scale-Adjusted Training from [Towards Efficient Training for Neural Network Quantization](https://arxiv.org/abs/1912.10207) including:
1.  Constant rescaling Dorefa-quantize
2.  Calibrated gradient PACT

## TODO
- [x] constant rescaling DoReFaQuantize layer
- [x] CGPACT layer
- [ ] test with mobilenetv1
- [ ] test with mobilenetv2
- [ ] test with resnet50

## Acknowledgement
  - https://github.com/marvis/pytorch-mobilenet  
  - https://github.com/tonylins/pytorch-mobilenet-v2
  - https://github.com/ricky40403/PyTransformer/tree/hotfix