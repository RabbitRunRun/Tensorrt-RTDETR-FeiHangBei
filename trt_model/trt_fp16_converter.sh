#!/bin/bash

/workingspace/tensorrt/TensorRT-8.6.1.6/bin/trtexec  --onnx=../onnx_model/rtdetr-l_op17.onnx  --workspace=4096  --saveEngine=rtdetr-l_fp16.engine  --fp16 2>&1 | tee fp16.log



