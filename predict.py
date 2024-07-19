#!/usr/bin/env python
# encoding: utf-8

# --------------------------------------------------------
# file: predict.py
# Copyright(c) 2017-2022 SeetaTech
# Written by Zhuang Liu
# 2024/06/29 14:05
# --------------------------------------------------------
import os
import os.path as osp
import argparse
import time
import torch

parser = argparse.ArgumentParser(description='predict')
parser.add_argument('--model_type', type=str, help='model choice, yolov10s or rtdetr', default='rtdetr')
parser.add_argument('--batch_size', type=int, help='batch size', default=16)
parser.add_argument('--imgsz', type=int, help='imgsz', default=1024)
parser.add_argument('--thresh', type=float, help='thresh', default=0.01)
parser.add_argument('--iou_thresh', type=float, help='iou_thresh', default=0.1)
parser.add_argument('--max_det', type=int, help='max_det', default=20)
parser.add_argument('--agnostic_nms', type=bool, help='class-agnostic Non-Maximum Suppression (NMS)', default=True)
parser.add_argument('--verbose', type=bool, help='show predict info', default=True)

args = parser.parse_args()

start_time_import = time.time()
print(args.model_type)

from ultralytics import RTDETR as YOLO


def predict (model_path, test_dir, out_dir, args):
    thresh = args.thresh
    iou_thresh = args.iou_thresh
    max_det = args.max_det
    agnostic_nms = args.agnostic_nms
    imgsz = args.imgsz
    verbose = args.verbose

    # init model
    model = YOLO(model_path)
    # print(model.task_map)
    # model = model.task_map["detect"]["predictor"]
    # print(model)
    model.training = False

    # get test list
    test_images = os.listdir(test_dir)
    batchsize = args.batch_size
    m = len(test_images) // batchsize

    for n in range(m):
        start = n * batchsize
        end = start + batchsize
        sub_test_images = test_images[start:end]
        # inference

        results = model.predict([osp.join(test_dir, test_image) for test_image in sub_test_images],
                        imgsz=imgsz,
                        conf=thresh)

        # onnx format
        # results = model([osp.join(test_dir, test_image) for test_image in sub_test_images], conf=0.01)

        # save results
        for i, result in enumerate(results):
            boxes = result.boxes
            cls = boxes.cls
            conf = boxes.conf
            xyxy = boxes.xyxy
            # save output
            out_txt_path = osp.join(out_dir, test_images[start + i][:-4] + '.txt')
            with open(out_txt_path, 'w') as fwriter:
                for j in range(len(cls)):
                    fwriter.writelines(
                        f'{int(j + 1)} {int(cls[j])} {conf[j]} {xyxy[j][0]} {xyxy[j][1]} {xyxy[j][2]} {xyxy[j][1]} {xyxy[j][2]} {xyxy[j][3]} {xyxy[j][0]} {xyxy[j][3]} {(xyxy[j][0] + xyxy[j][2]) / 2.} {(xyxy[j][1] + xyxy[j][3]) / 2.}\n')
                    print(f'{int(j + 1)} {int(cls[j])} {conf[j]} {xyxy[j][0]} {xyxy[j][1]} {xyxy[j][2]} {xyxy[j][1]} {xyxy[j][2]} {xyxy[j][3]} {xyxy[j][0]} {xyxy[j][3]} {(xyxy[j][0] + xyxy[j][2]) / 2.} {(xyxy[j][1] + xyxy[j][3]) / 2.}\n')
            
            # save show
            # out_img_path = osp.join(out_dir, test_images[start + i])
            # result.save(filename=out_img_path)

    ## remain
    start = m * batchsize
    remain_list = [osp.join(test_dir, test_images[i]) for i in range(start, len(test_images))]
    if len(remain_list) > 0:
        results = model(remain_list,
                        imgsz=imgsz,
                        conf=thresh,
                        iou=iou_thresh,
                        max_det=max_det,
                        agnostic_nms=agnostic_nms,
                        verbose=verbose,
                        half=False)

        # onnx format
        # results = model(remain_list, conf=0.01)

        # save results
        for i, result in enumerate(results):
            boxes = result.boxes
            cls = boxes.cls
            conf = boxes.conf
            xyxy = boxes.xyxy
            # save output
            out_txt_path = osp.join(out_dir, test_images[start + i][:-4] + '.txt')
            with open(out_txt_path, 'w') as fwriter:
                for j in range(len(cls)):
                    fwriter.writelines(
                        f'{int(j + 1)} {int(cls[j])} {conf[j]} {xyxy[j][0]} {xyxy[j][1]} {xyxy[j][2]} {xyxy[j][1]} {xyxy[j][2]} {xyxy[j][3]} {xyxy[j][0]} {xyxy[j][3]} {(xyxy[j][0] + xyxy[j][2]) / 2.} {(xyxy[j][1] + xyxy[j][3]) / 2.}\n')
            # save show
            # out_img_path = osp.join(out_dir, test_images[start + i])
            # result.save(filename=out_img_path)


def predict_test_time (test_dir):
    import time
    start_time = time.time()
    print(f'start_time: {start_time}')
    test_images = os.listdir(test_dir)
    for test_image in test_images:
        out_txt_path = osp.join(out_dir, test_image[:-4] + '.txt')
        with open(out_txt_path, 'w') as fwriter:
            print(f'creating {out_txt_path}')
    end_time = time.time()
    print(f'end_time: {end_time}')
    print(f'predict_test_time-total_time for {len(test_images)}: {end_time - start_time} s.')


if __name__ == '__main__':
    # 提交
    # model_path = '/workspace/model/best.pt'
    # test_dir = '/workspace/dataset/test/'
    # out_dir = '/workspace/outputs/predict/'

    # 测试
    test_dir = 'demo'
    out_dir = 'predict'

    model_path = 'rtdetr-l.pt'

    os.makedirs(out_dir, exist_ok=True)
    predict(model_path, test_dir, out_dir, args)

    # predict_test_time(test_dir)
    # print(f'__main__-total_time: {end_time - start_time} s.')
    end_time_import = time.time()
    spent_time_import = end_time_import - start_time_import
    print(f"processing 3000 images spent time:{spent_time_import * 1000}ms")
