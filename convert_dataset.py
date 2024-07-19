#!/usr/bin/env python
# encoding: utf-8

# --------------------------------------------------------
# file: convert_dataset.py
# Copyright(c) 2017-2022 SeetaTech
# Written by Zhuang Liu
# 2024/07/02 14:47
# --------------------------------------------------------

import os
import os.path as osp
import cv2

def letter_box(img, size=(1024, 1024)):
    img_h, img_w = img.shape[:2]
    scale = min(size[0] / img_w, size[1] / img_h)
    new_size = (int(img_w * scale), int(img_h * scale))
    img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
    pad_left = (size[0] - new_size[0]) // 2
    pad_top = (size[1] - new_size[1]) // 2
    pad_right = size[0] - new_size[0] - pad_left
    pad_bottom = size[1] - new_size[1] - pad_top
    img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return img

def convert_jpg_1024_gray_bmp(img, size=(1024, 1024)):
    # letter_box
    convert_img = letter_box(img, size)
    # gray
    convert_img = cv2.cvtColor(convert_img, cv2.COLOR_BGR2GRAY)
    return convert_img


if __name__ == '__main__':
    src_dir = 'G:/Datasets/Detection/drone/uva-testdata'
    dst_dir = 'demo'
    if not osp.exists(dst_dir):
        os.makedirs(dst_dir, exist_ok=True)

    # 遍历源文件夹中的所有文件
    total_count = 3000
    count = 0
    for root, _, files in os.walk(src_dir):
        if 'JpegImages' not in root: continue
        for file in files:
            if file.endswith('.jpg'):
                count += 1
    dup_times = total_count // count

    start_id = 0
    for root, _, files in os.walk(src_dir):
        if 'JpegImages' not in root: continue
        for file in files:
            if file.endswith('.jpg'):
                for i in range(dup_times):
                    img_id = f'{start_id:06d}'
                    img_path = osp.join(root, file)
                    img = cv2.imread(img_path)
                    convert_img = convert_jpg_1024_gray_bmp(img)
                    cv2.imwrite(osp.join(dst_dir, f'{img_id}.bmp'), convert_img)
                    start_id += 1
                    print(f'{img_id}.bmp')

    remain_count = total_count % count
    for root, _, files in os.walk(src_dir):
        if 'JpegImages' not in root: continue
        if start_id >= total_count:
            break
        for file in files:
            if file.endswith('.jpg'):
                if start_id >= total_count:
                    break
                img_id = f'{start_id:06d}'
                img_path = osp.join(root, file)
                img = cv2.imread(img_path)
                convert_img = convert_jpg_1024_gray_bmp(img)
                cv2.imwrite(osp.join(dst_dir, f'{img_id}.bmp'), convert_img)
                start_id += 1
                print(f'{img_id}.bmp')

