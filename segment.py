#!/usr/bin/python
# -*- coding: UTF-8 -*-

# 学信图片切割

import os
import shutil
import matplotlib.pyplot as plt
import cv2
import numpy as np

import ocr_training
import utils
from new_splitter import Splitter

output_path = '/var/www/tmp/'

def segment_and_pred(source_path, print_path):
    print('Start process pic:' + source_path)
    splitter = Splitter()
    image_color = cv2.imread(source_path)
    image_color = cv2.resize(image_color, (image_color.shape[1]*3, image_color.shape[0]*3))

    image_color = cv2.GaussianBlur(image_color, (11, 11), 19)

    image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    ret, adaptive_threshold = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
    """cv2.threshold()函数的作用是将一幅灰度图二值化,ret是true或false，adaptive_threshold是目标图像"""
    cv2.imwrite("white.png", adaptive_threshold)
    ret, at = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY_INV) #at shape(291, 657)
    # 计算换行内容索引
    """这一步是将有文字的行数提取出来"""
    cv2.imwrite("black.png", at)


    horizontal_sum = np.sum(at, axis=1)#返回矩阵中每一行的元f素相加
    peek_ranges = splitter.extract_peek_ranges_from_array(horizontal_sum)

    line_empty_count = 0
    result = ""
    for i in range(len(peek_ranges)):
        tmp1 = adaptive_threshold[peek_ranges[i - line_empty_count][0]: peek_ranges[i - line_empty_count][1], :]
        splitter.show_img('first image', tmp1)
        """将上面的到的tmp1图像保存为kv0，并存入resources中的degree里面"""
        kv0_path = print_path + str(i) + '/'
        if not os.path.exists(kv0_path):
            os.makedirs(kv0_path)
        cv2.imwrite(kv0_path + 'kv0.png', tmp1)

        space_number=splitter.process_by_path(kv0_path + 'kv0.png', kv0_path, minimun_range=10)

        os.remove(kv0_path + 'kv0.png')
        files = os.listdir(kv0_path)
        num_png = len(files)
        """舍弃一些图片小于1的文件夹"""
        if num_png > 1:
            pred_result, pred_val_list = chinese_ocr.pred(kv0_path,space_number=space_number)
            if i<len(peek_ranges)-1:
                result=result+pred_result+"\n"
            elif i==len(peek_ranges)-1:
                result=result+pred_result
    return result

# 校验图片size
def check_img(path, img_type):
    i = cv2.imread(path)
    if 'school' in img_type:
        if i.shape[0] >= 378 and (i.shape[0] - 378) % 20 == 0 and i.shape[1] == 660:
            result = {'code': 0, 'desc': ''}
        else:
            result = {'code': -1, 'desc': '图片size非法'}
            print("图片size非法")
    else:
        if i.shape[0] >= 294 and (i.shape[0] - 294) % 20 == 0 and i.shape[1] == 660:
            result = {'code': 0, 'desc': ''}
        else:
            result = {'code': -1, 'desc': '图片size非法'}
            print("图片size非法")
    return result

if __name__ == '__main__':
    result=segment_and_pred(
        './resources/frame.jpg',
        './resources/degree_frame/')