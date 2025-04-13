import os
import sys
import glob
import json
import cv2


import paddle
import paddle.nn as nn
import paddle.nn.functional as F

# 加载Erasenet改
from models.swin_gan import STRnet2_change
import utils
from paddle.vision.transforms import Compose, ToTensor
from PIL import Image


# 加载我们训练到的最好的模型
netG = STRnet2_change()
weights = paddle.load('train_models_swin_erasenet/best_submit_model.pdparams')
# weights = paddle.load('C:\Users\28266\Desktop\project\EraseText\train_models_swin_erasenet\best_submit_model.pdparams')
netG.load_dict(weights)
netG.eval()


def ImageTransform():
    return Compose([ToTensor()])

ImgTrans = ImageTransform()

def process(src_image_path, save_dir):
    # 加载图片
    img = Image.open(src_image_path)
    print(f"Loaded image from: {src_image_path}")
    print(f"Original image size: {img.size}")

    # 图片预处理
    inputImage = paddle.to_tensor([ImgTrans(img)])
    print(f"Input image shape after transform: {inputImage.shape}")
    print(f"Input image min/max values: {inputImage.min().item()}, {inputImage.max().item()}")

    # 填充图片到 512 的倍数
    _, _, h, w = inputImage.shape
    rh, rw = h, w
    step = 512
    pad_h = step - h if h < step else 0
    pad_w = step - w if w < step else 0
    m = nn.Pad2D((0, pad_w, 0, pad_h))
    imgs = m(inputImage)
    print(f"Padded image shape: {imgs.shape}")

    # 初始化结果张量
    res = paddle.zeros_like(imgs)

    # 分块处理
    for i in range(0, h, step):
        for j in range(0, w, step):
            if h - i < step:
                i = h - step
            if w - j < step:
                j = w - step
            clip = imgs[:, :, i:i + step, j:j + step]
            print(f"Processing clip at position: ({i}, {j}), shape: {clip.shape}")

            # 将数据移到 GPU
            clip = clip.cuda()
            print(f"Clip moved to GPU: {clip.place}")

            # 模型推理
            with paddle.no_grad():
                g_images_clip, mm = netG(clip)
            print(f"Model output g_images_clip shape: {g_images_clip.shape}")
            print(f"Model output mm shape: {mm.shape}")
            print(f"g_images_clip min/max: {g_images_clip.min().item()}, {g_images_clip.max().item()}")
            print(f"mm min/max: {mm.min().item()}, {mm.max().item()}")

            # 将数据移回 CPU
            g_images_clip = g_images_clip.cpu()
            mm = mm.cpu()
            clip = clip.cpu()

            # 限制 g_images_clip 的值范围
            g_images_clip = paddle.clip(g_images_clip, 0, 1)

            # 处理掩码
            mm = mm.mean(axis=1, keepdim=True)  # 将三通道掩码转换为单通道
            mm = (mm > 0.8).astype('float32')  # 提高阈值，增强掩码效果

            # 融合结果
            g_image_clip_with_mask = g_images_clip * mm + clip * (1 - mm)
            res[:, :, i:i + step, j:j + step] = g_image_clip_with_mask
            print(f"Clip processed successfully")

            # 释放内存
            del g_image_clip_with_mask, g_images_clip, mm, clip

    # 裁剪回原始大小
    res = res[:, :, :rh, :rw]
    print(f"Final result shape after cropping: {res.shape}")

    # 转换为图片格式
    output = (res.squeeze().transpose([1, 2, 0]).numpy() * 255).astype('uint8')
    print(f"Output image shape: {output.shape}")
    print(f"Output image min/max values: {output.min()}, {output.max()}")

    # 保存结果
    save_path = os.path.join(save_dir, os.path.basename(src_image_path).replace(".jpg", ".png"))
    cv2.imwrite(save_path, output)
    print(f"Result saved to: {save_path}")

    # 释放内存
    del output, res

# old code:
# def process(src_image_dir, save_dir):
#     image_paths = glob.glob(os.path.join(src_image_dir, "*.jpg"))
#     for image_path in image_paths:
#
#         # do something
#         img = Image.open(image_path)
#         inputImage = paddle.to_tensor([ImgTrans(img)])
#
#         _, _, h, w = inputImage.shape
#         rh, rw = h, w
#         step = 512
#         pad_h = step - h if h < step else 0
#         pad_w = step - w if w < step else 0
#         m = nn.Pad2D((0, pad_w, 0, pad_h))
#         imgs = m(inputImage)
#         _, _, h, w = imgs.shape
#         res = paddle.zeros_like(imgs)
#
#         for i in range(0, h, step):
#             for j in range(0, w, step):
#                 if h - i < step:
#                     i = h - step
#                 if w - j < step:
#                     j = w - step
#                 clip = imgs[:, :, i:i + step, j:j + step]
#                 clip = clip.cuda()
#                 with paddle.no_grad():
#                     g_images_clip, mm = netG(clip)
#                 g_images_clip = g_images_clip.cpu()
#                 mm = mm.cpu()
#                 clip = clip.cpu()
#                 g_image_clip_with_mask = g_images_clip * mm + clip * (1 - mm)
#                 res[:, :, i:i + step, j:j + step] = g_image_clip_with_mask
#                 del g_image_clip_with_mask, g_images_clip, mm, clip
#         res = res[:, :, :rh, :rw]
#         output = utils.pd_tensor2img(res)
#
#         # 保存结果图片
#         save_path = os.path.join(save_dir, os.path.basename(image_path).replace(".jpg", ".png"))
#         cv2.imwrite(save_path, output)
#         del output, res


if __name__ == "__main__":

    # assert len(sys.argv) == 3
    # src_image_dir = sys.argv[1]
    # save_dir = sys.argv[2]

    src_image_path = "test/test.jpg"
    save_dir = "test/test_result"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    process(src_image_path, save_dir)

