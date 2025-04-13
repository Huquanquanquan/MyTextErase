# 生成mask的函数如下

import os
import random
import numpy as np
from PIL import Image
import tqdm

# 输入：水印图像路劲，原图路劲，保存的mask的路径
def generate_one_mask(image_path, gt_path, save_path):
    # 读取图像
    image = Image.open(image_path)
    gt = Image.open(gt_path)

    # 转成numpy数组格式
    image = 255 - np.array(image)[:, :, :3]
    gt = 255 - np.array(gt)[:, :, :3]

    # 设置阈值
    threshold = 15
    # 真实图片与手写图片做差，找出mask的位置
    diff_image = np.abs(image.astype(np.float32) - gt.astype(np.float32))
    mean_image = np.max(diff_image, axis=-1)

    # 将mask二值化，即0和255。
    mask = np.greater(mean_image, threshold).astype(np.uint8) * 255
    mask[mask < 2] = 0
    mask[mask >= 1] = 255
    mask = 255 - mask
    mask = np.clip(mask, 0, 255)

    # 保存
    mask = np.array([mask, mask, mask, mask])
    mask = mask.transpose(1, 2, 0)
    mask = Image.fromarray(mask[:, :, :3])
    mask.save(save_path)

def generate_all_masks(dataset_path):
    """
    为整个数据集生成mask
    
    Args:
        dataset_path: 数据集根目录路径
    """
    # 获取图像和ground truth路径
    images_dir = os.path.join(dataset_path, 'images')
    gts_dir = os.path.join(dataset_path, 'gts')
    mask_dir = os.path.join(dataset_path, 'mask')
    
    # 确保mask目录存在
    os.makedirs(mask_dir, exist_ok=True)
    
    # 获取所有图像文件
    image_files = os.listdir(images_dir)
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 处理每个图像
    for image_file in tqdm.tqdm(image_files):
        image_path = os.path.join(images_dir, image_file)
        
        # 确定ground truth文件路径（可能是jpg或png）
        gt_file = image_file[:-4] + '.jpg'
        gt_path = os.path.join(gts_dir, gt_file)
        
        if not os.path.exists(gt_path):
            gt_file = image_file[:-4] + '.png'
            gt_path = os.path.join(gts_dir, gt_file)
            
            if not os.path.exists(gt_path):
                print(f"警告: 找不到图像 {image_file} 对应的ground truth文件")
                continue
        
        # 确定mask保存路径
        mask_file = image_file[:-4] + '.png'
        mask_path = os.path.join(mask_dir, mask_file)
        
        # 生成并保存mask
        try:
            generate_one_mask(image_path, gt_path, mask_path)
        except Exception as e:
            print(f"处理图像 {image_file} 时出错: {e}")

if __name__ == "__main__":
    # 设置数据集路径
    dataset_path = "dataset"
    
    # 生成所有mask
    generate_all_masks(dataset_path)
    
    print("所有mask生成完成！")