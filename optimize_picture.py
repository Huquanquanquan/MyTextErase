from PIL import Image
import cv2
import numpy as np
from scipy import signal
import os


def optimized_exam_paper_processing(image_path, output_path, debug_dir=None):
    """
    优化的试卷预处理函数，基于大窗口自适应阈值处理的效果
    
    参数：
        image_path: 输入图像路径
        output_path: 输出图像路径
        debug_dir: 调试图像保存目录，如果为None则不保存调试图像
    """
    # 创建调试目录
    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
    
    # 保存调试图像函数
    def save_debug(name, img):
        if debug_dir is not None:
            debug_path = os.path.join(debug_dir, f"{name}.jpg")
            cv2.imwrite(debug_path, img)
    
    # 读取图片
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 保存原始图像
    save_debug("01_original", original)
    
    # 1. 基础预处理
    # 转换为灰度图
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    save_debug("02_gray", gray)
    
    # 2. 增强的降噪处理
    # 先使用双边滤波保留边缘
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    save_debug("03_bilateral", bilateral)
    
    # 应用非局部均值去噪 - 使用更强的参数
    denoised = cv2.fastNlMeansDenoising(bilateral, None, h=15, templateWindowSize=7, searchWindowSize=21)
    save_debug("04_denoised", denoised)
    
    # 3. 增强的对比度处理
    # 先进行直方图均衡化
    equalized = cv2.equalizeHist(denoised)
    save_debug("05_equalized", equalized)
    
    # 使用CLAHE增强局部对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(equalized)
    save_debug("06_enhanced", enhanced)
    
    # 4. 直接使用大窗口自适应阈值 - 这是关键步骤
    adaptive_large = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 51, 15  # 使用更大的窗口和偏移值
    )
    save_debug("07_adaptive_large", adaptive_large)
    
    # 5. 形态学操作 - 使用小结构元素保持清晰度
    # 创建结构元素
    kernel_small = np.ones((2, 2), np.uint8)
    
    # 先进行闭运算填充文字内小空洞
    closed = cv2.morphologyEx(adaptive_large, cv2.MORPH_CLOSE, kernel_small, iterations=1)
    save_debug("08_closed", closed)
    
    # 再进行开运算去除小噪点
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_small, iterations=1)
    save_debug("09_opened", opened)
    
    # 6. 确保白底黑字
    if np.mean(opened) < 127:
        opened = cv2.bitwise_not(opened)
    save_debug("10_white_background", opened)
    
    # 7. 连通区域分析 - 去除小噪点和手写标记
    # 寻找所有连通区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        cv2.bitwise_not(opened), connectivity=8
    )
    
    # 创建输出图像
    filtered = np.ones_like(opened) * 255
    
    # 计算图像面积
    img_area = opened.shape[0] * opened.shape[1]
    
    # 过滤连通区域
    min_size = 30  # 最小连通区域大小阈值
    max_size = img_area * 0.05  # 最大连通区域大小阈值
    
    # 保存有效区域的统计信息
    valid_regions = []
    
    # 从1开始，因为0是背景
    for i in range(1, num_labels):
        # 获取连通区域统计信息
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        # 计算长宽比
        aspect_ratio = w / h if h > 0 else 0
        
        # 判断是否为有效区域
        is_valid = True
        
        # 如果连通区域太小或太大，则忽略
        if area < min_size or area > max_size:
            is_valid = False
        
        # 如果长宽比异常（太细长或太扁平），可能是手写标记
        if aspect_ratio < 0.1 or aspect_ratio > 10:
            is_valid = False
        
        # 保存有效区域信息
        if is_valid:
            valid_regions.append({
                'id': i,
                'x': x, 'y': y, 'w': w, 'h': h,
                'area': area,
                'aspect_ratio': aspect_ratio
            })
            # 绘制有效区域
            filtered[labels == i] = 0
    
    save_debug("11_filtered", filtered)
    
    # 8. 文本行分析 - 基于有效区域的垂直分布
    if valid_regions:
        # 按y坐标排序
        valid_regions.sort(key=lambda r: r['y'])
        
        # 计算垂直距离
        y_distances = []
        for i in range(1, len(valid_regions)):
            prev_y = valid_regions[i-1]['y'] + valid_regions[i-1]['h']
            curr_y = valid_regions[i]['y']
            y_distances.append(curr_y - prev_y)
        
        # 如果有足够的区域来分析
        if y_distances:
            # 计算平均垂直距离
            avg_distance = np.mean(y_distances)
            
            # 创建文本行掩码
            text_line_mask = np.ones_like(filtered) * 255
            
            # 对每个有效区域判断是否属于文本行
            for region in valid_regions:
                # 检查周围是否有其他区域（文本行特征）
                has_neighbors = False
                region_center_x = region['x'] + region['w'] // 2
                region_center_y = region['y'] + region['h'] // 2
                
                for other in valid_regions:
                    if other['id'] == region['id']:
                        continue
                    
                    other_center_x = other['x'] + other['w'] // 2
                    other_center_y = other['y'] + other['h'] // 2
                    
                    # 水平距离
                    x_dist = abs(region_center_x - other_center_x)
                    # 垂直距离
                    y_dist = abs(region_center_y - other_center_y)
                    
                    # 如果在同一行附近且水平距离合理
                    if y_dist < avg_distance * 1.5 and x_dist < region['w'] * 5:
                        has_neighbors = True
                        break
                
                # 如果有邻居，更可能是文本的一部分
                if has_neighbors:
                    text_line_mask[labels == region['id']] = 0
            
            save_debug("12_text_line_mask", text_line_mask)
            
            # 结合文本行掩码和形状过滤结果
            filtered = cv2.bitwise_and(filtered, text_line_mask)
            save_debug("13_filtered_by_text_line", filtered)
    
    # 9. 最终清理 - 直接使用形态学操作，不进行平滑
    final_cleaned = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel_small, iterations=1)
    save_debug("14_final_cleaned", final_cleaned)
    
    # 10. 自动裁剪有效区域
    # 寻找非零区域
    coords = cv2.findNonZero(cv2.bitwise_not(final_cleaned))
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        # 添加边距
        margin = 50
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(final_cleaned.shape[1] - x, w + 2 * margin)
        h = min(final_cleaned.shape[0] - y, h + 2 * margin)
        # 裁剪图像
        cropped = final_cleaned[y:y+h, x:x+w]
    else:
        cropped = final_cleaned
    save_debug("15_cropped", cropped)
    
    # 11. 保存最终结果
    result_img = Image.fromarray(cropped)
    result_img.save(output_path)
    
    # 12. 质量评估
    # 计算对比度
    white_pixels = np.sum(cropped == 255)
    black_pixels = np.sum(cropped == 0)
    contrast_ratio = white_pixels / (black_pixels + 1e-6)
    
    # 计算噪声水平（使用拉普拉斯算子）
    laplacian = cv2.Laplacian(cropped, cv2.CV_64F)
    noise_level = laplacian.var()
    
    # 计算文本区域比例
    text_area_ratio = black_pixels / (white_pixels + black_pixels) * 100
    
    # 输出质量评估结果
    print(f"质量评估:")
    print(f"- 对比度比例: {contrast_ratio:.2f}")
    print(f"- 噪声水平: {noise_level:.2f}")
    print(f"- 白色像素占比: {white_pixels / (white_pixels + black_pixels) * 100:.2f}%")
    print(f"- 文本区域占比: {text_area_ratio:.2f}%")
    print(f"- 有效文本区域数量: {len(valid_regions)}")
    
    # 返回处理后的图像和质量指标
    return {
        'image': cropped,
        'contrast_ratio': contrast_ratio,
        'noise_level': noise_level,
        'white_percentage': white_pixels / (white_pixels + black_pixels) * 100,
        'text_area_ratio': text_area_ratio,
        'valid_regions_count': len(valid_regions)
    }

# 使用示例
input_image = r'C:\Users\28266\Desktop\project\EraseText\test\test2.jpg'
output_image = r'C:\Users\28266\Desktop\project\EraseText\test\test_white_bg2.jpg'
debug_dir = r'C:\Users\28266\Desktop\project\EraseText\test\debug_images4'
optimized_exam_paper_processing(input_image, output_image, debug_dir)