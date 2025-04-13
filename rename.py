import os
import shutil
import re
from pathlib import Path

def rename_images(directory_path, prefix="initial"):
    """
    根据文件名中括号内的数字重命名图片文件，格式为 prefix_number
    
    参数：
        directory_path: 包含图片的目录路径
        prefix: 新文件名的前缀（默认: "initial"）
    """
    # 确保目录存在
    if not os.path.exists(directory_path):
        print(f"目录 {directory_path} 不存在。")
        return
    
    # 获取目录中的所有文件
    files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    
    # 编译正则表达式，用于提取括号中的数字
    pattern = re.compile(r'\((\d+)\)')
    
    # 重命名每个文件
    for filename in files:
        try:
            # 获取文件路径和扩展名
            file_path = os.path.join(directory_path, filename)
            _, extension = os.path.splitext(filename)
            
            # 查找括号中的数字
            match = pattern.search(filename)
            if match:
                # 提取括号中的数字
                number = match.group(1)
                
                # 创建新文件名
                new_filename = f"{prefix}_{number}{extension}"
                new_file_path = os.path.join(directory_path, new_filename)
                
                # 检查目标文件是否已存在
                if os.path.exists(new_file_path):
                    print(f"目标文件已存在，跳过: {new_filename}")
                    continue
                
                # 重命名文件
                os.rename(file_path, new_file_path)
                print(f"已重命名: {filename} -> {new_filename}")
            else:
                print(f"无法从 {filename} 中提取数字，跳过")
        except Exception as e:
            print(f"重命名 {filename} 时出错: {e}")

if __name__ == "__main__":
    # 包含图片的目录
    directory_path = r"C:\Users\28266\Desktop\project\EraseText\test\prepare_kuake"
    
    # 重命名图片
    rename_images(directory_path)
    
    print("重命名完成!")
