import os
import shutil
import pandas as pd
from tqdm import tqdm  # 如果报错说没有tqdm，请在终端输入 pip install tqdm

# ==================== 路径配置区域 ====================
# 1. 原始图片文件夹 (对应你截图里的 ISIC2018_Task3_Training_Input)
src_img_dir = r'E:\dataest\ISIC2018_Task3_Training_Input\ISIC2018_Task3_Training_Input'

# 2. 标签文件路径 (对应你截图里的 GroundTruth 文件夹下的 csv)
# 请确保文件名完全一致！
label_file = r'E:\dataest\ISIC2018_Task3_Training_GroundTruth\ISIC2018_Task3_Training_GroundTruth\ISIC2018_Task3_Training_GroundTruth.csv'
# 3. 目标输出文件夹 (程序会自动创建这个文件夹，不用你自己建)
target_dir = 'dataest_final'
print(f"图片文件夹是否存在：{os.path.exists(src_img_dir)}")
print(f"标签文件是否存在：{os.path.exists(label_file)}")
print(f"标签文件完整路径：{label_file}")
# 4. 切分比例 (80%训练, 10%验证, 10%测试)
train_ratio = 0.8
val_ratio = 0.1



def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def split_data():
    # 检查文件是否存在
    if not os.path.exists(src_img_dir):
        print(f"错误：找不到文件夹 {src_img_dir}")
        return
    if not os.path.exists(label_file):
        print(f"错误：找不到标签文件 {label_file}")
        return

    print("正在读取标签文件...")
    df = pd.read_csv(label_file)

    # 这里的类别顺序必须和CSV表头一致
    classes = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

    # 创建 train, val, test 及其子文件夹
    for split in ['train', 'val', 'test']:
        for cls in classes:
            make_dir(os.path.join(target_dir, split, cls))

    # 打乱数据，保证随机切分
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    total_num = len(df)
    train_end = int(total_num * train_ratio)
    val_end = train_end + int(total_num * val_ratio)

    print(f"开始处理 {total_num} 张图片...")

    for idx, row in tqdm(df.iterrows(), total=total_num):
        img_name = row['image']  # 图片文件名（不带后缀）

        # 找到这张图属于哪个类（找值为1.0的那一列）
        label = None
        for cls in classes:
            if row[cls] == 1.0:
                label = cls
                break

        # 如果没找到标签，跳过
        if label is None:
            continue

        # 确定源文件路径 (ISIC图片通常是 .jpg)
        src_path = os.path.join(src_img_dir, img_name + '.jpg')

        # 如果对应的图片文件不存在（有时csv里有但图没了），跳过
        if not os.path.exists(src_path):
            continue

        # 决定去哪个集合
        if idx < train_end:
            split = 'train'
        elif idx < val_end:
            split = 'val'
        else:
            split = 'test'

        # 复制文件到新家
        dst_path = os.path.join(target_dir, split, label, img_name + '.jpg')
        shutil.copy(src_path, dst_path)

    print("\n大功告成！")
    print(f"新的数据集在 '{target_dir}' 文件夹里。")


if __name__ == '__main__':
    split_data()