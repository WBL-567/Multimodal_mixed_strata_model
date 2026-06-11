## 再次修改程序，一个文件夹里面有多个.mat 文件，可以使程序能够自动把文件夹里面的每个.mat文件运行一遍，每个.mat 文件的结果单独放在一个文件夹里，命名规则不变。

import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from PIL import Image
import matplotlib

# 设置 Matplotlib 支持中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# ----------------------------------
# 配置参数
# ----------------------------------
SAMPLE_RATE = 2560  # 采样率（Hz）
SEGMENT_DURATION = 5  # 每个片段的时长（秒）
IMAGE_SIZE = (224, 224)  # 规范化图像大小


# 读取 MATLAB 文件
def load_matlab_data(mat_path):
    mat_data = scipy.io.loadmat(mat_path)
    time_array = mat_data['a2'].flatten()
    vibration_data = mat_data['a3'].flatten()
    return time_array, vibration_data


# 数据分段
def segment_data(time_array, data_array, segment_duration):
    segments = []
    start_time = time_array[0]
    end_time = time_array[-1]

    while start_time + segment_duration <= end_time:
        start_idx = np.searchsorted(time_array, start_time)
        end_idx = np.searchsorted(time_array, start_time + segment_duration)

        if end_idx - start_idx >= 0.9 * SAMPLE_RATE * segment_duration:
            segments.append((start_idx, end_idx))

        start_time += segment_duration

    return segments


# 计算 FFT
def compute_fft(data_segment, sample_rate):
    N = len(data_segment)
    window = np.hamming(N)  # 汉明窗
    data_segment_windowed = data_segment * window
    fft_result = fft(data_segment_windowed)
    fft_magnitude = np.abs(fft_result)[:N // 2]
    freq_axis = np.fft.fftfreq(N, d=1 / sample_rate)[:N // 2]
    return freq_axis, fft_magnitude


# 保存图像并进行数据增强
def save_fft_image(freq_axis, fft_magnitude, img_path):
    output_dir = os.path.dirname(img_path)

    # **确保目标文件夹存在**
    if not os.path.exists(output_dir):
        print(f"创建目录: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    print(f"保存图像: {img_path}")  # **调试用**

    fig = plt.figure(figsize=(4, 4), dpi=56)  # 224x224 规范化大小
    plt.plot(freq_axis, fft_magnitude, color='k')
    plt.xlabel("频率 (Hz)")
    plt.ylabel("振幅")
    plt.title("FFT 频谱")
    plt.grid()
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 读取并转换为灰度图
    img = Image.open(img_path).convert('L')
    img = img.resize(IMAGE_SIZE)
    img.save(img_path)


# 主程序
if __name__ == "__main__":
    current_dir = os.getcwd()
    mat_files = [f for f in os.listdir(current_dir) if f.endswith('.mat')]

    for mat_file in mat_files:
        mat_path = os.path.join(current_dir, mat_file)
        mat_name = os.path.splitext(mat_file)[0]
        output_dir = os.path.join(current_dir, f"fft_images_{mat_name}")

        # **创建并检查目录**
        os.makedirs(output_dir, exist_ok=True)
        if not os.path.exists(output_dir):
            print(f"目录创建失败: {output_dir}")
            continue  # 跳过该文件

        print(f"处理文件: {mat_file}")

        time_array, vibration_data = load_matlab_data(mat_path)
        segments = segment_data(time_array, vibration_data, SEGMENT_DURATION)

        if not segments:
            print(f"文件 {mat_file} 没有足够的有效数据段，跳过。")
            continue

        for idx, (start_idx, end_idx) in enumerate(segments):
            segment = vibration_data[start_idx:end_idx]
            freq_axis, fft_magnitude = compute_fft(segment, SAMPLE_RATE)
            img_path = os.path.join(output_dir, f"{mat_name}_segment_{idx + 1}.png")

            save_fft_image(freq_axis, fft_magnitude, img_path)

    print("所有文件处理完毕。")
