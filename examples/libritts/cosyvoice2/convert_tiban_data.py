#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import shutil
import re
from pathlib import Path

# 定义路径
SOURCE_DIR = "/home/chenyu/workplace/CosyVoice/examples/libritts/cosyvoice2/tiban_voice_data"
TARGET_DIR = "/home/chenyu/workplace/CosyVoice/examples/libritts/cosyvoice2/tiban_voice_process_data"
WAV_DIR = os.path.join(SOURCE_DIR, "wav")  # 假设音频文件在wav目录下

# 确保目标目录存在
for split in ["train", "test"]:
    os.makedirs(os.path.join(TARGET_DIR, split), exist_ok=True)

def extract_info_from_filename(filename):
    """
    从文件名中提取speaker_id和name
    文件名格式: SP02-OTR002-{speaker_id}-{order}-{name}.wav
    """
    pattern = r"SP02-OTR002-([^-]+)-([^-]+)-([^.]+)\.wav"
    match = re.match(pattern, filename)
    
    if match:
        speaker_id = match.group(1)
        order = match.group(2)
        name = match.group(3)
        return speaker_id, name
    else:
        return None, None

def process_tsv_file(tsv_file, output_dir):
    """
    处理TSV文件并生成对应的音频和文本文件
    
    参数:
        tsv_file: TSV文件路径
        output_dir: 输出目录
    """
    print(f"处理文件: {tsv_file}")
    
    with open(tsv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        # 跳过标题行
        header = next(reader, None)
        
        for row in reader:
            # 根据test-uni.tsv的格式，第二列是path，第三列是sentence
            if len(row) >= 3:
                index = row[0]  # 索引列
                wav_path = row[1].strip()  # 音频路径列
                text = row[2].strip()  # 文本内容列
                
                # 从wav路径提取文件名
                wav_filename = os.path.basename(wav_path)
                
                # 提取speaker_id和name
                speaker_id, name = extract_info_from_filename(wav_filename)
                
                if speaker_id and name:
                    # 创建新的文件名
                    new_wav_filename = f"{speaker_id}_{name}.wav"
                    new_txt_filename = f"{speaker_id}_{name}.normalized.txt"
                    
                    # 源音频文件路径
                    source_wav_path = os.path.join(WAV_DIR, wav_filename)
                    
                    # 目标文件路径
                    target_wav_path = os.path.join(output_dir, new_wav_filename)
                    target_txt_path = os.path.join(output_dir, new_txt_filename)
                    
                    # 复制音频文件
                    if os.path.exists(source_wav_path):
                        shutil.copy2(source_wav_path, target_wav_path)
                        print(f"已复制音频: {source_wav_path} -> {target_wav_path}")
                    else:
                        print(f"警告: 音频文件不存在: {source_wav_path}")
                    
                    # 写入文本内容
                    with open(target_txt_path, 'w', encoding='utf-8') as txt_file:
                        txt_file.write(text)
                    print(f"已创建文本: {target_txt_path}")
                else:
                    print(f"警告: 无法从文件名提取信息: {wav_filename}")

def main():
    # 处理测试集
    test_uni_tsv = os.path.join(SOURCE_DIR, "test-wylie.tsv")
    if os.path.exists(test_uni_tsv):
        process_tsv_file(test_uni_tsv, os.path.join(TARGET_DIR, "test"))
    else:
        print(f"错误: 文件不存在: {test_uni_tsv}")
    
    # 处理训练集
    train_uni_tsv = os.path.join(SOURCE_DIR, "train-wylie.tsv")
    if os.path.exists(train_uni_tsv):
        process_tsv_file(train_uni_tsv, os.path.join(TARGET_DIR, "train"))
    else:
        print(f"错误: 文件不存在: {train_uni_tsv}")
    
    print("转换完成！")

if __name__ == "__main__":
    main()