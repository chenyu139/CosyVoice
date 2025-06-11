#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import re
from pathlib import Path

# 定义路径
SOURCE_DIR = "/home/chenyu/workplace/CosyVoice/examples/libritts/cosyvoice2/germany_voice_data"
TARGET_DIR = "/home/chenyu/workplace/CosyVoice/examples/libritts/cosyvoice2/germany_voice_process_data"
WAV_DIR = os.path.join(SOURCE_DIR, "wav")  # 假设音频文件在wav目录下

# 确保目标目录存在
for split in ["train", "test"]:
    os.makedirs(os.path.join(TARGET_DIR, split), exist_ok=True)

def extract_info_from_id(audio_id):
    """
    从音频ID中提取speaker_id和name
    ID格式: {speaker_id}_{book_id}_{segment_id}
    例如: 4705_13109_000000
    """
    parts = audio_id.split('_')
    if len(parts) >= 3:
        speaker_id = parts[0]
        book_id = parts[1]
        segment_id = parts[2]
        # 使用speaker_id和segment_id作为唯一标识
        name = f"{book_id}{segment_id}"
        return speaker_id, name
    else:
        return None, None

def process_transcripts_file(transcripts_file, output_dir, wav_source_dir=None):
    """
    处理transcripts.txt文件并生成对应的音频和文本文件
    
    参数:
        transcripts_file: transcripts.txt文件路径
        output_dir: 输出目录
        wav_source_dir: 音频文件源目录，如果为None则跳过音频文件复制
    """
    print(f"处理文件: {transcripts_file}")
    
    if not os.path.exists(transcripts_file):
        print(f"错误: 文件不存在: {transcripts_file}")
        return
    
    with open(transcripts_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            # 分割ID和文本内容
            parts = line.split('\t', 1)
            if len(parts) != 2:
                print(f"警告: 第{line_num}行格式不正确: {line}")
                continue
                
            audio_id = parts[0].strip()
            text = parts[1].strip()
            
            # 提取speaker_id和name
            speaker_id, name = extract_info_from_id(audio_id)
            
            if speaker_id and name:
                # 创建新的文件名
                new_wav_filename = f"{speaker_id}_{name}.wav"
                new_txt_filename = f"{speaker_id}_{name}.normalized.txt"
                
                # 目标文件路径
                target_txt_path = os.path.join(output_dir, new_txt_filename)
                
                # 写入文本内容
                with open(target_txt_path, 'w', encoding='utf-8') as txt_file:
                    txt_file.write(text)
                print(f"已创建文本: {target_txt_path}")
                
                # 如果指定了音频源目录，尝试复制音频文件
                if wav_source_dir:
                    # 尝试多种可能的音频文件名格式
                    possible_wav_names = [
                        f"{audio_id}.wav",
                        f"{audio_id}.flac",
                        f"{audio_id}.mp3"
                    ]
                    
                    source_wav_path = None
                    for wav_name in possible_wav_names:
                        potential_path = os.path.join(wav_source_dir, wav_name)
                        if os.path.exists(potential_path):
                            source_wav_path = potential_path
                            break
                    
                    if source_wav_path:
                        target_wav_path = os.path.join(output_dir, new_wav_filename)
                        shutil.copy2(source_wav_path, target_wav_path)
                        print(f"已复制音频: {source_wav_path} -> {target_wav_path}")
                    else:
                        print(f"警告: 未找到音频文件: {audio_id}")
            else:
                print(f"警告: 无法从ID提取信息: {audio_id}")

def main():
    # 处理测试集
    test_transcripts = os.path.join(SOURCE_DIR, "test", "transcripts.txt")
    if os.path.exists(test_transcripts):
        # 检查是否存在音频文件目录
        test_wav_dir = os.path.join(SOURCE_DIR, "test")
        if not any(f.endswith(('.wav', '.flac', '.mp3')) for f in os.listdir(test_wav_dir) if os.path.isfile(os.path.join(test_wav_dir, f))):
            test_wav_dir = None
            print("注意: 未找到音频文件，仅处理文本数据")
        
        process_transcripts_file(test_transcripts, os.path.join(TARGET_DIR, "test"), test_wav_dir)
    else:
        print(f"错误: 文件不存在: {test_transcripts}")
    
    # 处理训练集（如果存在）
    train_transcripts = os.path.join(SOURCE_DIR, "train", "transcripts.txt")
    if os.path.exists(train_transcripts):
        # 检查是否存在音频文件目录
        train_wav_dir = os.path.join(SOURCE_DIR, "train")
        if not any(f.endswith(('.wav', '.flac', '.mp3')) for f in os.listdir(train_wav_dir) if os.path.isfile(os.path.join(train_wav_dir, f))):
            train_wav_dir = None
            print("注意: 未找到训练集音频文件，仅处理文本数据")
        
        process_transcripts_file(train_transcripts, os.path.join(TARGET_DIR, "train"), train_wav_dir)
    else:
        print(f"注意: 训练集文件不存在: {train_transcripts}")
    
    print("德语数据转换完成！")

if __name__ == "__main__":
    main()