#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速功能測試 - 驗證系統各個組件是否正常工作
"""

import os
import sys
from pathlib import Path

# 修復OpenMP衝突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '4'

import argparse
sys.path.append(str(Path(__file__).parent))

from utils import setup_logger, load_config
import cv2
import librosa

def quick_test():
    """快速功能測試"""
    print("🚀 快速功能測試開始...")
    
    # 載入配置
    config = load_config('config.yaml')
    logger = setup_logger('quick_test', '../logs')
    
    # 測試文件路徑
    test_video = "../test_1.mp4"
    test_audio = "../data/test.mp3"
    
    if not Path(test_video).exists():
        print(f"❌ 測試影片不存在: {test_video}")
        return False
    
    if not Path(test_audio).exists():
        print(f"❌ 測試音頻不存在: {test_audio}")
        return False
    
    # 1. 測試影片讀取
    print("📹 測試影片讀取...")
    try:
        cap = cv2.VideoCapture(test_video)
        if not cap.isOpened():
            print("❌ 無法開啟影片")
            return False
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps
        
        # 只讀取前幾幀測試
        frames_to_test = min(10, frame_count)
        for i in range(frames_to_test):
            ret, frame = cap.read()
            if not ret:
                break
        
        cap.release()
        print(f"✅ 影片讀取成功 - 幀數: {frame_count}, 時長: {duration:.1f}秒")
        
    except Exception as e:
        print(f"❌ 影片讀取失敗: {e}")
        return False
    
    # 2. 測試音頻讀取
    print("🎵 測試音頻讀取...")
    try:
        y, sr = librosa.load(test_audio, duration=10)  # 只載入前10秒
        audio_duration = len(y) / sr
        print(f"✅ 音頻讀取成功 - 採樣率: {sr}, 測試時長: {audio_duration:.1f}秒")
        
    except Exception as e:
        print(f"❌ 音頻讀取失敗: {e}")
        return False
    
    # 3. 測試基本節拍檢測
    print("🎼 測試節拍檢測...")
    try:
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beats, sr=sr)
        print(f"✅ 節拍檢測成功 - BPM: {tempo:.1f}, 節拍點: {len(beat_times)}個")
        
    except Exception as e:
        print(f"❌ 節拍檢測失敗: {e}")
        return False
    
    # 4. 測試YOLO導入
    print("🎯 測試YOLO導入...")
    try:
        from ultralytics import YOLO
        print("✅ YOLO導入成功")
        
        # 測試模型載入（如果模型文件存在）
        model_path = "../models/yolov8n.pt"
        if Path(model_path).exists():
            model = YOLO(model_path)
            print("✅ YOLO模型載入成功")
        else:
            print("⚠️ YOLO模型文件不存在，但導入正常")
            
    except Exception as e:
        print(f"⚠️ YOLO測試失敗: {e}")
        # 不返回False，因為YOLO不是必需的
    
    # 5. 測試輸出目錄創建
    print("📁 測試輸出目錄...")
    try:
        output_dir = Path("output/quick_test")
        output_dir.mkdir(parents=True, exist_ok=True)
        print("✅ 輸出目錄創建成功")
        
    except Exception as e:
        print(f"❌ 輸出目錄創建失敗: {e}")
        return False
    
    print("\n🎉 快速功能測試完成！")
    print("系統基本功能正常，可以進行實際影片處理")
    return True

def run_quick_video_processing():
    """運行快速影片處理測試"""
    print("\n🔥 開始快速影片處理測試...")
    
    # 使用較短的測試參數
    test_video = "../test_1.mp4"
    test_audio = "../data/test.mp3" 
    output_path = "output/quick_test/result.mp4"
    
    # 構建簡化的處理命令
    cmd = f'python main.py --input "{test_video}" --audio "{test_audio}" --output "{output_path}"'
    print(f"執行命令: {cmd}")
    
    # 這裡可以添加實際的處理邏輯，但為了快速測試，暫時跳過
    print("⚠️ 完整處理測試需要較長時間，建議單獨執行")
    print("使用命令: python main.py --input ../test_1.mp4 --audio ../data/test.mp3")

if __name__ == "__main__":
    # 運行快速測試
    success = quick_test()
    
    if success:
        run_quick_video_processing()
    else:
        print("❌ 基本功能測試失敗，請檢查環境配置")
        sys.exit(1)