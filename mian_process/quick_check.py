#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速系統狀態檢查 - 驗證所有組件是否正常
"""

import os
import sys
from pathlib import Path

# 修復OpenMP衝突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '4'

def quick_system_check():
    """快速系統檢查"""
    print("🔍 快速系統狀態檢查")
    print("=" * 40)
    
    # 檢查基本導入
    modules_status = {}
    
    try:
        import torch
        modules_status['PyTorch'] = f"✅ {torch.__version__} (CUDA: {torch.cuda.is_available()})"
    except Exception as e:
        modules_status['PyTorch'] = f"❌ {e}"
    
    try:
        import cv2
        modules_status['OpenCV'] = f"✅ {cv2.__version__}"
    except Exception as e:
        modules_status['OpenCV'] = f"❌ {e}"
    
    try:
        from ultralytics import YOLO
        modules_status['YOLO'] = "✅ 可用"
    except Exception as e:
        modules_status['YOLO'] = f"❌ {e}"
    
    try:
        import librosa
        modules_status['Librosa'] = f"✅ {librosa.__version__}"
    except Exception as e:
        modules_status['Librosa'] = f"❌ {e}"
    
    try:
        import moviepy
        modules_status['MoviePy'] = f"✅ {moviepy.__version__}"
    except Exception as e:
        modules_status['MoviePy'] = f"❌ {e}"
    
    # 顯示結果
    print("📦 模組狀態:")
    for module, status in modules_status.items():
        print(f"  {module}: {status}")
    
    # 檢查重要文件
    print(f"\n📁 重要文件檢查:")
    important_files = [
        "config.yaml",
        "utils/__init__.py",
        "image_analysis/__init__.py",
        "music_processing/__init__.py",
        "video_generation/__init__.py",
        "../test_1.mp4",
        "../data/test.mp3"
    ]
    
    for file_path in important_files:
        if Path(file_path).exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path}")
    
    # 檢查GPU狀態
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"\n🖥️ GPU狀態: ✅ {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            print(f"\n🖥️ GPU狀態: ⚠️ 不可用，使用CPU模式")
    except:
        print(f"\n🖥️ GPU狀態: ❌ 檢查失敗")
    
    # 統計成功的模組
    success_count = sum(1 for status in modules_status.values() if status.startswith('✅'))
    total_count = len(modules_status)
    
    print(f"\n📊 整體狀態: {success_count}/{total_count} 模組正常")
    
    if success_count >= total_count * 0.8:  # 80%以上模組正常
        print("🎉 系統狀態良好，可以正常使用")
        return True
    else:
        print("⚠️ 部分模組有問題，建議檢查安裝")
        return False

if __name__ == "__main__":
    success = quick_system_check()
    print("\n" + "=" * 40)
    if success:
        print("✅ 系統檢查完成，狀態良好")
    else:
        print("⚠️ 系統檢查發現問題")