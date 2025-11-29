#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
專案安裝和設置腳本
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """安裝必要的依賴套件"""
    print("正在安裝Python依賴套件...")
    
    requirements = [
        "opencv-python>=4.8.0",
        "pillow>=9.5.0", 
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "torch>=1.13.0",
        "torchvision>=0.14.0",
        "ultralytics>=8.0.0",
        "librosa>=0.9.0",
        "pydub>=0.25.0",
        "demucs>=4.0.0",
        "moviepy>=1.0.3",
        "ffmpeg-python>=0.2.0",
        "pandas>=1.5.0",
        "pyyaml>=6.0",
        "tqdm>=4.64.0",
        "pytest>=7.0.0"
    ]
    
    for package in requirements:
        try:
            print(f"安裝 {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            print(f"安裝 {package} 失敗: {e}")
            return False
    
    print("所有依賴套件安裝完成！")
    return True

def check_ffmpeg():
    """檢查FFmpeg是否已安裝"""
    try:
        result = subprocess.run(["ffmpeg", "-version"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ FFmpeg 已安裝")
            return True
    except FileNotFoundError:
        pass
    
    print("✗ FFmpeg 未安裝")
    print("請安裝FFmpeg:")
    print("- Windows: 從 https://ffmpeg.org/ 下載並添加到PATH")
    print("- macOS: brew install ffmpeg") 
    print("- Linux: sudo apt-get install ffmpeg")
    return False

def create_directories():
    """創建必要的目錄"""
    directories = [
        "../data/video",
        "../data/audio", 
        "../data/lyrics",
        "../output",
        "../temp",
        "../logs",
        "../models"
    ]
    
    print("創建必要目錄...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ {directory}")

def download_models():
    """下載必要的模型文件"""
    print("檢查模型文件...")
    
    # 檢查YOLO模型
    yolo_model_path = Path("../models/yolov8n.pt")
    if not yolo_model_path.exists():
        print("下載YOLO模型...")
        try:
            from ultralytics import YOLO
            model = YOLO('yolov8n.pt')  # 這會自動下載模型
            # 移動到models目錄
            import shutil
            if Path('yolov8n.pt').exists():
                shutil.move('yolov8n.pt', yolo_model_path)
            print("✓ YOLO模型下載完成")
        except Exception as e:
            print(f"✗ YOLO模型下載失敗: {e}")

def run_basic_test():
    """執行基本測試"""
    print("執行基本功能測試...")
    
    try:
        # 測試導入
        print("測試模組導入...")
        import yaml
        import cv2
        import numpy as np
        
        # 測試配置載入
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print("✓ 基本測試通過")
        return True
        
    except Exception as e:
        print(f"✗ 基本測試失敗: {e}")
        return False

def main():
    """主安裝流程"""
    print("=" * 50)
    print("自動剪輯影片生成系統 - 安裝腳本")
    print("=" * 50)
    
    # 檢查Python版本
    if sys.version_info < (3, 7):
        print("錯誤: 需要Python 3.7或更高版本")
        sys.exit(1)
    
    print(f"Python版本: {sys.version}")
    
    # 創建目錄
    create_directories()
    
    # 安裝依賴
    if not install_requirements():
        print("依賴安裝失敗，請手動安裝")
        sys.exit(1)
    
    # 檢查FFmpeg
    check_ffmpeg()
    
    # 下載模型
    download_models()
    
    # 基本測試
    if run_basic_test():
        print("\n" + "=" * 50)
        print("安裝完成！")
        print("=" * 50)
        print("使用方法:")
        print("python main.py --video input.mp4 --audio music.mp3 --output result.mp4")
        print("\n詳細說明請參閱 README.md")
    else:
        print("安裝過程中發現問題，請檢查錯誤訊息")

if __name__ == "__main__":
    main()