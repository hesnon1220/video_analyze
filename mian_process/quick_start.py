#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速開始腳本 - 演示系統基本功能
"""

import os
import sys
from pathlib import Path

def quick_start_demo():
    """快速開始演示"""
    print("=" * 60)
    print("自動剪輯影片生成系統 - 快速開始演示")
    print("=" * 60)
    
    # 檢查必要文件
    required_files = {
        'config.yaml': '配置檔案',
        'main.py': '主程式',
        'utils/__init__.py': '工具模組',
        'image_analysis/__init__.py': '影像分析模組',
        'music_processing/__init__.py': '音樂處理模組',
        'video_generation/__init__.py': '影片生成模組'
    }
    
    print("檢查專案文件...")
    missing_files = []
    for file_path, description in required_files.items():
        if Path(file_path).exists():
            print(f"✓ {description}: {file_path}")
        else:
            print(f"✗ {description}: {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n錯誤: 缺少必要文件: {missing_files}")
        return False
    
    # 測試模組導入
    print("\n測試模組導入...")
    try:
        import yaml
        from utils import setup_logger, load_config
        from image_analysis import SceneDetector, FeatureExtractor
        from music_processing import AudioSeparator, RhythmAnalyzer
        from video_generation import VideoComposer
        print("✓ 所有模組導入成功")
    except ImportError as e:
        print(f"✗ 模組導入失敗: {e}")
        print("請先執行: python setup.py")
        return False
    
    # 載入配置
    print("\n測試配置載入...")
    try:
        config = load_config('config.yaml')
        print(f"✓ 配置載入成功")
        print(f"  - 專案名稱: {config['project']['name']}")
        print(f"  - 版本: {config['project']['version']}")
    except Exception as e:
        print(f"✗ 配置載入失敗: {e}")
        return False
    
    # 創建處理器實例
    print("\n創建處理器實例...")
    try:
        scene_detector = SceneDetector(config['image_analysis'])
        feature_extractor = FeatureExtractor(config['image_analysis'])
        rhythm_analyzer = RhythmAnalyzer(config['music_processing'])
        video_composer = VideoComposer(config['video_generation'])
        print("✓ 所有處理器創建成功")
    except Exception as e:
        print(f"✗ 處理器創建失敗: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("系統檢查完成！")
    print("=" * 60)
    
    # 顯示使用範例
    print("\n使用範例:")
    print("1. 基本用法:")
    print("   python main.py --video input.mp4 --audio music.mp3 --output result.mp4")
    
    print("\n2. 包含歌詞:")
    print("   python main.py --video input.mp4 --audio music.mp3 --lyrics lyrics.json --output result.mp4")
    
    print("\n3. 自定義配置:")
    print("   python main.py --video input.mp4 --audio music.mp3 --config custom_config.yaml --output result.mp4")
    
    # 顯示可用的測試文件
    print("\n可用的示例文件:")
    sample_files = {
        'docs/sample_lyrics.json': '示例歌詞文件',
        'config.yaml': '配置文件',
        'tests/test_main.py': '測試腳本'
    }
    
    for file_path, description in sample_files.items():
        if Path(file_path).exists():
            print(f"  - {description}: {file_path}")
    
    print("\n注意事項:")
    print("1. 確保已安裝FFmpeg並添加到PATH")
    print("2. 首次使用YOLO時會自動下載模型文件")
    print("3. 使用demucs進行音源分離時需要較長時間")
    print("4. 建議使用GPU加速（設置config.yaml中的device為'cuda'）")
    
    return True

def run_simple_test():
    """執行簡單功能測試"""
    print("\n執行功能測試...")
    
    try:
        # 測試時間轉換
        from utils.common import time_to_seconds, seconds_to_time
        
        test_time = time_to_seconds("01:30")
        converted_back = seconds_to_time(test_time)
        print(f"✓ 時間轉換測試: 01:30 -> {test_time}s -> {converted_back}")
        
        # 測試配置載入
        from utils.common import load_config
        config = load_config('config.yaml')
        print(f"✓ 配置載入測試: 成功載入 {len(config)} 個配置項目")
        
        print("✓ 基本功能測試通過")
        return True
        
    except Exception as e:
        print(f"✗ 功能測試失敗: {e}")
        return False

def main():
    """主函數"""
    if not quick_start_demo():
        print("\n系統檢查失敗，請檢查安裝")
        sys.exit(1)
    
    if not run_simple_test():
        print("\n功能測試失敗，請檢查配置")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🎉 系統準備就緒！您可以開始使用自動剪輯影片生成系統了！")
    print("=" * 60)

if __name__ == "__main__":
    main()