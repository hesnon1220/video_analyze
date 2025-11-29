#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
簡化的系統測試腳本 - 逐步測試各模組基本功能
"""

import sys
import os
from pathlib import Path
import json
import time

# 添加專案根目錄到路徑
sys.path.append(str(Path(__file__).parent))

from utils import setup_logger, load_config, save_json

def test_basic_imports():
    """測試基本模組導入"""
    print("=" * 60)
    print("步驟1: 測試基本模組導入")
    print("=" * 60)
    
    try:
        print("導入工具模組...")
        from utils import setup_logger, load_config
        print("✓ 工具模組導入成功")
        
        print("導入影像分析模組...")
        from image_analysis import SceneDetector, FeatureExtractor
        print("✓ 影像分析模組導入成功")
        
        print("導入音樂處理模組...")
        from music_processing import AudioSeparator, RhythmAnalyzer
        print("✓ 音樂處理模組導入成功")
        
        print("導入影片生成模組...")
        from video_generation import VideoComposer
        print("✓ 影片生成模組導入成功")
        
        return True
        
    except Exception as e:
        print(f"✗ 模組導入失敗: {e}")
        return False

def test_config_loading():
    """測試配置載入"""
    print("\n" + "=" * 60)
    print("步驟2: 測試配置載入")
    print("=" * 60)
    
    try:
        config = load_config('config.yaml')
        print(f"✓ 配置載入成功")
        print(f"  - 專案名稱: {config.get('project', {}).get('name', 'N/A')}")
        print(f"  - 版本: {config.get('project', {}).get('version', 'N/A')}")
        print(f"  - 配置項目數: {len(config)}")
        
        return config
        
    except Exception as e:
        print(f"✗ 配置載入失敗: {e}")
        return None

def test_simple_scene_detection():
    """簡化的場景檢測測試"""
    print("\n" + "=" * 60)
    print("步驟3: 簡化場景檢測測試")
    print("=" * 60)
    
    try:
        config = load_config('config.yaml')
        from image_analysis import SceneDetector
        
        # 創建場景檢測器
        scene_detector = SceneDetector(config['image_analysis'])
        print(f"✓ 場景檢測器創建成功")
        print(f"  - 直方圖閾值: {scene_detector.threshold}")
        print(f"  - 最小場景長度: {scene_detector.min_scene_length}")
        
        # 使用較短的測試影片進行測試
        test_videos = [
            r"F:\work\video_analyze\test_1.mp4",
            r"F:\work\video_analyze\test.mp4",
            r"F:\work\video_analyze\data\video\[Erai-raws] Beelzebub-jou no Okinimesu mama - 01 [720p][Multiple Subtitle].mp4"
        ]
        
        test_video = None
        for video in test_videos:
            if Path(video).exists():
                test_video = video
                break
        
        if test_video is None:
            print("✗ 找不到可用的測試影片")
            return False
        
        print(f"使用測試影片: {Path(test_video).name}")
        
        # 獲取影片資訊但不執行完整檢測
        from utils.common import get_video_info
        video_info = get_video_info(test_video)
        print(f"  - 影片時長: {video_info['duration']:.2f} 秒")
        print(f"  - 影片幀數: {video_info['frame_count']}")
        print(f"  - 影片解析度: {video_info['width']}x{video_info['height']}")
        
        print("✓ 場景檢測器基本測試通過")
        return True
        
    except Exception as e:
        print(f"✗ 場景檢測測試失敗: {e}")
        return False

def test_simple_music_analysis():
    """簡化的音樂分析測試"""
    print("\n" + "=" * 60)
    print("步驟4: 簡化音樂分析測試")
    print("=" * 60)
    
    try:
        config = load_config('config.yaml')
        from music_processing import RhythmAnalyzer
        
        # 創建節拍分析器
        rhythm_analyzer = RhythmAnalyzer(config['music_processing'])
        print(f"✓ 節拍分析器創建成功")
        
        # 尋找可用的音頻文件
        audio_files = [
            r"F:\work\video_analyze\data\test.mp3",
            r"F:\work\video_analyze\data\test.wav"
        ]
        
        test_audio = None
        for audio_file in audio_files:
            if Path(audio_file).exists():
                test_audio = audio_file
                break
        
        if test_audio is None:
            print("✗ 找不到可用的測試音頻文件")
            return False
        
        print(f"找到測試音頻: {Path(test_audio).name}")
        
        # 測試音頻載入
        try:
            y, sr = rhythm_analyzer.load_audio(test_audio)
            duration = len(y) / sr
            print(f"✓ 音頻載入成功")
            print(f"  - 時長: {duration:.2f} 秒")
            print(f"  - 採樣率: {sr} Hz")
            print(f"  - 樣本數: {len(y)}")
            
            return True
            
        except Exception as e:
            print(f"✗ 音頻載入失敗: {e}")
            return False
        
    except Exception as e:
        print(f"✗ 音樂分析測試失敗: {e}")
        return False

def test_video_composer():
    """測試影片合成器"""
    print("\n" + "=" * 60)
    print("步驟5: 測試影片合成器")
    print("=" * 60)
    
    try:
        config = load_config('config.yaml')
        from video_generation import VideoComposer
        
        # 創建影片合成器
        video_composer = VideoComposer(config['video_generation'])
        print(f"✓ 影片合成器創建成功")
        print(f"  - 輸出FPS: {video_composer.fps}")
        print(f"  - 輸出解析度: {video_composer.resolution}")
        print(f"  - 輸出格式: {video_composer.format}")
        
        # 測試歌詞載入
        lyrics_file = "docs/sample_lyrics.json"
        if Path(lyrics_file).exists():
            lyrics_data = video_composer.load_lyrics(lyrics_file)
            print(f"✓ 歌詞載入測試成功，載入了 {len(lyrics_data)} 句歌詞")
        else:
            print("⚠ 示例歌詞文件不存在")
        
        return True
        
    except Exception as e:
        print(f"✗ 影片合成器測試失敗: {e}")
        return False

def generate_test_report():
    """生成測試報告"""
    print("\n" + "=" * 60)
    print("步驟6: 生成測試報告")
    print("=" * 60)
    
    report = {
        "test_date": "2025-11-29",
        "test_time": time.strftime("%H:%M:%S"),
        "system_status": "基本功能測試完成",
        "modules_tested": [
            "utils - 工具模組",
            "image_analysis - 影像分析模組",
            "music_processing - 音樂處理模組", 
            "video_generation - 影片生成模組"
        ],
        "next_steps": [
            "執行完整的場景檢測測試",
            "執行音樂節拍分析測試",
            "執行端到端影片生成測試"
        ]
    }
    
    # 儲存測試報告
    output_file = "output/test_report.json"
    save_json(report, output_file)
    
    print(f"✓ 測試報告已生成: {output_file}")
    print("\n測試摘要:")
    print(f"  - 測試時間: {report['test_date']} {report['test_time']}")
    print(f"  - 測試模組數: {len(report['modules_tested'])}")
    print("\n下一步建議:")
    for i, step in enumerate(report['next_steps'], 1):
        print(f"  {i}. {step}")

def main():
    """執行簡化系統測試"""
    print("開始簡化系統測試...")
    
    # 測試步驟
    tests = [
        ("基本模組導入", test_basic_imports),
        ("配置載入", test_config_loading),
        ("場景檢測器", test_simple_scene_detection),
        ("音樂分析器", test_simple_music_analysis),
        ("影片合成器", test_video_composer)
    ]
    
    passed_tests = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed_tests += 1
            else:
                print(f"⚠ {test_name} 測試未完全通過")
        except Exception as e:
            print(f"✗ {test_name} 測試發生異常: {e}")
    
    # 生成測試報告
    generate_test_report()
    
    print(f"\n" + "=" * 60)
    print(f"🎯 系統測試完成: {passed_tests}/{len(tests)} 個測試通過")
    print("=" * 60)
    
    if passed_tests == len(tests):
        print("✅ 所有基本功能測試通過！系統準備就緒。")
        return True
    else:
        print("⚠️ 部分測試未通過，請檢查相關模組。")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)