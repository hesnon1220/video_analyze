#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
影像分析模組測試
"""

import sys
import os
from pathlib import Path
import json

# 添加專案根目錄到路徑
sys.path.append(str(Path(__file__).parent))

from utils import setup_logger, load_config, save_json
from image_analysis import SceneDetector, FeatureExtractor

def test_scene_detection():
    """測試場景檢測功能"""
    print("=" * 60)
    print("測試場景檢測模組")
    print("=" * 60)
    
    # 載入配置
    config = load_config('config.yaml')
    logger = setup_logger('scene_test', 'output/logs')
    
    # 選擇測試影片
    test_video = r"F:\work\video_analyze\data\video\[Erai-raws] Beelzebub-jou no Okinimesu mama - 01 [720p][Multiple Subtitle].mp4"
    
    if not Path(test_video).exists():
        print(f"測試影片不存在: {test_video}")
        return False
    
    print(f"測試影片: {Path(test_video).name}")
    
    try:
        # 創建場景檢測器
        scene_detector = SceneDetector(config['image_analysis'])
        
        # 執行場景檢測
        print("執行場景檢測...")
        scenes = scene_detector.detect_scenes(test_video)
        
        # 儲存結果
        output_file = "output/scene_detection/scenes_result.json"
        save_json(scenes, output_file)
        
        print(f"✓ 場景檢測完成！檢測到 {len(scenes)} 個場景")
        print(f"✓ 結果已儲存到: {output_file}")
        
        # 顯示前3個場景的資訊
        print("\n前3個場景詳情:")
        for i, scene in enumerate(scenes[:3]):
            print(f"  場景 {i+1}: {scene['start_time_str']} - {scene['end_time_str']} "
                  f"(時長: {scene['duration']:.2f}秒)")
        
        return True
        
    except Exception as e:
        print(f"✗ 場景檢測失敗: {e}")
        return False

def test_feature_extraction():
    """測試特徵提取功能"""
    print("\n" + "=" * 60)
    print("測試特徵提取模組")
    print("=" * 60)
    
    # 載入配置和之前的場景結果
    config = load_config('config.yaml')
    
    scenes_file = "output/scene_detection/scenes_result.json"
    if not Path(scenes_file).exists():
        print("需要先執行場景檢測測試")
        return False
    
    with open(scenes_file, 'r', encoding='utf-8') as f:
        scenes = json.load(f)
    
    test_video = r"F:\work\video_analyze\data\video\[Erai-raws] Beelzebub-jou no Okinimesu mama - 01 [720p][Multiple Subtitle].mp4"
    
    try:
        # 創建特徵提取器
        feature_extractor = FeatureExtractor(config['image_analysis'])
        
        # 只對前3個場景進行特徵提取（避免處理時間過長）
        test_scenes = scenes[:3]
        
        print(f"對前 {len(test_scenes)} 個場景進行特徵提取...")
        enhanced_scenes = feature_extractor.extract_features(test_video, test_scenes)
        
        # 儲存結果
        output_file = "output/feature_extraction/features_result.json"
        save_json(enhanced_scenes, output_file)
        
        print(f"✓ 特徵提取完成！")
        print(f"✓ 結果已儲存到: {output_file}")
        
        # 顯示特徵摘要
        print("\n特徵提取摘要:")
        for i, scene in enumerate(enhanced_scenes):
            features = scene.get('features', {})
            analysis = scene.get('analysis', {})
            
            print(f"  場景 {i+1}:")
            print(f"    - 亮度: {features.get('brightness', 0):.1f}")
            print(f"    - 對比度: {features.get('contrast', 0):.1f}")
            print(f"    - 檢測物體數: {features.get('total_objects', 0)}")
            print(f"    - 場景類型: {analysis.get('scene_type', 'unknown')}")
            print(f"    - 活動程度: {analysis.get('activity_level', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"✗ 特徵提取失敗: {e}")
        return False

def main():
    """執行影像分析測試"""
    print("開始影像分析模組測試...")
    
    # 測試場景檢測
    if not test_scene_detection():
        return False
    
    # 測試特徵提取
    if not test_feature_extraction():
        return False
    
    print("\n" + "=" * 60)
    print("🎉 影像分析模組測試完成！")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)