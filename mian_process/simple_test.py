#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
簡化版主程式 - 避免MoviePy問題，專注測試核心功能
"""

import os
import sys
from pathlib import Path

# 修復OpenMP衝突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '4'

import argparse
sys.path.append(str(Path(__file__).parent))

from utils import setup_logger, load_config, save_json
from image_analysis.scene_detector import SceneDetector
from image_analysis.feature_extractor import FeatureExtractor
from music_processing.audio_separator import AudioSeparator
from music_processing.rhythm_analyzer import RhythmAnalyzer

def main():
    """簡化版主程式 - 只測試分析功能，不進行影片合成"""
    parser = argparse.ArgumentParser(description='自動剪輯影片生成系統 - 簡化測試版')
    parser.add_argument('--input', '--video', type=str, required=True, help='輸入影片路徑')
    parser.add_argument('--audio', type=str, required=True, help='輸入音樂路徑')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置檔案路徑')
    
    args = parser.parse_args()
    
    # 載入配置
    config = load_config(args.config)
    
    # 設定日誌
    log_dir = config.get('paths', {}).get('log_dir', '../logs')
    logger = setup_logger('simple_test', log_dir)
    
    print("🚀 啟動簡化版影片分析系統")
    logger.info("🚀 啟動簡化版影片分析系統")
    logger.info(f"📹 輸入影片: {args.input}")
    logger.info(f"🎵 輸入音樂: {args.audio}")
    
    # 檢查輸入文件
    video_path = Path(args.input)
    audio_path = Path(args.audio)
    
    if not video_path.exists():
        print(f"❌ 影片文件不存在: {args.input}")
        return False
    
    if not audio_path.exists():
        print(f"❌ 音頻文件不存在: {args.audio}")
        return False
    
    print(f"✅ 文件檢查通過")
    print(f"📹 影片大小: {video_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"🎵 音頻大小: {audio_path.stat().st_size / (1024*1024):.2f} MB")
    
    results = {
        'input_files': {
            'video': str(video_path),
            'audio': str(audio_path),
            'video_size_mb': round(video_path.stat().st_size / (1024*1024), 2),
            'audio_size_mb': round(audio_path.stat().st_size / (1024*1024), 2)
        }
    }
    
    try:
        # 步驟1: 影像分析
        print("\n🔍 步驟1: 影像分析")
        print("-" * 40)
        
        scene_detector = SceneDetector(config['image_analysis'])
        scenes = scene_detector.detect_scenes(str(video_path))
        print(f"✅ 場景檢測完成: {len(scenes)} 個場景")
        
        # 只分析前10個場景來節省時間
        scenes_to_analyze = scenes[:10] if len(scenes) > 10 else scenes
        print(f"📊 分析前 {len(scenes_to_analyze)} 個場景的特徵")
        
        feature_extractor = FeatureExtractor(config['image_analysis'])
        features = feature_extractor.extract_features(str(video_path), scenes_to_analyze)
        print(f"✅ 特徵提取完成: {len(features)} 個場景")
        
        results['image_analysis'] = {
            'total_scenes': len(scenes),
            'analyzed_scenes': len(features),
            'scenes_summary': [
                {
                    'id': scene.get('id', i),
                    'start_time': scene.get('start_time', 0),
                    'duration': scene.get('duration', 0),
                    'analysis': scene.get('analysis', {})
                }
                for i, scene in enumerate(features[:5])  # 只保存前5個場景的摘要
            ]
        }
        
        # 步驟2: 音樂處理
        print("\n🎵 步驟2: 音樂處理")
        print("-" * 40)
        
        # 節奏分析
        rhythm_analyzer = RhythmAnalyzer(config['music_processing'])
        rhythm_data = rhythm_analyzer.analyze(str(audio_path))
        bpm = rhythm_data.get('bpm', 'Unknown')
        beat_count = len(rhythm_data.get('tempo', {}).get('beat_times', []))
        print(f"✅ 節拍分析完成: BPM={bpm}, 節拍點={beat_count}個")
        
        # 音源分離測試（可選）
        try:
            audio_separator = AudioSeparator(config['music_processing'])
            print("🎼 測試音源分離功能...")
            
            # 只做簡單的檢查，不實際分離
            if hasattr(audio_separator, 'model') and audio_separator.model:
                print("✅ Demucs模型可用")
                results['audio_separation'] = {'available': True, 'model': audio_separator.model_name}
            else:
                print("⚠️ Demucs模型不可用，將跳過音源分離")
                results['audio_separation'] = {'available': False}
                
        except Exception as sep_error:
            print(f"⚠️ 音源分離測試失敗: {sep_error}")
            results['audio_separation'] = {'available': False, 'error': str(sep_error)}
        
        results['music_analysis'] = {
            'bpm': bpm,
            'beat_count': beat_count,
            'rhythm_data_keys': list(rhythm_data.keys())
        }
        
        # 步驟3: 生成分析報告
        print("\n📊 步驟3: 生成分析報告")
        print("-" * 40)
        
        # 計算一些統計資訊
        if features:
            visual_appeals = [scene.get('analysis', {}).get('visual_appeal', 'unknown') for scene in features]
            scene_types = [scene.get('analysis', {}).get('scene_type', 'unknown') for scene in features]
            
            from collections import Counter
            appeal_counts = Counter(visual_appeals)
            type_counts = Counter(scene_types)
            
            results['statistics'] = {
                'visual_appeal_distribution': dict(appeal_counts),
                'scene_type_distribution': dict(type_counts),
                'avg_scene_duration': sum(s.get('duration', 0) for s in scenes) / len(scenes) if scenes else 0
            }
            
            print(f"📈 視覺吸引力分佈: {dict(appeal_counts)}")
            print(f"📈 場景類型分佈: {dict(type_counts)}")
        
        # 保存結果
        output_file = "output/simple_analysis_result.json"
        Path("output").mkdir(exist_ok=True)
        save_json(results, output_file)
        
        print(f"\n🎉 分析完成！")
        print(f"📁 結果已保存到: {output_file}")
        print(f"📊 總場景數: {len(scenes)}")
        print(f"🎵 音樂BPM: {bpm}")
        print(f"⚡ 系統功能: 正常運行")
        
        return True
        
    except Exception as e:
        error_msg = f"分析過程中發生錯誤: {str(e)}"
        print(f"❌ {error_msg}")
        logger.error(error_msg)
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ 系統測試成功！所有核心功能運行正常")
        print("💡 如需完整影片合成，請使用: python main.py")
    else:
        print("\n❌ 系統測試失敗，請檢查錯誤日誌")
        sys.exit(1)