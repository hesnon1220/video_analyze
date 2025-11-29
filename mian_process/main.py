#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自動剪輯影片生成系統 - 主程式入口點
"""

import os
import sys
from pathlib import Path

# 🔧 修復OpenMP衝突問題 - 必須在其他import之前
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '4'

import argparse
import yaml
import logging

# 添加專案根目錄到 Python 路徑
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from utils import setup_logger, load_config
from image_analysis.scene_detector import SceneDetector
from image_analysis.feature_extractor import FeatureExtractor
from music_processing.audio_separator import AudioSeparator
from music_processing.rhythm_analyzer import RhythmAnalyzer
from video_generation.video_composer import VideoComposer

def main():
    """主程式"""
    parser = argparse.ArgumentParser(description='自動剪輯影片生成系統')
    parser.add_argument('--video', '--input', type=str, required=True, help='輸入影片路徑')
    parser.add_argument('--audio', type=str, required=True, help='輸入音樂路徑')
    parser.add_argument('--lyrics', type=str, help='歌詞檔案路徑')
    parser.add_argument('--output', type=str, default='output/final_output.mp4', help='輸出影片路徑')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置檔案路徑')
    
    args = parser.parse_args()
    
    # 確保輸出目錄存在
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 載入配置
    config = load_config(args.config)
    
    # 設定日誌 - 修復路徑問題
    log_dir = config.get('paths', {}).get('log_dir', '../logs')
    logger = setup_logger('main', log_dir)
    
    logger.info("🚀 啟動自動剪輯影片生成系統")
    logger.info(f"📹 輸入影片: {args.video}")
    logger.info(f"🎵 輸入音樂: {args.audio}")
    logger.info(f"📤 輸出路徑: {args.output}")
    
    # 檢查輸入文件是否存在
    video_path = Path(args.video)
    audio_path = Path(args.audio)
    
    if not video_path.exists():
        logger.error(f"❌ 影片文件不存在: {args.video}")
        print(f"❌ 影片文件不存在: {args.video}")
        sys.exit(1)
    
    if not audio_path.exists():
        logger.error(f"❌ 音頻文件不存在: {args.audio}")
        print(f"❌ 音頻文件不存在: {args.audio}")
        sys.exit(1)
    
    # 顯示文件資訊
    logger.info(f"影片文件大小: {video_path.stat().st_size / (1024*1024):.2f} MB")
    logger.info(f"音頻文件大小: {audio_path.stat().st_size / (1024*1024):.2f} MB")
    
    try:
        # 步驟1: 影像分析
        print("🔍 步驟1: 開始影像分析...")
        logger.info("開始影像分析...")
        
        scene_detector = SceneDetector(config['image_analysis'])
        scenes = scene_detector.detect_scenes(str(video_path))
        print(f"✅ 檢測到 {len(scenes)} 個場景")
        
        feature_extractor = FeatureExtractor(config['image_analysis'])
        features = feature_extractor.extract_features(str(video_path), scenes)
        print(f"✅ 提取了 {len(features)} 個場景的特徵")
        
        # 步驟2: 音樂處理
        print("🎵 步驟2: 開始音樂處理...")
        logger.info("開始音樂處理...")
        
        # 音源分離
        audio_separator = AudioSeparator(config['music_processing'])
        separated_audio = audio_separator.separate(str(audio_path))
        
        if separated_audio.get('separated', False):
            print("✅ 音源分離完成")
        else:
            print("⚠️ 音源分離跳過，使用原始音頻")
        
        # 節奏分析
        rhythm_analyzer = RhythmAnalyzer(config['music_processing'])
        rhythm_data = rhythm_analyzer.analyze(str(audio_path))
        print(f"✅ 節拍分析完成，BPM: {rhythm_data.get('bpm', 'Unknown')}")
        
        # 步驟3: 影片生成
        print("🎬 步驟3: 開始影片生成...")
        logger.info("開始影片生成...")
        
        video_composer = VideoComposer(config['video_generation'])
        output_path = video_composer.compose(
            video_path=str(video_path),
            scenes=scenes,
            features=features,
            rhythm_data=rhythm_data,
            lyrics_path=args.lyrics,
            output_path=args.output
        )
        
        print(f"🎉 影片生成完成: {output_path}")
        logger.info(f"影片生成完成: {output_path}")
        
        # 顯示輸出文件資訊
        if Path(output_path).exists():
            output_size = Path(output_path).stat().st_size / (1024*1024)
            print(f"📊 輸出文件大小: {output_size:.2f} MB")
        
    except Exception as e:
        error_msg = f"處理過程中發生錯誤: {str(e)}"
        print(f"❌ {error_msg}")
        logger.error(error_msg)
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()