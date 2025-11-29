#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自動剪輯影片生成系統 - 主程式入口點
"""

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path

# 添加專案根目錄到 Python 路徑
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from utils.logger import setup_logger
from image_analysis.scene_detector import SceneDetector
from image_analysis.feature_extractor import FeatureExtractor
from music_processing.audio_separator import AudioSeparator
from music_processing.rhythm_analyzer import RhythmAnalyzer
from video_generation.video_composer import VideoComposer

def load_config(config_path="config.yaml"):
    """載入配置檔案"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    """主程式"""
    parser = argparse.ArgumentParser(description='自動剪輯影片生成系統')
    parser.add_argument('--video', type=str, required=True, help='輸入影片路徑')
    parser.add_argument('--audio', type=str, required=True, help='輸入音樂路徑')
    parser.add_argument('--lyrics', type=str, help='歌詞檔案路徑')
    parser.add_argument('--output', type=str, default='output.mp4', help='輸出影片路徑')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置檔案路徑')
    
    args = parser.parse_args()
    
    # 載入配置
    config = load_config(args.config)
    
    # 設定日誌
    logger = setup_logger('main', config['paths']['log_dir'])
    logger.info("啟動自動剪輯影片生成系統")
    
    try:
        # 步驟1: 影像分析
        logger.info("開始影像分析...")
        scene_detector = SceneDetector(config['image_analysis'])
        scenes = scene_detector.detect_scenes(args.video)
        
        feature_extractor = FeatureExtractor(config['image_analysis'])
        features = feature_extractor.extract_features(args.video, scenes)
        
        # 步驟2: 音樂處理
        logger.info("開始音樂處理...")
        audio_separator = AudioSeparator(config['music_processing'])
        separated_audio = audio_separator.separate(args.audio)
        
        rhythm_analyzer = RhythmAnalyzer(config['music_processing'])
        rhythm_data = rhythm_analyzer.analyze(args.audio)
        
        # 步驟3: 影片生成
        logger.info("開始影片生成...")
        video_composer = VideoComposer(config['video_generation'])
        output_path = video_composer.compose(
            video_path=args.video,
            scenes=scenes,
            features=features,
            rhythm_data=rhythm_data,
            lyrics_path=args.lyrics,
            output_path=args.output
        )
        
        logger.info(f"影片生成完成: {output_path}")
        
    except Exception as e:
        logger.error(f"處理過程中發生錯誤: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()