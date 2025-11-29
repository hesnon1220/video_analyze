#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自動剪輯影片生成系統

這是一個基於影像分析、音樂處理和智能剪輯的自動化影片生成系統。
輸入影像、音樂、歌詞，自動生成剪輯影片。

主要模組：
- image_analysis: 影像分析和場景檢測
- music_processing: 音樂處理和節拍分析
- video_generation: 影片合成和生成
- utils: 通用工具函數

使用方法：
python main.py --video input.mp4 --audio music.mp3 --lyrics lyrics.txt --output result.mp4
"""

__version__ = "1.0.0"
__author__ = "Video Analyze Team"
__description__ = "自動剪輯影片生成系統"

from utils import setup_logger, load_config
from image_analysis import SceneDetector, FeatureExtractor
from music_processing import AudioSeparator, RhythmAnalyzer
from video_generation import VideoComposer

__all__ = [
    'setup_logger', 'load_config',
    'SceneDetector', 'FeatureExtractor',
    'AudioSeparator', 'RhythmAnalyzer',
    'VideoComposer'
]