#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
測試主模組功能
"""

import pytest
import os
import sys
from pathlib import Path

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.common import get_video_info, time_to_seconds, seconds_to_time
from image_analysis.scene_detector import SceneDetector
from image_analysis.feature_extractor import FeatureExtractor
from music_processing.rhythm_analyzer import RhythmAnalyzer
from video_generation.video_composer import VideoComposer

class TestUtils:
    """測試工具模組"""
    
    def test_time_conversion(self):
        """測試時間轉換函數"""
        # 測試 time_to_seconds
        assert time_to_seconds("01:30") == 90.0
        assert time_to_seconds("01:02:30") == 3750.0
        
        # 測試 seconds_to_time
        assert seconds_to_time(90.0) == "00:01:30.000"
        assert seconds_to_time(3750.0) == "01:02:30.000"

class TestImageAnalysis:
    """測試影像分析模組"""
    
    def setup_method(self):
        """測試前設置"""
        self.config = {
            'histogram': {
                'threshold': 0.3,
                'min_scene_length': 2.0
            },
            'yolo': {
                'confidence': 0.5,
                'device': 'cpu'
            }
        }
    
    def test_scene_detector_init(self):
        """測試場景檢測器初始化"""
        detector = SceneDetector(self.config)
        assert detector.threshold == 0.3
        assert detector.min_scene_length == 2.0
    
    def test_feature_extractor_init(self):
        """測試特徵提取器初始化"""
        extractor = FeatureExtractor(self.config)
        assert extractor.confidence == 0.5
        assert extractor.device == 'cpu'

class TestMusicProcessing:
    """測試音樂處理模組"""
    
    def setup_method(self):
        """測試前設置"""
        self.config = {
            'demucs': {
                'model': 'htdemucs',
                'device': 'cpu'
            },
            'tempo': {
                'hop_length': 512
            }
        }
    
    def test_rhythm_analyzer_init(self):
        """測試節拍分析器初始化"""
        analyzer = RhythmAnalyzer(self.config)
        assert analyzer.hop_length == 512

class TestVideoGeneration:
    """測試影片生成模組"""
    
    def setup_method(self):
        """測試前設置"""
        self.config = {
            'fps': 30,
            'resolution': [1920, 1080],
            'format': 'mp4'
        }
    
    def test_video_composer_init(self):
        """測試影片合成器初始化"""
        composer = VideoComposer(self.config)
        assert composer.fps == 30
        assert composer.resolution == [1920, 1080]
        assert composer.format == 'mp4'

def run_tests():
    """執行所有測試"""
    pytest.main([__file__, '-v'])

if __name__ == "__main__":
    run_tests()