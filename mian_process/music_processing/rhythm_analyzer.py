#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
節拍分析模組 - 分析音樂的節拍、節奏和時間特徵
"""

import librosa
import numpy as np
import scipy.signal
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

class RhythmAnalyzer:
    """節拍分析器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.tempo_config = config.get('tempo', {})
        self.hop_length = self.tempo_config.get('hop_length', 512)
        self.logger = logging.getLogger(__name__)
        
    def load_audio(self, audio_path: str, sr: Optional[int] = 22050) -> Tuple[np.ndarray, int]:
        """載入音訊檔案"""
        try:
            y, sr = librosa.load(audio_path, sr=sr)
            self.logger.info(f"音訊載入成功: {audio_path}, 採樣率: {sr}")
            return y, sr
        except Exception as e:
            self.logger.error(f"音訊載入失敗: {e}")
            raise
    
    def extract_tempo_features(self, y: np.ndarray, sr: int) -> Dict:
        """提取節拍特徵"""
        # 提取節拍
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=self.hop_length)
        
        # 轉換beat時間點
        beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=self.hop_length)
        
        # 計算beat間隔
        beat_intervals = np.diff(beat_times)
        
        return {
            'tempo': float(tempo),
            'beats': beats.tolist(),
            'beat_times': beat_times.tolist(),
            'beat_intervals': beat_intervals.tolist(),
            'avg_beat_interval': float(np.mean(beat_intervals)) if len(beat_intervals) > 0 else 0.0,
            'beat_stability': float(np.std(beat_intervals)) if len(beat_intervals) > 0 else 0.0
        }
    
    def extract_onset_features(self, y: np.ndarray, sr: int) -> Dict:
        """提取onset特徵（音符開始點）"""
        # 提取onset
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=self.hop_length)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=self.hop_length)
        
        # 計算onset強度
        onset_strength = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)
        
        return {
            'onset_frames': onset_frames.tolist(),
            'onset_times': onset_times.tolist(),
            'onset_count': len(onset_times),
            'onset_strength': onset_strength.tolist(),
            'avg_onset_strength': float(np.mean(onset_strength))
        }
    
    def extract_spectral_features(self, y: np.ndarray, sr: int) -> Dict:
        """提取頻譜特徵"""
        # 提取MFCC特徵
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=self.hop_length)
        
        # 提取chroma特徵
        chroma = librosa.feature.chroma(y=y, sr=sr, hop_length=self.hop_length)
        
        # 提取spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=self.hop_length)[0]
        
        # 提取zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)[0]
        
        return {
            'mfcc_mean': np.mean(mfcc, axis=1).tolist(),
            'mfcc_std': np.std(mfcc, axis=1).tolist(),
            'chroma_mean': np.mean(chroma, axis=1).tolist(),
            'chroma_std': np.std(chroma, axis=1).tolist(),
            'spectral_centroid_mean': float(np.mean(spectral_centroids)),
            'spectral_centroid_std': float(np.std(spectral_centroids)),
            'zcr_mean': float(np.mean(zcr)),
            'zcr_std': float(np.std(zcr))
        }
    
    def extract_energy_features(self, y: np.ndarray, sr: int) -> Dict:
        """提取能量特徵"""
        # 計算RMS能量
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        
        # 計算短時能量
        frame_length = 2048
        energy = []
        for i in range(0, len(y) - frame_length, self.hop_length):
            frame_energy = np.sum(y[i:i+frame_length] ** 2)
            energy.append(frame_energy)
        
        energy = np.array(energy)
        
        return {
            'rms_mean': float(np.mean(rms)),
            'rms_std': float(np.std(rms)),
            'rms_max': float(np.max(rms)),
            'energy_mean': float(np.mean(energy)),
            'energy_std': float(np.std(energy)),
            'energy_max': float(np.max(energy))
        }
    
    def detect_music_structure(self, y: np.ndarray, sr: int) -> Dict:
        """檢測音樂結構"""
        # 使用chroma特徵進行結構分析
        chroma = librosa.feature.chroma(y=y, sr=sr, hop_length=self.hop_length)
        
        # 計算自相似矩陣
        similarity_matrix = np.dot(chroma.T, chroma)
        
        # 尋找重複片段（簡化版本）
        # 這裡可以實現更複雜的結構檢測算法
        
        return {
            'similarity_matrix_shape': similarity_matrix.shape,
            'max_similarity': float(np.max(similarity_matrix)),
            'avg_similarity': float(np.mean(similarity_matrix))
        }
    
    def analyze(self, audio_path: str) -> Dict:
        """
        完整分析音樂文件
        
        Args:
            audio_path (str): 音訊檔案路徑
            
        Returns:
            Dict: 包含所有分析結果的字典
        """
        self.logger.info(f"開始音樂分析: {audio_path}")
        
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"音訊檔案不存在: {audio_path}")
        
        # 載入音訊
        y, sr = self.load_audio(audio_path)
        
        analysis_result = {
            'file_info': {
                'path': audio_path,
                'duration': float(len(y) / sr),
                'sample_rate': sr,
                'samples': len(y)
            }
        }
        
        # 提取各種特徵
        try:
            self.logger.info("提取節拍特徵...")
            analysis_result['tempo'] = self.extract_tempo_features(y, sr)
            
            self.logger.info("提取onset特徵...")
            analysis_result['onset'] = self.extract_onset_features(y, sr)
            
            self.logger.info("提取頻譜特徵...")
            analysis_result['spectral'] = self.extract_spectral_features(y, sr)
            
            self.logger.info("提取能量特徵...")
            analysis_result['energy'] = self.extract_energy_features(y, sr)
            
            self.logger.info("分析音樂結構...")
            analysis_result['structure'] = self.detect_music_structure(y, sr)
            
        except Exception as e:
            self.logger.error(f"特徵提取過程中發生錯誤: {e}")
            raise
        
        self.logger.info("音樂分析完成")
        return analysis_result
    
    def get_cut_points_from_beats(self, beat_times: List[float], video_duration: float, 
                                  target_segments: int = 10) -> List[Tuple[float, float]]:
        """
        根據節拍時間點生成剪輯點
        
        Args:
            beat_times: 節拍時間點列表
            video_duration: 影片總長度
            target_segments: 目標片段數量
            
        Returns:
            List[Tuple[float, float]]: 剪輯片段的開始和結束時間
        """
        if not beat_times:
            # 如果沒有檢測到節拍，平均分割
            segment_length = video_duration / target_segments
            return [(i * segment_length, (i + 1) * segment_length) 
                    for i in range(target_segments)]
        
        # 根據節拍點生成剪輯片段
        segments = []
        beat_times = np.array(beat_times)
        
        # 確保覆蓋整個音樂長度
        if beat_times[-1] < video_duration * 0.9:
            # 如果節拍檢測沒有覆蓋大部分音樂，補充節拍點
            avg_interval = np.mean(np.diff(beat_times))
            while beat_times[-1] < video_duration:
                next_beat = beat_times[-1] + avg_interval
                beat_times = np.append(beat_times, next_beat)
        
        # 生成片段
        segment_length = len(beat_times) // target_segments
        for i in range(target_segments):
            start_idx = i * segment_length
            end_idx = min((i + 1) * segment_length, len(beat_times) - 1)
            
            start_time = beat_times[start_idx]
            end_time = beat_times[end_idx]
            
            segments.append((float(start_time), float(end_time)))
        
        return segments