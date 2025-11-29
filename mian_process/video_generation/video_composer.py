#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
影片合成模組 - 整合影像、音樂和歌詞生成最終影片
"""

import cv2
import numpy as np
from moviepy.editor import (
    VideoFileClip, AudioFileClip, CompositeVideoClip,
    TextClip, concatenate_videoclips
)
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import json

from utils.common import get_video_info, ensure_dir

class VideoComposer:
    """影片合成器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.fps = config.get('fps', 30)
        self.resolution = config.get('resolution', [1920, 1080])
        self.format = config.get('format', 'mp4')
        self.logger = logging.getLogger(__name__)
        
    def load_lyrics(self, lyrics_path: str) -> List[Dict]:
        """載入歌詞檔案"""
        if not lyrics_path or not Path(lyrics_path).exists():
            self.logger.warning("歌詞檔案不存在，將不顯示歌詞")
            return []
        
        try:
            with open(lyrics_path, 'r', encoding='utf-8') as f:
                if lyrics_path.endswith('.json'):
                    lyrics_data = json.load(f)
                else:
                    # 簡單文本格式：每行一句歌詞
                    lines = f.readlines()
                    lyrics_data = []
                    for i, line in enumerate(lines):
                        lyrics_data.append({
                            'start': i * 3.0,  # 假設每句3秒
                            'end': (i + 1) * 3.0,
                            'text': line.strip()
                        })
                
            self.logger.info(f"載入歌詞: {len(lyrics_data)} 句")
            return lyrics_data
            
        except Exception as e:
            self.logger.error(f"歌詞載入失敗: {e}")
            return []
    
    def select_best_scenes(self, scenes: List[Dict], rhythm_data: Dict, 
                          target_count: int = 10) -> List[Dict]:
        """
        根據節拍和場景特徵選擇最佳場景
        
        Args:
            scenes: 場景列表
            rhythm_data: 節拍分析結果
            target_count: 目標場景數量
            
        Returns:
            選中的場景列表
        """
        if len(scenes) <= target_count:
            return scenes
        
        # 為每個場景計算評分
        scored_scenes = []
        for scene in scenes:
            score = self._calculate_scene_score(scene, rhythm_data)
            scored_scenes.append((scene, score))
        
        # 按評分排序並選擇前N個
        scored_scenes.sort(key=lambda x: x[1], reverse=True)
        selected_scenes = [scene for scene, score in scored_scenes[:target_count]]
        
        # 按時間順序重新排列
        selected_scenes.sort(key=lambda x: x['start_time'])
        
        self.logger.info(f"從 {len(scenes)} 個場景中選擇了 {len(selected_scenes)} 個")
        return selected_scenes
    
    def _calculate_scene_score(self, scene: Dict, rhythm_data: Dict) -> float:
        """計算場景評分"""
        score = 0.0
        
        # 基於場景分析的評分
        if 'analysis' in scene:
            analysis = scene['analysis']
            
            # 場景評分 (0-1)
            scene_score = analysis.get('scene_score', 0.5)
            score += scene_score * 0.4
            
            # 活動程度評分
            activity_level = analysis.get('activity_level', 'low')
            activity_scores = {'high': 1.0, 'medium': 0.7, 'low': 0.4}
            score += activity_scores.get(activity_level, 0.4) * 0.3
            
            # 物體數量評分
            if 'features' in scene and 'total_objects' in scene['features']:
                object_count = scene['features']['total_objects']
                object_score = min(object_count / 5.0, 1.0)  # 標準化到0-1
                score += object_score * 0.2
        
        # 場景長度評分（偏好適中長度的場景）
        duration = scene.get('duration', 0)
        if 2.0 <= duration <= 8.0:
            length_score = 1.0
        else:
            length_score = max(0.3, 1.0 - abs(duration - 5.0) / 10.0)
        score += length_score * 0.1
        
        return score
    
    def create_scene_clips(self, video_path: str, selected_scenes: List[Dict], 
                          beat_segments: List[Tuple[float, float]]) -> List:
        """創建場景片段"""
        video_clip = VideoFileClip(video_path)
        scene_clips = []
        
        for i, (scene, (start_time, end_time)) in enumerate(zip(selected_scenes, beat_segments)):
            # 確保時間範圍在場景內
            scene_start = max(scene['start_time'], start_time)
            scene_end = min(scene['end_time'], end_time)
            
            if scene_end <= scene_start:
                # 如果時間範圍無效，使用場景的前幾秒
                scene_start = scene['start_time']
                scene_end = min(scene['start_time'] + (end_time - start_time), scene['end_time'])
            
            # 創建片段
            clip = video_clip.subclip(scene_start, scene_end)
            
            # 調整片段長度以匹配節拍
            target_duration = end_time - start_time
            if clip.duration != target_duration:
                # 如果需要，調整播放速度
                speed_factor = clip.duration / target_duration
                if 0.5 <= speed_factor <= 2.0:  # 合理的速度範圍
                    clip = clip.fx(lambda clip: clip.speedx(speed_factor))
                else:
                    # 如果速度調整太大，就截取或循環
                    if clip.duration < target_duration:
                        clip = clip.loop(duration=target_duration)
                    else:
                        clip = clip.subclip(0, target_duration)
            
            scene_clips.append(clip)
            self.logger.info(f"創建場景片段 {i+1}: {scene_start:.2f}s - {scene_end:.2f}s")
        
        return scene_clips
    
    def add_lyrics_overlay(self, video_clip, lyrics_data: List[Dict]) -> CompositeVideoClip:
        """添加歌詞覆蓋"""
        if not lyrics_data:
            return video_clip
        
        lyrics_clips = []
        
        for lyric in lyrics_data:
            start_time = lyric['start']
            end_time = lyric['end']
            text = lyric['text']
            
            if start_time < video_clip.duration:
                # 創建文字片段
                txt_clip = TextClip(
                    text,
                    fontsize=50,
                    color='white',
                    stroke_color='black',
                    stroke_width=2,
                    font='Arial-Bold'
                ).set_position(('center', 'bottom')).set_start(start_time)
                
                # 調整結束時間
                actual_end = min(end_time, video_clip.duration)
                txt_clip = txt_clip.set_duration(actual_end - start_time)
                
                lyrics_clips.append(txt_clip)
        
        if lyrics_clips:
            self.logger.info(f"添加 {len(lyrics_clips)} 個歌詞片段")
            return CompositeVideoClip([video_clip] + lyrics_clips)
        
        return video_clip
    
    def compose(self, video_path: str, scenes: List[Dict], features: List[Dict],
               rhythm_data: Dict, lyrics_path: Optional[str] = None,
               output_path: str = "output.mp4") -> str:
        """
        合成最終影片
        
        Args:
            video_path: 原始影片路徑
            scenes: 場景列表
            features: 場景特徵
            rhythm_data: 節拍分析結果
            lyrics_path: 歌詞檔案路徑
            output_path: 輸出影片路徑
            
        Returns:
            輸出影片路徑
        """
        self.logger.info("開始影片合成...")
        
        # 確保輸出目錄存在
        ensure_dir(Path(output_path).parent)
        
        # 載入歌詞
        lyrics_data = self.load_lyrics(lyrics_path) if lyrics_path else []
        
        # 獲取節拍片段
        beat_times = rhythm_data.get('tempo', {}).get('beat_times', [])
        video_info = get_video_info(video_path)
        video_duration = video_info['duration']
        
        # 從節拍分析器獲取剪輯點
        from music_processing.rhythm_analyzer import RhythmAnalyzer
        analyzer = RhythmAnalyzer(self.config)
        beat_segments = analyzer.get_cut_points_from_beats(beat_times, video_duration)
        
        # 選擇最佳場景
        selected_scenes = self.select_best_scenes(scenes, rhythm_data, len(beat_segments))
        
        # 創建場景片段
        scene_clips = self.create_scene_clips(video_path, selected_scenes, beat_segments)
        
        if not scene_clips:
            raise ValueError("沒有有效的場景片段可以合成")
        
        # 合併所有片段
        self.logger.info("合併影片片段...")
        final_video = concatenate_videoclips(scene_clips)
        
        # 添加歌詞覆蓋
        if lyrics_data:
            final_video = self.add_lyrics_overlay(final_video, lyrics_data)
        
        # 輸出影片
        self.logger.info(f"輸出影片到: {output_path}")
        final_video.write_videofile(
            output_path,
            fps=self.fps,
            codec='libx264',
            audio_codec='aac'
        )
        
        # 清理資源
        final_video.close()
        
        self.logger.info("影片合成完成")
        return output_path