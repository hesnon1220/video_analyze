#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
場景檢測模組 - 基於直方圖差異進行場景切割
參考: old_process/get_3hist.py, old_process/Helper_private.py
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm
import logging

from utils.common import get_video_info, seconds_to_time

class SceneDetector:
    """場景檢測器"""
    
    def __init__(self, config: Dict):
        self.threshold = config.get('histogram', {}).get('threshold', 0.3)
        self.min_scene_length = config.get('histogram', {}).get('min_scene_length', 2.0)
        self.logger = logging.getLogger(__name__)
        
    def get_hist(self, img: np.ndarray, channel: int = 0) -> np.ndarray:
        """計算單通道直方圖"""
        hist = cv2.calcHist([img], [channel], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX, -1)
        return hist
    
    def hist_similar(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        """計算兩個直方圖的相似度"""
        assert len(hist1) == len(hist2)
        tmp = 0
        for h1, h2 in zip(hist1, hist2):
            h1 = max(0, h1[0]) if isinstance(h1, np.ndarray) else max(0, h1)
            h2 = max(0, h2[0]) if isinstance(h2, np.ndarray) else max(0, h2)
            tmp += 1 - (0 if (h1 == 0 and h2 == 0) else float(abs(h1 - h2)) / max(h1, h2))
        return tmp / len(hist1)
    
    def calculate_frame_difference(self, hist_dict1: Dict, hist_dict2: Dict) -> float:
        """計算兩個frame的直方圖差異"""
        total_diff = 0
        for channel in range(3):  # RGB三通道
            similarity = self.hist_similar(hist_dict1[channel], hist_dict2[channel])
            total_diff += (1 - similarity)  # 轉換為差異度
        return total_diff / 3  # 平均差異度
    
    def detect_scenes(self, video_path: str) -> List[Dict]:
        """
        檢測影片中的場景切換點
        
        Args:
            video_path (str): 影片路徑
            
        Returns:
            List[Dict]: 場景列表，每個場景包含開始和結束時間
        """
        self.logger.info(f"開始分析影片場景: {video_path}")
        
        # 獲取影片基本資訊
        video_info = get_video_info(video_path)
        fps = video_info['fps']
        total_frames = video_info['frame_count']
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"無法開啟影片檔案: {video_path}")
        
        # 存储每一帧的直方图
        hist_list = []
        frame_count = 0
        
        # 逐帧处理
        pbar = tqdm(total=total_frames, desc="分析影片幀")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 計算每個通道的直方圖
            hist_dict = {}
            for channel in range(3):
                hist_dict[channel] = self.get_hist(frame, channel)
            
            hist_list.append(hist_dict)
            frame_count += 1
            pbar.update(1)
            
        pbar.close()
        cap.release()
        
        self.logger.info(f"共處理 {frame_count} 幀")
        
        # 計算相鄰幀之間的差異
        differences = []
        for i in range(1, len(hist_list)):
            diff = self.calculate_frame_difference(hist_list[i-1], hist_list[i])
            differences.append(diff)
        
        # 找到場景切換點
        scene_changes = [0]  # 第一幀總是場景開始
        
        for i, diff in enumerate(differences):
            if diff > self.threshold:
                scene_changes.append(i + 1)  # i+1 是因為differences比hist_list少一個元素
        
        scene_changes.append(len(hist_list))  # 最後一幀
        
        # 生成場景列表
        scenes = []
        for i in range(len(scene_changes) - 1):
            start_frame = scene_changes[i]
            end_frame = scene_changes[i + 1]
            
            start_time = start_frame / fps
            end_time = end_frame / fps
            duration = end_time - start_time
            
            # 過濾太短的場景
            if duration >= self.min_scene_length:
                scene = {
                    'id': len(scenes),
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': duration,
                    'start_time_str': seconds_to_time(start_time),
                    'end_time_str': seconds_to_time(end_time)
                }
                scenes.append(scene)
        
        self.logger.info(f"檢測到 {len(scenes)} 個有效場景")
        
        return scenes