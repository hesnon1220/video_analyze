#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
特徵提取模組 - 使用YOLO進行物體檢測和場景分析
"""

import cv2
import numpy as np
from typing import List, Dict, Any
import logging
from pathlib import Path

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("YOLO未安裝，將使用基本特徵提取")

from utils.common import get_video_info

class FeatureExtractor:
    """影像特徵提取器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.yolo_config = config.get('yolo', {})
        self.confidence = self.yolo_config.get('confidence', 0.5)
        self.device = self.yolo_config.get('device', 'cpu')
        self.logger = logging.getLogger(__name__)
        
        # 初始化YOLO模型
        if YOLO_AVAILABLE:
            try:
                model_path = config.get('models', {}).get('yolo_model', 'yolov8n.pt')
                self.model = YOLO(model_path)
                self.logger.info(f"YOLO模型載入成功: {model_path}")
            except Exception as e:
                self.logger.error(f"YOLO模型載入失敗: {e}")
                self.model = None
        else:
            self.model = None
    
    def extract_basic_features(self, frame: np.ndarray) -> Dict[str, Any]:
        """提取基本影像特徵"""
        # 計算顏色直方圖
        hist_features = {}
        for i, color in enumerate(['blue', 'green', 'red']):
            hist = cv2.calcHist([frame], [i], None, [256], [0, 256])
            hist_features[f'{color}_hist_mean'] = np.mean(hist)
            hist_features[f'{color}_hist_std'] = np.std(hist)
        
        # 計算亮度特徵
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        # 計算邊緣特徵
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        return {
            'brightness': brightness,
            'contrast': contrast,
            'edge_density': edge_density,
            **hist_features
        }
    
    def extract_yolo_features(self, frame: np.ndarray) -> Dict[str, Any]:
        """使用YOLO提取物體檢測特徵"""
        if self.model is None:
            return {}
        
        try:
            results = self.model(frame, conf=self.confidence, device=self.device, verbose=False)
            
            detected_objects = []
            object_counts = {}
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # 獲取類別名稱
                        class_id = int(box.cls[0])
                        class_name = self.model.names[class_id]
                        confidence = float(box.conf[0])
                        
                        # 獲取邊界框
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        detected_objects.append({
                            'class': class_name,
                            'confidence': confidence,
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'area': float((x2 - x1) * (y2 - y1))
                        })
                        
                        # 統計物體類別
                        object_counts[class_name] = object_counts.get(class_name, 0) + 1
            
            return {
                'detected_objects': detected_objects,
                'object_counts': object_counts,
                'total_objects': len(detected_objects)
            }
            
        except Exception as e:
            self.logger.error(f"YOLO檢測失敗: {e}")
            return {}
    
    def analyze_scene_content(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """分析場景內容和情境"""
        analysis = {
            'scene_type': 'unknown',
            'activity_level': 'low',
            'dominant_objects': [],
            'scene_score': 0.0
        }
        
        # 基於檢測到的物體分析場景類型
        if 'object_counts' in features:
            object_counts = features['object_counts']
            
            # 根據物體類型判斷場景
            if any(obj in object_counts for obj in ['person', 'chair', 'tv', 'sofa']):
                analysis['scene_type'] = 'indoor'
            elif any(obj in object_counts for obj in ['car', 'truck', 'tree', 'building']):
                analysis['scene_type'] = 'outdoor'
            elif any(obj in object_counts for obj in ['person']):
                analysis['scene_type'] = 'character_focus'
            
            # 判斷活動程度
            total_objects = features.get('total_objects', 0)
            if total_objects > 5:
                analysis['activity_level'] = 'high'
            elif total_objects > 2:
                analysis['activity_level'] = 'medium'
            
            # 找出主要物體
            if object_counts:
                sorted_objects = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)
                analysis['dominant_objects'] = [obj[0] for obj in sorted_objects[:3]]
        
        # 基於基本特徵計算場景評分
        if 'brightness' in features and 'contrast' in features and 'edge_density' in features:
            # 正規化評分 (0-1)
            brightness_score = min(features['brightness'] / 255.0, 1.0)
            contrast_score = min(features['contrast'] / 100.0, 1.0)
            edge_score = min(features['edge_density'] * 10, 1.0)
            
            analysis['scene_score'] = (brightness_score + contrast_score + edge_score) / 3
        
        return analysis
    
    def extract_features(self, video_path: str, scenes: List[Dict]) -> List[Dict]:
        """
        為每個場景提取特徵
        
        Args:
            video_path (str): 影片路徑
            scenes (List[Dict]): 場景列表
            
        Returns:
            List[Dict]: 包含特徵的場景列表
        """
        self.logger.info(f"開始提取影片特徵: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"無法開啟影片檔案: {video_path}")
        
        video_info = get_video_info(video_path)
        fps = video_info['fps']
        
        enhanced_scenes = []
        
        for scene in scenes:
            self.logger.info(f"處理場景 {scene['id']}: {scene['start_time_str']} - {scene['end_time_str']}")
            
            # 取場景中間的幀作為代表幀
            middle_frame = (scene['start_frame'] + scene['end_frame']) // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
            ret, frame = cap.read()
            
            if not ret:
                self.logger.warning(f"無法讀取場景 {scene['id']} 的代表幀")
                enhanced_scenes.append(scene)
                continue
            
            # 提取基本特徵
            basic_features = self.extract_basic_features(frame)
            
            # 提取YOLO特徵
            yolo_features = self.extract_yolo_features(frame)
            
            # 合併所有特徵
            all_features = {**basic_features, **yolo_features}
            
            # 分析場景内容
            scene_analysis = self.analyze_scene_content(all_features)
            
            # 創建增強的場景資料
            enhanced_scene = {
                **scene,
                'features': all_features,
                'analysis': scene_analysis,
                'representative_frame': middle_frame
            }
            
            enhanced_scenes.append(enhanced_scene)
        
        cap.release()
        self.logger.info(f"特徵提取完成，處理了 {len(enhanced_scenes)} 個場景")
        
        return enhanced_scenes