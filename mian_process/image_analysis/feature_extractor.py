#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
特徵提取模組 - 使用GPU加速YOLO進行物體檢測和場景分析
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
from utils.hardware_manager import HardwareManager

class FeatureExtractor:
    """GPU加速影像特徵提取器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.yolo_config = config.get('yolo', {})
        self.confidence = self.yolo_config.get('confidence', 0.4)
        self.iou_threshold = self.yolo_config.get('iou_threshold', 0.5)
        self.max_detections = self.yolo_config.get('max_detections', 300)
        self.target_classes = self.yolo_config.get('target_classes', None)
        self.half_precision = self.yolo_config.get('half_precision', True)
        
        self.logger = logging.getLogger(__name__)
        
        # 初始化硬體管理器
        self.hw_manager = HardwareManager(config)
        self.device = self.hw_manager.device
        
        # 初始化YOLO模型
        if YOLO_AVAILABLE:
            try:
                model_size = self.yolo_config.get('model_size', 'n')
                model_path = f"../models/yolov8{model_size}.pt"
                
                self.model = YOLO(model_path)
                
                # 配置模型設備 - 修復半精度問題
                if self.device == 'cuda':
                    self.model.to(self.device)
                    # 注意：某些YOLO版本不支持.half()方法，改為在推論時使用half參數
                    try:
                        if self.half_precision:
                            # 嘗試設置半精度，如果失敗則忽略
                            if hasattr(self.model.model, 'half'):
                                self.model.model.half()
                    except Exception as half_error:
                        self.logger.warning(f"半精度設置失敗，將在推論時使用: {half_error}")
                        # 禁用半精度標誌，改為在推論時處理
                        self.half_precision = False
                
                self.logger.info(f"YOLO模型載入成功: {model_path} (設備: {self.device})")
            except Exception as e:
                self.logger.error(f"YOLO模型載入失敗: {e}")
                self.model = None
        else:
            self.model = None
    
    def extract_basic_features(self, frame: np.ndarray) -> Dict[str, Any]:
        """提取基本影像特徵（優化版本）"""
        h, w = frame.shape[:2]
        
        # 計算顏色直方圖（優化）
        hist_features = {}
        for i, color in enumerate(['blue', 'green', 'red']):
            hist = cv2.calcHist([frame], [i], None, [64], [0, 256])  # 減少bin數量加速
            hist_normalized = hist.flatten() / (h * w)
            hist_features[f'{color}_hist_mean'] = np.mean(hist_normalized)
            hist_features[f'{color}_hist_std'] = np.std(hist_normalized)
            hist_features[f'{color}_hist_peak'] = np.max(hist_normalized)
        
        # 計算亮度和對比度特徵
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        # 檢測模糊程度
        blur_threshold = self.config.get('histogram', {}).get('blur_threshold', 100)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_blurry = blur_score < blur_threshold
        
        # 計算邊緣密度（優化）
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # 計算色彩豐富度
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation_mean = np.mean(hsv[:,:,1])
        
        return {
            'brightness': float(brightness),
            'contrast': float(contrast),
            'edge_density': float(edge_density),
            'blur_score': float(blur_score),
            'is_blurry': bool(is_blurry),
            'saturation': float(saturation_mean),
            'resolution': [w, h],
            **hist_features
        }
    
    def extract_yolo_features(self, frame: np.ndarray) -> Dict[str, Any]:
        """使用GPU加速YOLO提取物體檢測特徵"""
        if self.model is None:
            return {}
        
        try:
            # GPU加速推論
            results = self.model(
                frame, 
                conf=self.confidence,
                iou=self.iou_threshold,
                max_det=self.max_detections,
                classes=self.target_classes,
                device=self.device,
                half=self.half_precision and self.device == 'cuda',
                verbose=False
            )
            
            detected_objects = []
            object_counts = {}
            total_area = 0
            person_count = 0
            
            h, w = frame.shape[:2]
            frame_area = h * w
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # 獲取類別資訊
                        class_id = int(box.cls[0])
                        class_name = self.model.names[class_id]
                        confidence = float(box.conf[0])
                        
                        # 獲取邊界框
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        bbox_area = float((x2 - x1) * (y2 - y1))
                        
                        detected_objects.append({
                            'class': class_name,
                            'confidence': confidence,
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'area': bbox_area,
                            'area_ratio': bbox_area / frame_area,
                            'center': [float((x1 + x2) / 2), float((y1 + y2) / 2)]
                        })
                        
                        # 統計計算
                        object_counts[class_name] = object_counts.get(class_name, 0) + 1
                        total_area += bbox_area
                        
                        if class_name == 'person':
                            person_count += 1
            
            # 計算場景分析指標
            object_density = len(detected_objects) / frame_area * 1000000  # 每百萬像素的物體數
            coverage_ratio = total_area / frame_area if frame_area > 0 else 0
            
            return {
                'detected_objects': detected_objects,
                'object_counts': object_counts,
                'total_objects': len(detected_objects),
                'person_count': person_count,
                'object_density': float(object_density),
                'coverage_ratio': float(coverage_ratio),
                'dominant_class': max(object_counts, key=object_counts.get) if object_counts else None
            }
            
        except Exception as e:
            self.logger.error(f"YOLO檢測失敗: {e}")
            return {}
    
    def analyze_scene_content(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """智能場景內容分析"""
        analysis = {
            'scene_type': 'unknown',
            'activity_level': 'low',
            'visual_appeal': 'low',
            'editing_priority': 0.0,
            'recommended_duration': 3.0,
            'transition_type': 'fade'
        }
        
        # 基於物體檢測的場景分類
        if 'object_counts' in features and features['object_counts']:
            object_counts = features['object_counts']
            person_count = features.get('person_count', 0)
            total_objects = features.get('total_objects', 0)
            
            # 場景類型判斷
            indoor_objects = sum(object_counts.get(obj, 0) for obj in 
                               ['chair', 'sofa', 'bed', 'tv', 'laptop', 'book', 'cup'])
            outdoor_objects = sum(object_counts.get(obj, 0) for obj in 
                                ['car', 'truck', 'tree', 'building', 'street sign'])
            
            if person_count >= 2:
                analysis['scene_type'] = 'group_scene'
                analysis['recommended_duration'] = 4.0
            elif person_count == 1:
                analysis['scene_type'] = 'character_focus'
                analysis['recommended_duration'] = 3.5
            elif indoor_objects > outdoor_objects:
                analysis['scene_type'] = 'indoor'
            elif outdoor_objects > 0:
                analysis['scene_type'] = 'outdoor'
            
            # 活動程度判斷
            density = features.get('object_density', 0)
            if density > 10 or total_objects > 8:
                analysis['activity_level'] = 'high'
                analysis['transition_type'] = 'slide'
            elif density > 5 or total_objects > 4:
                analysis['activity_level'] = 'medium'
                analysis['transition_type'] = 'dissolve'
        
        # 視覺吸引力評估
        visual_score = 0
        
        # 基於基本特徵的評分
        if 'brightness' in features:
            # 適中的亮度更有吸引力
            brightness_norm = features['brightness'] / 255.0
            if 0.3 <= brightness_norm <= 0.7:
                visual_score += 0.2
        
        if 'contrast' in features:
            # 適中的對比度
            if 30 <= features['contrast'] <= 80:
                visual_score += 0.2
        
        if 'saturation' in features:
            # 較高的飽和度更有吸引力
            saturation_norm = features['saturation'] / 255.0
            visual_score += saturation_norm * 0.2
        
        if not features.get('is_blurry', True):
            visual_score += 0.2
        
        if features.get('edge_density', 0) > 0.1:
            visual_score += 0.2
        
        # 物體檢測加分
        person_count = features.get('person_count', 0)
        if person_count > 0:
            visual_score += min(person_count * 0.1, 0.3)
        
        # 確定視覺吸引力等級
        if visual_score >= 0.7:
            analysis['visual_appeal'] = 'high'
            analysis['editing_priority'] = 0.8 + visual_score * 0.2
        elif visual_score >= 0.4:
            analysis['visual_appeal'] = 'medium'
            analysis['editing_priority'] = 0.4 + visual_score * 0.4
        else:
            analysis['visual_appeal'] = 'low'
            analysis['editing_priority'] = visual_score * 0.4
        
        return analysis
    
    def extract_features_batch(self, frames: List[np.ndarray]) -> List[Dict[str, Any]]:
        """批次處理多個幀（GPU優化）"""
        if not frames:
            return []
        
        all_features = []
        
        if self.model is not None and len(frames) > 1:
            try:
                # 批次YOLO推論
                results_batch = self.model(
                    frames,
                    conf=self.confidence,
                    iou=self.iou_threshold,
                    max_det=self.max_detections,
                    classes=self.target_classes,
                    device=self.device,
                    half=self.half_precision and self.device == 'cuda',
                    verbose=False
                )
                
                for i, (frame, result) in enumerate(zip(frames, results_batch)):
                    basic_features = self.extract_basic_features(frame)
                    
                    # 處理批次YOLO結果
                    yolo_features = self._process_yolo_result(result, frame.shape)
                    
                    all_features.append({**basic_features, **yolo_features})
                    
            except Exception as e:
                self.logger.warning(f"批次處理失敗，回退到單幀處理: {e}")
                # 回退到單幀處理
                for frame in frames:
                    basic_features = self.extract_basic_features(frame)
                    yolo_features = self.extract_yolo_features(frame)
                    all_features.append({**basic_features, **yolo_features})
        else:
            # 單幀處理
            for frame in frames:
                basic_features = self.extract_basic_features(frame)
                yolo_features = self.extract_yolo_features(frame)
                all_features.append({**basic_features, **yolo_features})
        
        return all_features
    
    def _process_yolo_result(self, result, frame_shape) -> Dict[str, Any]:
        """處理單個YOLO結果"""
        h, w = frame_shape[:2]
        frame_area = h * w
        
        detected_objects = []
        object_counts = {}
        total_area = 0
        person_count = 0
        
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                confidence = float(box.conf[0])
                
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                bbox_area = float((x2 - x1) * (y2 - y1))
                
                detected_objects.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'area': bbox_area,
                    'area_ratio': bbox_area / frame_area,
                    'center': [float((x1 + x2) / 2), float((y1 + y2) / 2)]
                })
                
                object_counts[class_name] = object_counts.get(class_name, 0) + 1
                total_area += bbox_area
                
                if class_name == 'person':
                    person_count += 1
        
        object_density = len(detected_objects) / frame_area * 1000000
        coverage_ratio = total_area / frame_area if frame_area > 0 else 0
        
        return {
            'detected_objects': detected_objects,
            'object_counts': object_counts,
            'total_objects': len(detected_objects),
            'person_count': person_count,
            'object_density': float(object_density),
            'coverage_ratio': float(coverage_ratio),
            'dominant_class': max(object_counts, key=object_counts.get) if object_counts else None
        }
    
    def extract_features(self, video_path: str, scenes: List[Dict]) -> List[Dict]:
        """
        為每個場景提取特徵（GPU加速版本）
        
        Args:
            video_path (str): 影片路徑
            scenes (List[Dict]): 場景列表
            
        Returns:
            List[Dict]: 包含特徵的場景列表
        """
        self.logger.info(f"開始GPU加速特徵提取: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"無法開啟影片檔案: {video_path}")
        
        video_info = get_video_info(video_path)
        fps = video_info['fps']
        
        enhanced_scenes = []
        
        # 批次處理配置
        batch_size = self.config.get('performance', {}).get('batch_size', 8)
        frames_batch = []
        scenes_batch = []
        
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
            
            frames_batch.append(frame)
            scenes_batch.append(scene)
            
            # 當批次滿了或者是最後一個場景時處理
            if len(frames_batch) >= batch_size or scene == scenes[-1]:
                # 批次提取特徵
                features_batch = self.extract_features_batch(frames_batch)
                
                # 處理每個場景
                for scene_data, features in zip(scenes_batch, features_batch):
                    # 分析場景內容
                    scene_analysis = self.analyze_scene_content(features)
                    
                    # 創建增強的場景資料
                    enhanced_scene = {
                        **scene_data,
                        'features': features,
                        'analysis': scene_analysis,
                        'representative_frame': (scene_data['start_frame'] + scene_data['end_frame']) // 2
                    }
                    
                    enhanced_scenes.append(enhanced_scene)
                
                # 清空批次
                frames_batch = []
                scenes_batch = []
        
        cap.release()
        self.logger.info(f"GPU加速特徵提取完成，處理了 {len(enhanced_scenes)} 個場景")
        
        return enhanced_scenes