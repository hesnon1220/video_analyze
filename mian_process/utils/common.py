#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
通用工具函數
"""

import os
import json
import yaml
import cv2  # 添加 cv2 導入
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple

def load_config(config_path: str) -> Dict[str, Any]:
    """載入YAML配置檔案"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_json(data: Any, file_path: str) -> None:
    """儲存資料為JSON格式"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(file_path: str) -> Any:
    """載入JSON檔案"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def ensure_dir(dir_path: str) -> None:
    """確保目錄存在"""
    Path(dir_path).mkdir(parents=True, exist_ok=True)

def get_video_info(video_path: str) -> Dict[str, Any]:
    """獲取影片基本資訊"""
    import cv2
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"無法開啟影片檔案: {video_path}")
    
    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    }
    
    cap.release()
    return info

def time_to_seconds(time_str: str) -> float:
    """將時間字串轉換為秒數 (格式: MM:SS 或 HH:MM:SS)"""
    parts = time_str.split(':')
    if len(parts) == 2:  # MM:SS
        return int(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 3:  # HH:MM:SS
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    else:
        raise ValueError(f"不支援的時間格式: {time_str}")

def seconds_to_time(seconds: float) -> str:
    """將秒數轉換為時間字串 (格式: HH:MM:SS)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

def calculate_histogram_difference(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """計算兩個直方圖的差異度"""
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)