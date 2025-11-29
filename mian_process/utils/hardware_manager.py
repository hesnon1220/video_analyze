#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
硬體加速和模型管理工具
提供GPU檢測、YOLO模型下載、demucs模型管理功能
"""

import torch
import os
import logging
from pathlib import Path
import urllib.request
from typing import Optional, Dict, Any
import yaml

logger = logging.getLogger(__name__)

class HardwareManager:
    """硬體管理器 - 處理GPU檢測和設定"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = self._detect_device()
        self._setup_gpu_settings()
    
    def _detect_device(self) -> str:
        """自動檢測最佳設備"""
        device_setting = self.config.get('hardware', {}).get('device', 'auto')
        
        if device_setting == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info(f"檢測到GPU: {gpu_name}, 記憶體: {gpu_memory:.1f}GB")
            else:
                device = 'cpu'
                logger.info("未檢測到GPU，使用CPU")
        else:
            device = device_setting
            
        logger.info(f"使用設備: {device}")
        return device
    
    def _setup_gpu_settings(self):
        """設定GPU相關參數"""
        if self.device == 'cuda':
            # 設定GPU記憶體使用比例
            memory_fraction = self.config.get('hardware', {}).get('gpu_memory_fraction', 0.8)
            if memory_fraction < 1.0:
                torch.cuda.set_per_process_memory_fraction(memory_fraction)
            
            # 啟用cudnn優化
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            
            logger.info(f"GPU記憶體使用限制: {memory_fraction*100}%")
    
    def get_device_info(self) -> Dict[str, Any]:
        """獲取設備資訊"""
        info = {
            'device': self.device,
            'torch_version': torch.__version__,
        }
        
        if self.device == 'cuda':
            info.update({
                'cuda_version': torch.version.cuda,
                'gpu_count': torch.cuda.device_count(),
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory_total': torch.cuda.get_device_properties(0).total_memory / (1024**3),
                'gpu_memory_allocated': torch.cuda.memory_allocated(0) / (1024**3),
            })
        
        return info

class ModelManager:
    """模型管理器 - 處理YOLO和demucs模型下載與管理"""
    
    YOLO_MODELS = {
        'yolov8n.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
        'yolov8s.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt',
        'yolov8m.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt',
        'yolov8n-seg.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt',
    }
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_dir = Path("../models")
        self.model_dir.mkdir(exist_ok=True)
    
    def download_yolo_model(self, model_name: str = 'yolov8n.pt') -> Path:
        """下載YOLO模型"""
        model_path = self.model_dir / model_name
        
        if model_path.exists():
            logger.info(f"模型已存在: {model_path}")
            return model_path
        
        if model_name not in self.YOLO_MODELS:
            raise ValueError(f"不支援的模型: {model_name}")
        
        url = self.YOLO_MODELS[model_name]
        logger.info(f"下載YOLO模型: {model_name}")
        
        try:
            urllib.request.urlretrieve(url, model_path)
            logger.info(f"模型下載完成: {model_path}")
        except Exception as e:
            logger.error(f"模型下載失敗: {e}")
            raise
        
        return model_path
    
    def setup_demucs_models(self) -> bool:
        """設定demucs模型"""
        try:
            # demucs模型會在首次使用時自動下載
            import demucs.pretrained
            
            model_name = self.config.get('music_processing', {}).get('demucs', {}).get('model', 'htdemucs')
            
            # 預載入模型以觸發下載
            logger.info(f"設定demucs模型: {model_name}")
            model = demucs.pretrained.get_model(model_name)
            
            logger.info("demucs模型設定完成")
            return True
            
        except Exception as e:
            logger.error(f"demucs模型設定失敗: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """獲取模型資訊"""
        info = {
            'model_directory': str(self.model_dir),
            'yolo_models_available': [],
            'demucs_ready': False
        }
        
        # 檢查YOLO模型
        for model_name in self.YOLO_MODELS.keys():
            model_path = self.model_dir / model_name
            if model_path.exists():
                info['yolo_models_available'].append({
                    'name': model_name,
                    'path': str(model_path),
                    'size_mb': model_path.stat().st_size / (1024*1024)
                })
        
        # 檢查demucs
        try:
            import demucs
            info['demucs_ready'] = True
            info['demucs_version'] = demucs.__version__
        except ImportError:
            info['demucs_ready'] = False
        
        return info

def initialize_hardware_and_models(config_path: str = 'config.yaml'):
    """初始化硬體和模型"""
    # 載入配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 初始化硬體管理器
    hardware_manager = HardwareManager(config)
    
    # 初始化模型管理器
    model_manager = ModelManager(config)
    
    # 下載必要的YOLO模型
    yolo_config = config.get('image_analysis', {}).get('yolo', {})
    model_size = yolo_config.get('model_size', 'n')
    
    try:
        # 下載主要檢測模型
        main_model = f'yolov8{model_size}.pt'
        model_manager.download_yolo_model(main_model)
        
        # 下載分割模型（如果需要）
        seg_model = f'yolov8{model_size}-seg.pt'
        if seg_model in model_manager.YOLO_MODELS:
            model_manager.download_yolo_model(seg_model)
        
        logger.info("YOLO模型設定完成")
        
    except Exception as e:
        logger.warning(f"YOLO模型設定警告: {e}")
    
    # 設定demucs模型
    try:
        model_manager.setup_demucs_models()
    except Exception as e:
        logger.warning(f"demucs模型設定警告: {e}")
    
    return hardware_manager, model_manager

if __name__ == "__main__":
    # 設定日誌
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 初始化
    hw_manager, model_manager = initialize_hardware_and_models()
    
    # 顯示系統資訊
    print("\n=== 硬體資訊 ===")
    hw_info = hw_manager.get_device_info()
    for key, value in hw_info.items():
        print(f"{key}: {value}")
    
    print("\n=== 模型資訊 ===")
    model_info = model_manager.get_model_info()
    for key, value in model_info.items():
        print(f"{key}: {value}")