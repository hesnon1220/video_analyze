#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
系統初始化和優化測試腳本
自動下載模型、設置GPU加速、測試所有優化功能
"""

import sys
import os
from pathlib import Path
import logging
import time
import json

# 添加專案根目錄到路徑
sys.path.append(str(Path(__file__).parent))

from utils import setup_logger, load_config, save_json
from utils.hardware_manager import initialize_hardware_and_models

def setup_system():
    """完整系統設置"""
    print("🚀 開始系統初始化和優化設置...")
    
    # 設定日誌
    logger = setup_logger('system_setup', level=logging.INFO)
    
    # 1. 初始化硬體和模型
    print("\n" + "=" * 60)
    print("步驟1: 初始化硬體和下載模型")
    print("=" * 60)
    
    try:
        hw_manager, model_manager = initialize_hardware_and_models()
        print("✅ 硬體和模型初始化完成")
        
        # 顯示硬體資訊
        hw_info = hw_manager.get_device_info()
        print(f"🖥️  使用設備: {hw_info['device']}")
        if hw_info['device'] == 'cuda':
            print(f"   GPU: {hw_info.get('gpu_name', 'Unknown')}")
            print(f"   GPU記憶體: {hw_info.get('gpu_memory_total', 0):.1f}GB")
        
        # 顯示模型資訊
        model_info = model_manager.get_model_info()
        print(f"📁 模型目錄: {model_info['model_directory']}")
        print(f"🎯 可用YOLO模型: {len(model_info['yolo_models_available'])}")
        for model in model_info['yolo_models_available']:
            print(f"   - {model['name']} ({model['size_mb']:.1f}MB)")
        print(f"🎵 Demucs準備就緒: {'✅' if model_info['demucs_ready'] else '❌'}")
        
    except Exception as e:
        print(f"❌ 硬體和模型初始化失敗: {e}")
        return False
    
    # 2. 測試GPU加速YOLO
    print("\n" + "=" * 60)
    print("步驟2: 測試GPU加速YOLO檢測")
    print("=" * 60)
    
    try:
        config = load_config('config.yaml')
        from image_analysis import FeatureExtractor
        
        feature_extractor = FeatureExtractor(config['image_analysis'])
        
        # 找測試影片
        test_videos = [
            r"F:\work\video_analyze\test_1.mp4",
            r"F:\work\video_analyze\test.mp4"
        ]
        
        test_video = None
        for video_path in test_videos:
            if Path(video_path).exists():
                test_video = video_path
                break
        
        if test_video:
            print(f"🎬 使用測試影片: {Path(test_video).name}")
            
            # 測試基本功能
            import cv2
            cap = cv2.VideoCapture(test_video)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                print("⏱️  執行YOLO檢測測試...")
                start_time = time.time()
                
                # 測試特徵提取
                basic_features = feature_extractor.extract_basic_features(frame)
                yolo_features = feature_extractor.extract_yolo_features(frame)
                
                end_time = time.time()
                
                print(f"✅ YOLO檢測完成，耗時: {end_time - start_time:.2f}秒")
                print(f"   檢測到物體: {yolo_features.get('total_objects', 0)}個")
                print(f"   人物數量: {yolo_features.get('person_count', 0)}人")
                print(f"   使用設備: {feature_extractor.device}")
                
                if yolo_features.get('object_counts'):
                    top_objects = sorted(yolo_features['object_counts'].items(), 
                                       key=lambda x: x[1], reverse=True)[:3]
                    print(f"   主要物體: {', '.join([f'{obj}({count})' for obj, count in top_objects])}")
            else:
                print("⚠️  無法讀取測試影片幀")
        else:
            print("⚠️  未找到測試影片，跳過YOLO測試")
        
    except Exception as e:
        print(f"❌ YOLO測試失敗: {e}")
    
    # 3. 測試GPU加速音源分離
    print("\n" + "=" * 60)
    print("步驟3: 測試GPU加速音源分離")
    print("=" * 60)
    
    try:
        from music_processing import AudioSeparator
        
        audio_separator = AudioSeparator(config['music_processing'])
        
        # 找測試音頻
        test_audios = [
            r"F:\work\video_analyze\data\test.mp3",
            r"F:\work\video_analyze\data\test.wav"
        ]
        
        test_audio = None
        for audio_path in test_audios:
            if Path(audio_path).exists():
                test_audio = audio_path
                break
        
        if test_audio:
            print(f"🎵 使用測試音頻: {Path(test_audio).name}")
            print(f"   使用設備: {audio_separator.device}")
            print(f"   模型: {audio_separator.model_name}")
            
            # 測試分離（使用短音頻避免耗時過長）
            if audio_separator.model is not None:
                print("⏱️  執行音源分離測試...")
                start_time = time.time()
                
                # 簡單測試模型載入
                print("✅ Demucs模型載入成功")
                print("   (完整分離測試需要較長時間，已跳過)")
                
            else:
                print("⚠️  Demucs API不可用，將使用命令列方式")
        else:
            print("⚠️  未找到測試音頻，跳過音源分離測試")
    
    except Exception as e:
        print(f"❌ 音源分離測試失敗: {e}")
    
    # 4. 性能優化建議
    print("\n" + "=" * 60)
    print("步驟4: 性能優化建議")
    print("=" * 60)
    
    recommendations = []
    
    # GPU建議
    if hw_info['device'] == 'cuda':
        gpu_memory = hw_info.get('gpu_memory_total', 0)
        if gpu_memory > 8:
            recommendations.append("✅ GPU記憶體充足，可以使用較大的YOLO模型 (yolov8m 或 yolov8l)")
            recommendations.append("✅ 可以增加batch_size到16以提升處理速度")
        elif gpu_memory > 4:
            recommendations.append("✅ GPU記憶體適中，建議使用 yolov8s 模型")
            recommendations.append("✅ batch_size設為8-12較為合適")
        else:
            recommendations.append("⚠️  GPU記憶體較小，建議使用 yolov8n 模型")
            recommendations.append("⚠️  batch_size建議設為4-6")
    else:
        recommendations.append("⚠️  未檢測到GPU，處理速度會較慢")
        recommendations.append("💡 建議安裝CUDA以獲得GPU加速")
        recommendations.append("💡 CPU模式下建議降低影片解析度以提升速度")
    
    # 模型建議
    if len(model_info['yolo_models_available']) == 0:
        recommendations.append("❌ 未找到YOLO模型，請檢查網路連線")
    
    if not model_info['demucs_ready']:
        recommendations.append("❌ Demucs未就緒，請檢查套件安裝")
    
    # 配置建議
    recommendations.append("💡 建議根據影片類型調整confidence閾值")
    recommendations.append("💡 人物較多的影片可設confidence=0.3")
    recommendations.append("💡 風景影片可設confidence=0.5")
    
    print("📊 性能優化建議:")
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
    
    # 5. 生成配置摘要
    print("\n" + "=" * 60)
    print("步驟5: 生成系統配置摘要")
    print("=" * 60)
    
    system_summary = {
        "setup_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": hw_info,
        "models": model_info,
        "optimization_status": {
            "gpu_acceleration": hw_info['device'] == 'cuda',
            "yolo_ready": len(model_info['yolo_models_available']) > 0,
            "demucs_ready": model_info['demucs_ready'],
            "batch_processing": True
        },
        "recommendations": recommendations,
        "config_optimizations": {
            "suggested_model_size": "s" if hw_info['device'] == 'cuda' else "n",
            "suggested_batch_size": 8 if hw_info['device'] == 'cuda' else 4,
            "suggested_confidence": 0.4,
            "suggested_gpu_memory_fraction": 0.8
        }
    }
    
    # 儲存配置摘要
    save_json(system_summary, "output/system_setup_summary.json")
    print("✅ 系統配置摘要已儲存至: output/system_setup_summary.json")
    
    # 6. 更新配置檔案建議
    print("\n" + "=" * 60)
    print("步驟6: 配置優化建議")
    print("=" * 60)
    
    config_updates = {}
    
    if hw_info['device'] == 'cuda':
        config_updates.update({
            'hardware.device': 'cuda',
            'image_analysis.yolo.device': 'cuda',
            'music_processing.demucs.device': 'cuda',
            'performance.batch_size': 8 if hw_info.get('gpu_memory_total', 0) > 6 else 4
        })
    
    if config_updates:
        print("建議的配置更新:")
        for key, value in config_updates.items():
            print(f"   {key}: {value}")
    
    print("\n🎉 系統初始化和優化設置完成！")
    print("\n下一步:")
    print("1. 執行 test_end_to_end.py 進行完整測試")
    print("2. 根據建議調整 config.yaml 配置")
    print("3. 開始使用 main.py 處理您的影片")
    
    return True

if __name__ == "__main__":
    success = setup_system()
    if not success:
        sys.exit(1)