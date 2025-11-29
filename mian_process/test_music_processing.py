#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
音樂處理模組測試
"""

import sys
import os
from pathlib import Path
import json

# 添加專案根目錄到路徑
sys.path.append(str(Path(__file__).parent))

from utils import setup_logger, load_config, save_json
from music_processing import RhythmAnalyzer

def test_rhythm_analysis():
    """測試節拍分析功能"""
    print("=" * 60)
    print("測試節拍分析模組")
    print("=" * 60)
    
    # 載入配置
    config = load_config('config.yaml')
    logger = setup_logger('music_test', 'output/logs')
    
    # 尋找可用的音頻文件
    audio_files = [
        r"F:\work\video_analyze\data\test.mp3",
        r"F:\work\video_analyze\data\test.wav",
        r"F:\work\video_analyze\data\audio\test\test.mp3",
    ]
    
    test_audio = None
    for audio_file in audio_files:
        if Path(audio_file).exists():
            test_audio = audio_file
            break
    
    if test_audio is None:
        print("找不到可用的測試音頻文件")
        return False
    
    print(f"測試音頻: {Path(test_audio).name}")
    
    try:
        # 創建節拍分析器
        rhythm_analyzer = RhythmAnalyzer(config['music_processing'])
        
        # 執行節拍分析
        print("執行節拍分析...")
        rhythm_data = rhythm_analyzer.analyze(test_audio)
        
        # 儲存結果
        output_file = "output/music_analysis/rhythm_result.json"
        save_json(rhythm_data, output_file)
        
        print(f"✓ 節拍分析完成！")
        print(f"✓ 結果已儲存到: {output_file}")
        
        # 顯示分析摘要
        print(f"\n音樂分析摘要:")
        file_info = rhythm_data.get('file_info', {})
        tempo_info = rhythm_data.get('tempo', {})
        onset_info = rhythm_data.get('onset', {})
        
        print(f"  - 時長: {file_info.get('duration', 0):.2f} 秒")
        print(f"  - 採樣率: {file_info.get('sample_rate', 0)} Hz")
        print(f"  - BPM: {tempo_info.get('tempo', 0):.1f}")
        print(f"  - 檢測到的節拍點: {len(tempo_info.get('beats', []))}")
        print(f"  - 檢測到的onset: {onset_info.get('onset_count', 0)}")
        
        return True
        
    except Exception as e:
        print(f"✗ 節拍分析失敗: {e}")
        return False

def test_cut_points_generation():
    """測試剪輯點生成功能"""
    print("\n" + "=" * 60)
    print("測試剪輯點生成")
    print("=" * 60)
    
    # 載入之前的節拍分析結果
    rhythm_file = "output/music_analysis/rhythm_result.json"
    if not Path(rhythm_file).exists():
        print("需要先執行節拍分析測試")
        return False
    
    config = load_config('config.yaml')
    
    with open(rhythm_file, 'r', encoding='utf-8') as f:
        rhythm_data = json.load(f)
    
    try:
        # 創建節拍分析器
        rhythm_analyzer = RhythmAnalyzer(config['music_processing'])
        
        # 獲取節拍時間點
        beat_times = rhythm_data.get('tempo', {}).get('beat_times', [])
        video_duration = rhythm_data.get('file_info', {}).get('duration', 60.0)
        
        print(f"基於 {len(beat_times)} 個節拍點生成剪輯片段...")
        
        # 生成剪輯片段
        segments = rhythm_analyzer.get_cut_points_from_beats(
            beat_times, video_duration, target_segments=8
        )
        
        # 儲存剪輯點結果
        cut_points_data = {
            'segments': segments,
            'total_segments': len(segments),
            'original_duration': video_duration,
            'beat_count': len(beat_times)
        }
        
        output_file = "output/music_analysis/cut_points.json"
        save_json(cut_points_data, output_file)
        
        print(f"✓ 剪輯點生成完成！生成了 {len(segments)} 個片段")
        print(f"✓ 結果已儲存到: {output_file}")
        
        # 顯示前5個片段
        print(f"\n前5個剪輯片段:")
        for i, (start, end) in enumerate(segments[:5]):
            duration = end - start
            print(f"  片段 {i+1}: {start:.2f}s - {end:.2f}s (時長: {duration:.2f}s)")
        
        return True
        
    except Exception as e:
        print(f"✗ 剪輯點生成失敗: {e}")
        return False

def main():
    """執行音樂處理測試"""
    print("開始音樂處理模組測試...")
    
    # 測試節拍分析
    if not test_rhythm_analysis():
        return False
    
    # 測試剪輯點生成
    if not test_cut_points_generation():
        return False
    
    print("\n" + "=" * 60)
    print("🎵 音樂處理模組測試完成！")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)