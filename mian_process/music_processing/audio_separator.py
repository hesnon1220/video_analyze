#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPU加速音源分離模組 - 使用demucs分離人聲、鼓聲、貝斯和其他樂器
"""

import os
import subprocess
from pathlib import Path
from typing import Dict, Optional, List
import logging
import torch
import librosa
import numpy as np

try:
    import demucs.api
    import demucs.pretrained
    DEMUCS_AVAILABLE = True
except ImportError:
    DEMUCS_AVAILABLE = False

from utils.hardware_manager import HardwareManager

class AudioSeparator:
    """GPU加速音源分離器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.demucs_config = config.get('demucs', {})
        self.model_name = self.demucs_config.get('model', 'htdemucs')
        self.shifts = self.demucs_config.get('shifts', 1)
        self.overlap = self.demucs_config.get('overlap', 0.25)
        self.split = self.demucs_config.get('split', True)
        self.segment_length = self.demucs_config.get('segment_length', 10)
        self.output_format = self.demucs_config.get('output_format', 'wav')
        self.bitrate = self.demucs_config.get('bitrate', 320)
        self.stems = self.demucs_config.get('stems', ['vocals', 'drums', 'bass', 'other'])
        
        self.logger = logging.getLogger(__name__)
        
        # 初始化硬體管理器
        self.hw_manager = HardwareManager(config)
        self.device = self.hw_manager.device
        
        # 初始化demucs模型
        self.model = None
        if DEMUCS_AVAILABLE:
            try:
                self.model = demucs.pretrained.get_model(self.model_name)
                if self.device == 'cuda':
                    self.model.cuda()
                self.logger.info(f"Demucs模型載入成功: {self.model_name} (設備: {self.device})")
            except Exception as e:
                self.logger.error(f"Demucs模型載入失敗: {e}")
                self.model = None
        else:
            self.logger.warning("demucs未安裝，將使用命令列方式或跳過音源分離")
    
    def separate_with_api(self, audio_path: str, output_dir: Optional[str] = None) -> Dict:
        """使用demucs API進行GPU加速音源分離"""
        if self.model is None:
            return self._fallback_separate(audio_path, output_dir)
        
        self.logger.info(f"使用GPU加速API進行音源分離: {audio_path}")
        
        # 載入音頻
        try:
            waveform, sample_rate = librosa.load(audio_path, sr=44100, mono=False)
            if waveform.ndim == 1:
                waveform = waveform[np.newaxis, :]
            elif waveform.ndim == 2 and waveform.shape[0] > 2:
                waveform = waveform[:2, :]  # 只取前兩個聲道
            
            # 轉換為torch tensor
            waveform_tensor = torch.from_numpy(waveform).float()
            if self.device == 'cuda':
                waveform_tensor = waveform_tensor.cuda()
            
            self.logger.info(f"音頻載入完成: {waveform.shape}, 採樣率: {sample_rate}")
            
        except Exception as e:
            self.logger.error(f"音頻載入失敗: {e}")
            return self._fallback_separate(audio_path, output_dir)
        
        # 設定輸出目錄
        if output_dir is None:
            output_dir = Path("output/music_analysis/separated")
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        audio_name = Path(audio_path).stem
        separated_dir = output_dir / audio_name
        separated_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 執行分離
            self.logger.info("開始GPU音源分離...")
            
            with torch.no_grad():
                if self.split and waveform.shape[1] > sample_rate * self.segment_length:
                    # 分段處理長音頻
                    separated_parts = []
                    segment_samples = int(sample_rate * self.segment_length)
                    
                    for i in range(0, waveform.shape[1], segment_samples):
                        end_idx = min(i + segment_samples, waveform.shape[1])
                        segment = waveform_tensor[:, i:end_idx].unsqueeze(0)
                        
                        # 分離音源
                        separated_segment = self.model(segment)
                        separated_parts.append(separated_segment.cpu())
                        
                        # 釋放GPU記憶體
                        if self.device == 'cuda':
                            torch.cuda.empty_cache()
                    
                    # 合併分離結果
                    separated_audio = torch.cat(separated_parts, dim=2)
                else:
                    # 處理整個音頻
                    waveform_batch = waveform_tensor.unsqueeze(0)
                    separated_audio = self.model(waveform_batch).cpu()
            
            # 儲存分離後的音軌
            separated_files = {
                'original': audio_path,
                'separated': True,
                'output_dir': str(separated_dir)
            }
            
            stem_names = ['drums', 'bass', 'other', 'vocals']  # htdemucs的預設順序
            
            for i, stem_name in enumerate(stem_names):
                if stem_name in self.stems:
                    output_path = separated_dir / f"{stem_name}.{self.output_format}"
                    
                    # 轉換為numpy並儲存
                    stem_audio = separated_audio[0, i].numpy()
                    
                    if self.output_format == 'wav':
                        librosa.output.write_wav(str(output_path), stem_audio, sample_rate)
                    elif self.output_format == 'mp3':
                        # 需要安裝ffmpeg
                        temp_wav = separated_dir / f"{stem_name}_temp.wav"
                        librosa.output.write_wav(str(temp_wav), stem_audio, sample_rate)
                        
                        # 轉換為MP3
                        cmd = [
                            'ffmpeg', '-i', str(temp_wav), '-b:a', f'{self.bitrate}k',
                            '-y', str(output_path)
                        ]
                        subprocess.run(cmd, capture_output=True)
                        temp_wav.unlink()  # 刪除臨時文件
                    
                    separated_files[stem_name] = str(output_path)
                    self.logger.info(f"已儲存 {stem_name}: {output_path}")
            
            self.logger.info("GPU音源分離完成")
            return separated_files
            
        except Exception as e:
            self.logger.error(f"GPU音源分離失敗: {e}")
            return self._fallback_separate(audio_path, output_dir)
    
    def _fallback_separate(self, audio_path: str, output_dir: Optional[str] = None) -> Dict:
        """備用的命令列音源分離方法"""
        self.logger.info("使用備用命令列方式進行音源分離")
        
        if output_dir is None:
            output_dir = Path("output/music_analysis/separated")
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 構建demucs命令
            cmd = [
                'python', '-m', 'demucs.separate',
                '--name', self.model_name,
                '--device', self.device if self.device == 'cuda' else 'cpu',
                '--shifts', str(self.shifts),
                '--overlap', str(self.overlap),
                '--out', str(output_dir),
            ]
            
            if self.split:
                cmd.extend(['--segment', str(self.segment_length)])
            
            cmd.append(audio_path)
            
            self.logger.info(f"執行命令: {' '.join(cmd)}")
            
            # 執行分離
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            
            if result.returncode != 0:
                self.logger.error(f"命令列分離失敗: {result.stderr}")
                return self._create_fallback_result(audio_path)
            
            # 尋找分離後的文件
            audio_name = Path(audio_path).stem
            separated_dir = output_dir / self.model_name / audio_name
            
            separated_files = {
                'original': audio_path,
                'separated': True,
                'output_dir': str(separated_dir)
            }
            
            for stem_name in self.stems:
                file_path = separated_dir / f'{stem_name}.wav'
                if file_path.exists():
                    separated_files[stem_name] = str(file_path)
                else:
                    separated_files[stem_name] = None
                    self.logger.warning(f"分離文件未找到: {stem_name}")
            
            return separated_files
            
        except subprocess.TimeoutExpired:
            self.logger.error("音源分離超時")
            return self._create_fallback_result(audio_path)
        except Exception as e:
            self.logger.error(f"命令列分離過程中發生錯誤: {e}")
            return self._create_fallback_result(audio_path)
    
    def _create_fallback_result(self, audio_path: str) -> Dict:
        """創建失敗時的備用結果"""
        return {
            'vocals': audio_path,
            'drums': audio_path,
            'bass': audio_path,
            'other': audio_path,
            'original': audio_path,
            'separated': False,
            'output_dir': None
        }
    
    def analyze_separated_audio(self, separated_files: Dict) -> Dict:
        """分析分離後的音頻特徵"""
        analysis = {
            'vocal_energy': 0.0,
            'drum_energy': 0.0,
            'bass_energy': 0.0,
            'instrumental_balance': 0.0,
            'separation_quality': 'unknown'
        }
        
        if not separated_files.get('separated', False):
            return analysis
        
        try:
            # 分析各音軌的能量
            for stem in ['vocals', 'drums', 'bass', 'other']:
                file_path = separated_files.get(stem)
                if file_path and Path(file_path).exists():
                    # 載入音頻並計算RMS能量
                    y, sr = librosa.load(file_path, duration=30)  # 只分析前30秒
                    rms_energy = np.sqrt(np.mean(y**2))
                    
                    if stem == 'vocals':
                        analysis['vocal_energy'] = float(rms_energy)
                    elif stem == 'drums':
                        analysis['drum_energy'] = float(rms_energy)
                    elif stem == 'bass':
                        analysis['bass_energy'] = float(rms_energy)
            
            # 計算樂器平衡度
            total_energy = (analysis['vocal_energy'] + analysis['drum_energy'] + 
                           analysis['bass_energy'])
            
            if total_energy > 0:
                # 計算各部分的比例標準差，越小表示越平衡
                energies = [analysis['vocal_energy'], analysis['drum_energy'], 
                           analysis['bass_energy']]
                balance_std = np.std([e/total_energy for e in energies])
                analysis['instrumental_balance'] = float(1.0 - balance_std)
            
            # 評估分離品質
            if analysis['vocal_energy'] > 0.01 and analysis['drum_energy'] > 0.01:
                if analysis['instrumental_balance'] > 0.7:
                    analysis['separation_quality'] = 'high'
                elif analysis['instrumental_balance'] > 0.5:
                    analysis['separation_quality'] = 'medium'
                else:
                    analysis['separation_quality'] = 'low'
            
        except Exception as e:
            self.logger.warning(f"音頻分析失敗: {e}")
        
        return analysis
    
    def separate(self, audio_path: str, output_dir: Optional[str] = None) -> Dict:
        """
        主要的音源分離介面
        
        Args:
            audio_path (str): 音訊檔案路徑
            output_dir (str, optional): 輸出目錄
            
        Returns:
            Dict: 分離結果，包含各音軌的路徑和分析資料
        """
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"音訊檔案不存在: {audio_path}")
        
        # 優先使用API方法（GPU加速）
        if DEMUCS_AVAILABLE and self.model is not None:
            result = self.separate_with_api(audio_path, output_dir)
        else:
            result = self._fallback_separate(audio_path, output_dir)
        
        # 分析分離結果
        if result.get('separated', False):
            analysis = self.analyze_separated_audio(result)
            result['analysis'] = analysis
        
        return result
    
    def batch_separate(self, audio_paths: List[str], output_dir: Optional[str] = None) -> Dict[str, Dict]:
        """批次音源分離"""
        results = {}
        
        for audio_path in audio_paths:
            try:
                self.logger.info(f"批次處理: {Path(audio_path).name}")
                result = self.separate(audio_path, output_dir)
                results[audio_path] = result
            except Exception as e:
                self.logger.error(f"處理 {audio_path} 時發生錯誤: {e}")
                results[audio_path] = self._create_fallback_result(audio_path)
        
        return results