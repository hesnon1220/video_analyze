#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
音源分離模組 - 使用demucs分離人聲、鼓聲、貝斯和其他樂器
"""

import os
import subprocess
from pathlib import Path
from typing import Dict, Optional
import logging

class AudioSeparator:
    """音源分離器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.demucs_config = config.get('demucs', {})
        self.model = self.demucs_config.get('model', 'htdemucs')
        self.device = self.demucs_config.get('device', 'cpu')
        self.logger = logging.getLogger(__name__)
        
        # 檢查demucs是否可用
        self.demucs_available = self._check_demucs()
    
    def _check_demucs(self) -> bool:
        """檢查demucs是否已安裝"""
        try:
            result = subprocess.run(['python', '-m', 'demucs', '--help'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                self.logger.info("demucs 可用")
                return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        self.logger.warning("demucs 未安裝或不可用，將跳過音源分離")
        return False
    
    def separate(self, audio_path: str, output_dir: Optional[str] = None) -> Dict:
        """
        分離音源
        
        Args:
            audio_path (str): 音訊檔案路徑
            output_dir (str, optional): 輸出目錄
            
        Returns:
            Dict: 分離結果，包含各音軌的路徑
        """
        self.logger.info(f"開始音源分離: {audio_path}")
        
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"音訊檔案不存在: {audio_path}")
        
        if not self.demucs_available:
            self.logger.warning("demucs不可用，返回原始音頻路徑")
            return {
                'vocals': audio_path,
                'drums': audio_path,
                'bass': audio_path,
                'other': audio_path,
                'original': audio_path,
                'separated': False
            }
        
        # 設定輸出目錄
        if output_dir is None:
            output_dir = Path("output/music_analysis/separated")
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 構建demucs命令
            cmd = [
                'python', '-m', 'demucs',
                '--model', self.model,
                '--device', self.device,
                '--out', str(output_dir),
                audio_path
            ]
            
            self.logger.info(f"執行demucs命令: {' '.join(cmd)}")
            
            # 執行分離
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                self.logger.error(f"demucs執行失敗: {result.stderr}")
                raise RuntimeError(f"音源分離失敗: {result.stderr}")
            
            # 尋找分離後的文件
            audio_name = Path(audio_path).stem
            separated_dir = output_dir / self.model / audio_name
            
            separated_files = {
                'vocals': separated_dir / 'vocals.wav',
                'drums': separated_dir / 'drums.wav',
                'bass': separated_dir / 'bass.wav',
                'other': separated_dir / 'other.wav',
                'original': audio_path,
                'separated': True,
                'output_dir': str(separated_dir)
            }
            
            # 檢查分離文件是否存在
            missing_files = []
            for track_name, file_path in separated_files.items():
                if track_name not in ['original', 'separated', 'output_dir']:
                    if not Path(file_path).exists():
                        missing_files.append(track_name)
                        separated_files[track_name] = None
            
            if missing_files:
                self.logger.warning(f"部分分離文件未找到: {missing_files}")
            else:
                self.logger.info("音源分離完成")
            
            return separated_files
            
        except subprocess.TimeoutExpired:
            self.logger.error("demucs執行超時")
            raise RuntimeError("音源分離超時")
        except Exception as e:
            self.logger.error(f"音源分離過程中發生錯誤: {e}")
            raise