#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
日誌設定工具
"""

import logging
import os
from pathlib import Path
from datetime import datetime

def setup_logger(name, log_dir="../logs", level=logging.INFO):
    """
    設定日誌記錄器
    
    Args:
        name (str): 日誌記錄器名稱
        log_dir (str): 日誌檔案目錄
        level: 日誌級別
    
    Returns:
        logging.Logger: 配置好的日誌記錄器
    """
    # 確保日誌目錄存在
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # 建立日誌記錄器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重複添加處理器
    if logger.handlers:
        return logger
    
    # 建立格式器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 建立檔案處理器
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"{name}_{timestamp}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    
    # 建立控制台處理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # 添加處理器到日誌記錄器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger