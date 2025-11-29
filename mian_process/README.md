# 自動剪輯影片生成系統

## 專案概述

這是一個基於影像分析、音樂處理和智能剪輯的自動化影片生成系統。輸入原始影片、音樂和歌詞，系統會自動分析影片場景、提取音樂節拍，並生成符合音樂節奏的剪輯影片。

## 主要功能

### 🎥 影像分析
- **場景檢測**: 基於直方圖差異自動檢測場景切換點
- **物體識別**: 使用YOLO模型進行物體檢測和場景分析
- **特徵提取**: 提取亮度、對比度、邊緣密度等影像特徵

### 🎵 音樂處理
- **音源分離**: 使用demucs分離人聲、鼓聲、貝斯和其他樂器
- **節拍分析**: 提取音樂的節拍、tempo和節奏特徵
- **頻譜分析**: 分析MFCC、Chroma和頻譜重心等音樂特徵

### 🎬 智能剪輯
- **場景評分**: 根據視覺和音頻特徵為場景打分
- **節拍同步**: 將影片剪輯點與音樂節拍同步
- **歌詞疊加**: 自動添加歌詞字幕

## 系統架構

```
mian_process/
├── image_analysis/          # 影像分析模組
│   ├── scene_detector.py   # 場景檢測
│   └── feature_extractor.py # 特徵提取
├── music_processing/        # 音樂處理模組
│   ├── audio_separator.py  # 音源分離
│   └── rhythm_analyzer.py  # 節拍分析
├── video_generation/        # 影片生成模組
│   └── video_composer.py   # 影片合成
├── utils/                   # 工具模組
│   ├── logger.py           # 日誌工具
│   └── common.py           # 通用函數
├── tests/                   # 測試模組
├── docs/                    # 文檔
├── config.yaml             # 配置檔案
├── main.py                 # 主程式入口
└── requirements.txt        # 依賴套件
```

## 安裝指南

### 1. 環境需求
- Python 3.7+
- FFmpeg (用於影片處理)
- CUDA (可選，用於GPU加速)

### 2. 安裝依賴
```bash
# 克隆或下載專案
cd mian_process

# 安裝Python依賴
pip install -r requirements.txt

# 安裝demucs (音源分離)
pip install demucs

# 安裝YOLO (物體檢測)
pip install ultralytics
```

### 3. FFmpeg安裝
- Windows: 下載並安裝 FFmpeg，添加到PATH
- macOS: `brew install ffmpeg`
- Linux: `sudo apt-get install ffmpeg`

## 使用方法

### 基本用法
```bash
python main.py --video input.mp4 --audio music.mp3 --output result.mp4
```

### 完整參數
```bash
python main.py \
  --video input.mp4 \        # 輸入影片
  --audio music.mp3 \        # 輸入音樂
  --lyrics lyrics.txt \      # 歌詞檔案 (可選)
  --output result.mp4 \      # 輸出影片
  --config config.yaml       # 配置檔案 (可選)
```

### 歌詞格式
支持兩種歌詞格式：

1. **純文字格式** (.txt)
```
第一句歌詞
第二句歌詞
第三句歌詞
```

2. **時間軸格式** (.json)
```json
[
  {"start": 0.0, "end": 3.0, "text": "第一句歌詞"},
  {"start": 3.0, "end": 6.0, "text": "第二句歌詞"},
  {"start": 6.0, "end": 9.0, "text": "第三句歌詞"}
]
```

## 配置選項

編輯 `config.yaml` 文件來調整系統參數：

```yaml
# 影像分析設定
image_analysis:
  histogram:
    threshold: 0.3          # 場景切換閾值
    min_scene_length: 2.0   # 最小場景長度(秒)
  yolo:
    confidence: 0.5         # 物體檢測信心度
    device: "cpu"          # 使用設備 (cpu/cuda)

# 音樂處理設定
music_processing:
  demucs:
    model: "htdemucs"      # demucs模型
    device: "cpu"          # 使用設備
  tempo:
    hop_length: 512        # 音頻分析參數

# 影片生成設定
video_generation:
  fps: 30                  # 輸出幀率
  resolution: [1920, 1080] # 輸出解析度
  format: "mp4"           # 輸出格式
```

## 測試

執行測試套件：
```bash
cd tests
python test_main.py
```

## 開發指南

### 添加新功能
1. 在相應模組下創建新的Python檔案
2. 實現相關類別和方法
3. 在模組的 `__init__.py` 中導出新功能
4. 添加相應的測試用例

### 調試
啟用詳細日誌：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 故障排除

### 常見問題

1. **FFmpeg錯誤**
   - 確保FFmpeg已正確安裝並在PATH中
   - 檢查輸入檔案格式是否支援

2. **記憶體不足**
   - 降低影片解析度
   - 使用較小的YOLO模型
   - 分段處理長影片

3. **CUDA錯誤**
   - 設定 `device: "cpu"` 使用CPU處理
   - 確保CUDA版本與PyTorch相容

4. **音源分離失敗**
   - 檢查demucs是否正確安裝
   - 確保音頻檔案格式正確

## 效能優化

### GPU加速
設定配置使用GPU：
```yaml
image_analysis:
  yolo:
    device: "cuda"
music_processing:
  demucs:
    device: "cuda"
```

### 批次處理
對於多個檔案，可以編寫批次處理腳本：
```python
import glob
from main import main

for video_file in glob.glob("*.mp4"):
    # 處理每個影片檔案
    main(video=video_file, audio="music.mp3", output=f"output_{video_file}")
```

## 版本更新

### v1.0.0 (當前版本)
- 基本的場景檢測和特徵提取
- 音源分離和節拍分析
- 智能影片剪輯和合成
- 歌詞疊加功能

## 授權

本專案採用 MIT 授權條款。

## 貢獻

歡迎提交Issue和Pull Request來改進這個專案！

## 聯絡

如有問題請聯絡開發團隊或提交Issue。