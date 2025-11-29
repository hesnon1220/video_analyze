#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
環境修復腳本 - 解決OpenMP衝突和依賴問題
"""

import os
import sys
import subprocess
import logging

def setup_environment_fixes():
    """設置環境修復"""
    print("🔧 開始環境修復...")
    
    # 1. 設置OpenMP環境變數
    print("📋 設置OpenMP環境變數...")
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    os.environ['OMP_NUM_THREADS'] = '4'  # 限制OpenMP線程數
    print("✅ OpenMP衝突修復完成")
    
    # 2. 檢查並修復Python路徑
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    print(f"✅ Python路徑已添加: {current_dir}")
    
    # 3. 檢查關鍵套件
    required_packages = [
        'torch', 'torchvision', 'opencv-python', 'ultralytics', 
        'librosa', 'moviepy', 'numpy', 'pyyaml'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} 已安裝")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} 未安裝")
    
    return missing_packages

def install_missing_packages(missing_packages):
    """安裝缺失的套件"""
    if not missing_packages:
        print("🎉 所有必需套件都已安裝")
        return True
    
    print(f"📦 需要安裝的套件: {missing_packages}")
    
    for package in missing_packages:
        try:
            print(f"正在安裝 {package}...")
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', package
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"✅ {package} 安裝成功")
            else:
                print(f"❌ {package} 安裝失敗: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⏱️ {package} 安裝超時")
            return False
        except Exception as e:
            print(f"❌ 安裝 {package} 時發生錯誤: {e}")
            return False
    
    return True

def test_imports():
    """測試關鍵模組導入"""
    print("\n🧪 測試模組導入...")
    
    test_modules = [
        ('torch', 'PyTorch'),
        ('cv2', 'OpenCV'),
        ('ultralytics', 'YOLO'),
        ('librosa', 'Librosa'),
        ('moviepy.editor', 'MoviePy'),
        ('yaml', 'PyYAML'),
        ('numpy', 'NumPy')
    ]
    
    success_count = 0
    for module_name, display_name in test_modules:
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'Unknown')
            print(f"✅ {display_name}: {version}")
            success_count += 1
        except ImportError as e:
            print(f"❌ {display_name}: 導入失敗 - {e}")
    
    print(f"\n📊 導入測試結果: {success_count}/{len(test_modules)} 成功")
    return success_count == len(test_modules)

def check_gpu_status():
    """檢查GPU狀態"""
    print("\n🖥️ 檢查GPU狀態...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✅ GPU可用: {gpu_name}")
            print(f"   GPU數量: {gpu_count}")
            print(f"   GPU記憶體: {gpu_memory:.1f}GB")
            return True
        else:
            print("⚠️ GPU不可用，將使用CPU模式")
            return False
    except Exception as e:
        print(f"❌ GPU檢查失敗: {e}")
        return False

if __name__ == "__main__":
    print("🚀 環境診斷和修復工具")
    print("=" * 50)
    
    # 設置環境修復
    missing_packages = setup_environment_fixes()
    
    # 安裝缺失套件
    if missing_packages:
        print(f"\n📦 發現 {len(missing_packages)} 個缺失套件，開始安裝...")
        install_success = install_missing_packages(missing_packages)
        if not install_success:
            print("❌ 套件安裝失敗，請檢查網路連線或手動安裝")
            sys.exit(1)
    
    # 測試導入
    import_success = test_imports()
    if not import_success:
        print("❌ 模組導入測試失敗")
        sys.exit(1)
    
    # 檢查GPU
    gpu_available = check_gpu_status()
    
    print("\n" + "=" * 50)
    print("🎉 環境修復完成！")
    print(f"💻 運行模式: {'GPU加速' if gpu_available else 'CPU'}")
    print("現在可以運行端到端測試了")
    print("=" * 50)