#!/usr/bin/env python3
"""
覆蓋率分析工具
生成詳細的測試覆蓋率報告和分析
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_coverage_analysis():
    """運行覆蓋率分析並生成報告"""
    
    print("🔍 FireGuard AI 覆蓋率分析")
    print("=" * 40)
    
    # 1. 運行測試並生成覆蓋率數據
    print("📊 正在運行測試...")
    result = subprocess.run([
        sys.executable, "-m", "pytest",
        "tests/unit/models/test_simple.py",
        "tests/unit/utils/test_security_utils.py",
        "--cov=utils",
        "--cov=config", 
        "--cov=models",
        "--cov=services",
        "--cov=api",
        "--cov-report=json",
        "--cov-report=term-missing",
        "--cov-report=html",
        "--quiet"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("❌ 測試失敗!")
        print(result.stderr)
        return
    
    # 2. 讀取JSON覆蓋率數據
    try:
        with open("coverage.json", "r") as f:
            coverage_data = json.load(f)
    except FileNotFoundError:
        print("❌ 找不到覆蓋率數據文件")
        return
    
    # 3. 分析覆蓋率數據
    analyze_coverage(coverage_data)
    
    # 4. 生成建議
    generate_recommendations(coverage_data)


def analyze_coverage(coverage_data):
    """分析覆蓋率數據"""
    
    print("\n📈 覆蓋率分析結果")
    print("-" * 40)
    
    files = coverage_data.get("files", {})
    total_stats = coverage_data.get("totals", {})
    
    # 總體覆蓋率
    total_coverage = total_stats.get("percent_covered", 0)
    print(f"📊 總體覆蓋率: {total_coverage:.1f}%")
    print(f"📋 總行數: {total_stats.get('num_statements', 0)}")
    print(f"✅ 已測試行數: {total_stats.get('covered_lines', 0)}")
    print(f"❌ 未測試行數: {total_stats.get('missing_lines', 0)}")
    
    # 按模組分析
    print(f"\n🗂️  模組覆蓋率詳情:")
    print(f"{'模組':<30} {'覆蓋率':<10} {'行數':<8} {'測試':<8} {'未測試':<8}")
    print("-" * 70)
    
    module_stats = []
    
    for file_path, file_data in files.items():
        if file_path.startswith(('tests/', '__pycache__')):
            continue
            
        coverage_percent = file_data["summary"]["percent_covered"]
        num_statements = file_data["summary"]["num_statements"]
        covered_lines = file_data["summary"]["covered_lines"]
        missing_lines = file_data["summary"]["missing_lines"]
        
        module_stats.append({
            'path': file_path,
            'coverage': coverage_percent,
            'statements': num_statements,
            'covered': covered_lines,
            'missing': missing_lines
        })
        
        # 格式化顯示
        short_path = file_path.replace("./", "")
        print(f"{short_path:<30} {coverage_percent:>6.1f}%    {num_statements:>6} {covered_lines:>8} {missing_lines:>8}")
    
    # 覆蓋率等級分類
    print(f"\n🎯 覆蓋率等級分布:")
    excellent = sum(1 for m in module_stats if m['coverage'] >= 90)
    good = sum(1 for m in module_stats if 70 <= m['coverage'] < 90)
    fair = sum(1 for m in module_stats if 50 <= m['coverage'] < 70)
    poor = sum(1 for m in module_stats if m['coverage'] < 50)
    
    print(f"  🟢 優秀 (≥90%): {excellent} 個模組")
    print(f"  🟡 良好 (70-89%): {good} 個模組") 
    print(f"  🟠 普通 (50-69%): {fair} 個模組")
    print(f"  🔴 需改進 (<50%): {poor} 個模組")


def generate_recommendations(coverage_data):
    """生成改進建議"""
    
    print(f"\n💡 改進建議")
    print("-" * 40)
    
    files = coverage_data.get("files", {})
    
    # 找出需要改進的文件
    low_coverage_files = []
    untested_files = []
    
    for file_path, file_data in files.items():
        if file_path.startswith(('tests/', '__pycache__')):
            continue
            
        coverage = file_data["summary"]["percent_covered"]
        
        if coverage == 0:
            untested_files.append(file_path)
        elif coverage < 70:
            low_coverage_files.append((file_path, coverage))
    
    # 優先級建議
    print("🎯 優先改進項目:")
    
    if untested_files:
        print(f"\n1. 📝 創建測試文件 (優先級: 高)")
        for file_path in untested_files[:5]:  # 只顯示前5個
            print(f"   - {file_path}")
        if len(untested_files) > 5:
            print(f"   ... 還有 {len(untested_files) - 5} 個文件")
    
    if low_coverage_files:
        print(f"\n2. 🔧 增強現有測試 (優先級: 中)")
        sorted_files = sorted(low_coverage_files, key=lambda x: x[1])
        for file_path, coverage in sorted_files[:3]:  # 只顯示前3個
            print(f"   - {file_path} ({coverage:.1f}%)")
    
    # 具體建議
    print(f"\n📋 具體建議:")
    print("   • 為 API 端點添加整合測試")
    print("   • 為模型載入和預測添加單元測試") 
    print("   • 為 AI 服務添加 Mock 測試")
    print("   • 為圖像處理服務添加測試")
    print("   • 增加錯誤處理的測試案例")
    
    # 目標設定
    total_coverage = coverage_data.get("totals", {}).get("percent_covered", 0)
    
    print(f"\n🎯 建議的覆蓋率目標:")
    if total_coverage < 50:
        print("   • 短期目標: 50% (基礎功能覆蓋)")
        print("   • 中期目標: 70% (主要功能覆蓋)")
        print("   • 長期目標: 85% (高品質覆蓋)")
    elif total_coverage < 70:
        print("   • 短期目標: 70% (主要功能覆蓋)")
        print("   • 長期目標: 85% (高品質覆蓋)")
    else:
        print("   • 目標: 85%+ (維持高品質)")


def generate_coverage_report():
    """生成覆蓋率報告文件"""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report_content = f"""# FireGuard AI 測試覆蓋率報告

生成時間: {timestamp}

## 快速開始

```bash
# 運行基礎測試
./run_tests.sh

# 或手動運行
python3 -m pytest tests/unit/models/test_simple.py tests/unit/utils/test_security_utils.py --cov=utils --cov=config --cov-report=html

# 查看HTML報告
open htmlcov/index.html
```

## 覆蓋率分析

```bash
# 運行覆蓋率分析工具
python3 coverage_analysis.py
```

## 測試目標

- [ ] 基礎工具函數測試: ✅ 95%+
- [ ] API端點測試: 🚧 需要添加
- [ ] 模型測試: 🚧 需要添加  
- [ ] 服務層測試: 🚧 需要添加
- [ ] 整合測試: 🚧 需要添加

## 文件說明

- `run_tests.sh`: 主要測試腳本
- `coverage_analysis.py`: 覆蓋率分析工具
- `htmlcov/`: HTML覆蓋率報告目錄
- `.coveragerc`: 覆蓋率配置文件
- `pytest.ini`: pytest配置文件

## 注意事項

某些測試需要外部依賴（torch, flask, openai），如果未安裝這些依賴，相關測試會被跳過。
"""

    with open("TESTING.md", "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print(f"\n📄 測試文檔已生成: TESTING.md")


if __name__ == "__main__":
    run_coverage_analysis()
    generate_coverage_report()
    
    print(f"\n🎉 覆蓋率分析完成!")
    print(f"📁 查看詳細報告: htmlcov/index.html")
    print(f"📖 閱讀測試指南: TESTING.md")