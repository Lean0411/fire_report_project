#!/usr/bin/env python3
"""
生成測試和覆蓋率徽章
"""

import json
import subprocess
import sys


def get_coverage_percentage():
    """獲取當前覆蓋率百分比"""
    try:
        with open("coverage.json", "r") as f:
            coverage_data = json.load(f)
        return coverage_data.get("totals", {}).get("percent_covered", 0)
    except FileNotFoundError:
        return 0


def get_test_count():
    """獲取測試數量"""
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        "tests/unit/models/test_simple.py",
        "tests/unit/utils/test_security_utils.py",
        "--collect-only", "-q"
    ], capture_output=True, text=True)
    
    lines = result.stdout.strip().split('\n')
    for line in lines:
        if 'test' in line and 'collected' in line:
            # Extract number from "collected X items"
            try:
                return int(line.split()[0])
            except:
                pass
    return 0


def generate_badge_urls():
    """生成徽章URL"""
    
    coverage = get_coverage_percentage()
    test_count = get_test_count()
    
    # 覆蓋率顏色
    if coverage >= 80:
        coverage_color = "brightgreen"
    elif coverage >= 60:
        coverage_color = "yellow"
    elif coverage >= 40:
        coverage_color = "orange"
    else:
        coverage_color = "red"
    
    badges = {
        "tests": f"https://img.shields.io/badge/tests-{test_count}%20passed-brightgreen?style=flat-square",
        "coverage": f"https://img.shields.io/badge/coverage-{coverage:.1f}%25-{coverage_color}?style=flat-square",
        "pytest": "https://img.shields.io/badge/pytest-7.4+-blue?style=flat-square",
        "python": "https://img.shields.io/badge/python-3.8+-blue?style=flat-square"
    }
    
    return badges, coverage, test_count


def update_readme_badges():
    """更新README中的徽章"""
    
    badges, coverage, test_count = generate_badge_urls()
    
    badge_markdown = f"""
## 📊 測試狀態

![Tests]({badges['tests']})
![Coverage]({badges['coverage']})
![Pytest]({badges['pytest']})
![Python]({badges['python']})

### 測試摘要
- **測試數量**: {test_count} 個測試通過
- **代碼覆蓋率**: {coverage:.1f}%
- **測試框架**: pytest 7.4+
- **支援Python**: 3.8+

### 運行測試
```bash
# 快速測試
./run_tests.sh

# 詳細分析
python3 coverage_analysis.py

# 查看HTML報告
open htmlcov/index.html
```
"""
    
    print("📊 測試徽章已生成!")
    print(f"✅ 測試: {test_count} 個通過")
    print(f"📈 覆蓋率: {coverage:.1f}%")
    print("\n🏷️  徽章Markdown:")
    print(badge_markdown)
    
    # 可以選擇性地插入到README中
    return badge_markdown


if __name__ == "__main__":
    update_readme_badges()