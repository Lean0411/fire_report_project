#!/bin/bash

# FireGuard AI 測試運行腳本

echo "🔥 FireGuard AI 測試套件"
echo "========================="

# 檢查pytest-cov是否安裝
if ! python3 -c "import pytest_cov" 2>/dev/null; then
    echo "⚠️  pytest-cov 未安裝，正在安裝..."
    pip install pytest-cov --break-system-packages --user
fi

echo ""
echo "📊 運行測試並生成覆蓋率報告..."
echo "==============================================="

# 運行基礎測試（無外部依賴）
python3 -m pytest \
    tests/unit/models/test_simple.py \
    tests/unit/utils/test_security_utils.py \
    --cov=utils \
    --cov=config \
    --cov-report=term-missing \
    --cov-report=html \
    --cov-config=.coveragerc \
    -v

TEST_EXIT_CODE=$?

echo ""
echo "📈 覆蓋率報告已生成："
echo "  - 終端報告: 上方顯示"
echo "  - HTML報告: htmlcov/index.html"

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ 所有測試通過！"
    
    # 顯示覆蓋率摘要
    echo ""
    echo "📊 覆蓋率摘要："
    python3 -c "
import subprocess
import re

try:
    result = subprocess.run(['coverage', 'report', '--show-missing'], 
                          capture_output=True, text=True)
    lines = result.stdout.split('\n')
    for line in lines:
        if 'utils/security_utils.py' in line or 'utils/constants.py' in line or 'TOTAL' in line:
            print(f'  {line}')
except:
    print('  無法顯示詳細覆蓋率')
"
    
    echo ""
    echo "🌐 要查看詳細的HTML覆蓋率報告，請開啟:"
    echo "  file://$(pwd)/htmlcov/index.html"
    
else
    echo ""
    echo "❌ 測試失敗！"
    exit $TEST_EXIT_CODE
fi