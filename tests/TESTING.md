# FireGuard AI 測試覆蓋率報告

生成時間: 2025-06-18 20:34:48

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
