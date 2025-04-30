# 火災偵測與報告系統

這是一個基於深度學習的火災偵測系統，可以自動分析圖片中是否存在火災情況，並根據使用者的角色提供相應的建議方案。

## 功能特點

- 火災影像自動偵測
- 多角色建議方案（一般民眾、消防隊員、管理單位）
- 即時分析報告生成
- 支援拖放上傳圖片
- 響應式網頁設計

## 系統需求

- Python 3.12+
- CUDA 支援（用於 GPU 加速，可選）
- 足夠的磁碟空間（用於模型文件）

## 安裝步驟

1. 克隆專案：
```bash
git clone https://github.com/Lean0411/fire_report_project.git
cd fire_report_project
```

2. 建立虛擬環境並啟動：
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或者在 Windows 上：
# .venv\Scripts\activate
```

3. 安裝依賴套件：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 啟動應用程式：
```bash
python app.py
```

2. 在瀏覽器中打開：`http://127.0.0.1:5000`

3. 使用系統：
   - 上傳圖片（支援 JPG、PNG、JPEG 格式）
   - 選擇您的身份（一般民眾、消防隊員、管理單位）
   - 點擊「上傳並分析」按鈕
   - 查看分析結果和建議方案
   - 可以使用「複製報告」功能複製完整報告

## 目錄結構

- `app.py`: 主要應用程式檔案
- `templates/`: HTML 模板文件
- `static/`: 靜態資源（圖片、上傳文件等）
- `models/`: 模型文件
- `knowledge_base/`: 知識庫文件（SOP 建議等）

## 注意事項

- 模型文件 `deep_wildfire_cnn_model_amp.pth` 請從 [此連結] 下載並放置於 `models/fire_detection/` 目錄下
- 第一次運行可能需要一些時間來載入模型
- 建議使用現代瀏覽器以獲得最佳體驗

## 授權

MIT License