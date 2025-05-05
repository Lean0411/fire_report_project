以下是你專案的文字版 `README.md`，包含完整說明、更新內容與技術細節，適合直接貼上 GitHub 使用：

---

````markdown
# 🔥 火災偵測與報告系統

本專案是一個結合深度學習與大型語言模型（LLM）的火災偵測系統，能夠自動分析上傳的圖片是否含有火災情況，並依照使用者身份生成對應的應變建議報告。支援 Web 界面操作，搭配 GPU 加速與本地推論部署，提升實用性與反應速度。

---

## ✅ 系統功能特色

- 🔍 圖像火災自動偵測（CNN 模型推論）
- 🧠 火災發生時整合 LLM（Gemma 模型 via Ollama）自動補全建議
- 🧑‍🚒 支援三種使用者身份（一般民眾、消防隊員、管理單位）
- 🧾 根據 SOP 規則提供具體應變建議（知識庫 JSON 結構）
- 💻 響應式前端設計（Bootstrap 5，支援拖曳上傳與複製報告）
- 🔌 模組化設計，可擴充部署於工廠、實驗室、無人機等場域

---

## 🖥️ 環境需求

- Python 3.12+
- 建議安裝 CUDA 支援（GPU 推論）
- 作業系統：Linux / macOS / WSL / Windows（部分功能需調整）

---

## 🚀 安裝與啟動流程

### 1. 專案下載與虛擬環境建立

```bash
git clone https://github.com/Lean0411/fire_report_project.git
cd fire_report_project

python -m venv .venv
source .venv/bin/activate  # Windows 請使用 .venv\Scripts\activate
````

### 2. 安裝依賴

```bash
pip install -r requirements.txt
```

### 3. 放入模型檔案

請將已訓練的模型 `deep_wildfire_cnn_model_amp.pth` 放入：

```
models/fire_detection/deep_wildfire_cnn_model_amp.pth
```



---

## 🧪 執行應用

```bash
python run.py
```

啟動成功後，打開瀏覽器：

```
http://127.0.0.1:5000
```

即可進行圖片上傳與火災偵測、報告建議生成。

---

## 🧠 進階功能：串接本地 LLM（Gemma）

系統支援透過 [Ollama](https://ollama.com) 快速部署 LLM 模型，範例如下：

```bash
# 安裝 Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 拉取模型
ollama pull gemma:7b
```

你也可以將火災圖片分析結果送入 LLM 進行報告補全與行動建議自動產出，請參考 `/fire_report_project/llm_srv/` 下的 `serve.py` 實作。

---

## 📁 專案結構總覽

```
fire_report_project/
├── app.py                    # Flask 主應用程式
├── run.py                   # 可選執行入口（可改為 run server）
├── app/                     # Flask 輔助模組（config, routes, utils 等）
├── templates/index.html     # 前端頁面（Jinja2）
├── static/                  # 上傳圖像與樣式
├── models/                  # 模型存放位置（需手動放置）
├── knowledge_base/sop.json  # SOP 建議知識庫（依角色分類）
├── tests/                   # 測試模組（pytest）
├── requirements.txt         # Python 套件需求
├── README.md                # 本說明文件
└── .env                     # 可選環境變數設定檔
```

---

## ⚠️ 注意事項

* 若首次使用或圖片來源多樣，模型可能會有偵測誤差，請配合人工複檢。
* LLM 產出建議內容不作為唯一判斷依據，實際情況仍需依照 SOP 與專業人員判斷。
* 使用過程若報錯 `Object of type Undefined is not JSON serializable`，請確認 `category_labels` 有正確傳入模板。

---

## 📜 授權 License

本專案採用 MIT License，歡迎自由使用、修改與再散布。

---

## 🙋‍♂️ 聯絡方式

作者：Lean
Email：[113753207@g.nccu.edu.tw](mailto:113753207@g.nccu.edu.tw)

---

```

如果你需要我幫你加上專案圖、系統架構圖、或轉成 PDF 格式也可以告訴我！你想加上專案執行畫面或部署指令嗎？
```
