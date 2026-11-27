# WallGuard：牆面危害 AI 辨識系統

## 🎯 這是什麼

DTAI 課程專題 WallGuard 的辨識系統，原名「牆牆出事」。拍一張牆面照片，系統先以 ResNet50V2 判斷是裂縫、壁癌還是鏽蝕，再以 U-Net 標出缺損範圍並估計面積，最後請 GPT 給出白話的處理建議。

```
照片 ─▶ ResNet50V2 判斷種類 ─┐
     └▶ U-Net 標出缺損範圍 ──┼─▶ 面積估計 ─▶ GPT 處理建議
```

專案曾受邀在 InnoFest@ICI 2024 展出，也參加了 2025 AI Award。

▶️ [Demo 影片（AI Award 2025）](https://www.youtube.com/watch?v=pVikml-YPsw)

| 筆記本 | 內容 | 開啟 |
|---|---|---|
| `3_wallguard_full.ipynb` | **完整版**：展出時的整合系統，分類、分割、面積與建議一次完成 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Yunyuhu/WallGuard/blob/main/3_wallguard_full.ipynb) |
| `1_resnet_classification.ipynb` | 開發紀錄・**初版**：ResNet50V2 分類實驗（2024） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Yunyuhu/WallGuard/blob/main/1_resnet_classification.ipynb) |
| `2_unet_segmentation.ipynb` | 開發紀錄・**進階版**：U-Net 分割實驗，含面積換算與 GPT 建議（2024） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Yunyuhu/WallGuard/blob/main/2_unet_segmentation.ipynb) |

## 💡 為什麼做

臺灣超過一半的房屋屋齡在 30 年以上，北部七成以上的住宅是需要共同決策的集合住宅。壁癌、裂縫、鋼筋外露處處可見，但多數人分不清它們的差別，也不知道該怎麼處理、該找誰，常常錯過處理的時機。我們想讓民眾拍一張照片，就能知道是什麼問題、大概多嚴重、下一步該做什麼。

## 🕹️ 怎麼用

建議直接執行**完整版**：

1. 點完整版的 **Open in Colab**。
2. 在「執行階段 → 變更執行階段類型」選 **T4 GPU**。
3. （可選）在左側欄的 🔑 **Secrets** 新增 `OPENAI_API_KEY`，並開啟「筆記本存取權」。沒有設定也能執行，只是不會產生 AI 建議。
4. 從第一格依序執行：
   - 程式會從 [dtaidataset](https://github.com/Yunyuhu/dtaidataset) 下載約 190 MB 的訓練資料。
   - 依序訓練 ResNet50V2 與 U-Net，T4 上約 20～30 分鐘。想先確認流程，把第一格的 `QUICK_TEST` 改成 `True`，幾分鐘就能跑完。
   - 最後開啟 Gradio 介面，上傳牆面照片就能看到種類、範圍、面積與建議。

初版與進階版也能照同樣步驟執行，但兩份都**一定要**設定 `OPENAI_API_KEY`，因為建立 OpenAI 連線的那一格會讀取金鑰。進階版的訓練約需 50 分鐘。

> 模型檔（U-Net 約 120 MB）沒有放進這個 repo，每次執行都會重新訓練。

## 🛠️ 怎麼實作

### 系統流程（完整版）

| 步驟 | 做法 |
|---|---|
| 資料 | 3,000 張室內牆面照片，裂縫、壁癌、鏽蝕（混凝土剝落與鋼筋外露）各 1,000 張，來自 Roboflow Universe 的公開資料集 |
| 種類判斷 | 預訓練的 ResNet50V2 做遷移學習，判斷整張照片屬於哪一種牆損；信心不足時顯示「無法確定」 |
| 範圍標註 | U-Net 影像分割標出缺損的位置與範圍 |
| 面積估計 | 依分割結果計算缺損占照片的比例，並以每像素 0.5 mm 粗估面積 |
| 處理建議 | 把種類、信心與面積交給 GPT，以繁體中文說明成因、嚴重程度與處理方式 |
| 介面 | Gradio：上傳照片後一次看到種類機率、範圍疊圖、面積與建議 |

從兩個模型實驗整合成完整版時，也改進了實驗階段的問題：

| 實驗階段的問題 | 完整版的做法 |
|---|---|
| 初版用訓練資料評估準確率 | 分層切出 20% 驗證集，兩個模型共用同一份切分，並列出混淆矩陣 |
| 進階版的訓練／驗證集沒有打散，驗證集只有壁癌 | 打散後分層切分，三類都在驗證集 |
| 進階版的遮罩用固定門檻，沒有用到照片的類別，標籤只分出明暗 | 依類別用形態學運算找出比周圍亮（壁癌）或暗（裂縫、鏽蝕）的區域，標成「缺損／背景」；裂縫只保留夠長的線條 |
| 遮罩存成 JPEG，壓縮雜訊變成假的類別 | 遮罩只放在記憶體，不經過 JPEG |
| 三類 softmax 卻用 binary cross-entropy | 改成單一通道 sigmoid，損失函數用 binary cross-entropy＋Dice（缺損只占小部分，避免模型全部猜背景），並計算 IoU |
| 上傳的 RGB 照片用 BGR 轉灰階 | 一律以 RGB 轉換 |
| GPT 提示詞為英文、回答很短 | 改成繁體中文，依種類、面積說明成因、嚴重程度與處理方式 |

執行時會印出 ResNet50V2 的驗證集準確率與混淆矩陣，以及 U-Net 的像素準確率與 IoU。

### 開發過程：兩個模型實驗

兩個實驗使用同一份資料：3,000 張室內牆面照片，裂縫、壁癌、鏽蝕（混凝土剝落與鋼筋外露）各 1,000 張，來自 Roboflow Universe 的公開資料集。

| | 初版：ResNet50V2 分類 | 進階版：U-Net 影像分割 |
|---|---|---|
| 任務 | 判斷整張照片是裂縫、壁癌還是鏽蝕 | 標出缺損的位置與範圍，換算面積 |
| 輸入 | 彩色照片，224×224 | 灰階照片，224×224；遮罩以灰階門檻 127 二值化產生 |
| 模型 | 預訓練的 ResNet50V2 去掉頂層、凍結權重，接一層三類 softmax | U-Net：4 次「2 層卷積＋最大池化」擷取特徵，再以上採樣與跳接還原尺寸（約 3,100 萬個參數） |
| 訓練 | categorical cross-entropy、Adam；batch size 8，8 個 epoch | Adam（1e-4），搭配 EarlyStopping、ModelCheckpoint、ReduceLROnPlateau；batch size 4，5 個 epoch |
| 結果 | 準確率 98.97%（以訓練資料評估） | 驗證集準確率 90.42%、損失 0.124 |
| 後續 | 只知道整張照片屬於哪一類，看不出位置與大小 | 把面積送給 `gpt-3.5-turbo` 產生建議，並用 Gradio 做成介面 |

兩個實驗各有所長：ResNet50V2 能判斷種類，但看不出位置與大小；U-Net 能標出位置、計算面積，卻分不出種類。展出版本因此把兩者整合成同一條流程。

![進階版：原始照片與三個輸出通道的遮罩](images/wallguard-masks.png)

## 🙋 我負責什麼

我在 4 人團隊（DesignLens）裡負責資訊開發和專案管理（PM），程式由我和一位隊友共同撰寫。

## 📍 完成到哪

- **2024：** 分別完成 ResNet50V2 與 U-Net 的模型實驗，整合成完整系統後，受邀在 InnoFest@ICI 2024 展出。
- **2025：** 參加 AI Award。
- **2026：** 公開完整版與兩份開發紀錄的程式碼，可在 Colab 直接執行。

仍有的限制：

- **遮罩是近似標註：** U-Net 的遮罩由影像處理自動產生，不是人工標註，IoU 衡量的是模型重現這些遮罩的程度。裂縫資料夾裡也混有沒有裂縫的牆面照片，遮罩會標到一些紋理。
- **面積不是真實尺寸：** 以「每個像素 0.5 mm」粗估，沒有依拍攝距離校正。
- **GPT 模型名稱：** 完整版預設 `gpt-4o-mini`，若 OpenAI 停用，請在第一格修改 `OPENAI_MODEL`。

下一步：

- 改用人工標註的遮罩，讓 U-Net 學到真正的缺損範圍。
- 以 RAG 提升建議內容的精準度，減少幻覺。

## 🔗 去哪看成果

- [Demo 影片（AI Award 2025）](https://www.youtube.com/watch?v=pVikml-YPsw)
- [作品集網站的專案頁](https://yunyu-web-digital-profile.vercel.app/projects/wallguard)：使用者研究、功能設計與完整流程
- [訓練資料集 dtaidataset](https://github.com/Yunyuhu/dtaidataset)：資料來自 Roboflow Universe 上的公開資料集（包含 DeepCrack、CRACK500 等），授權依各原始資料集為準
