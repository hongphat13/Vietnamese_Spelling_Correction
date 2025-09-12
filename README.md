<!-- Banner -->
<p align="center">
  <a href="https://www.uit.edu.vn/" title="Trường Đại học Công nghệ Thông tin" style="border: none;">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="Trường Đại học Công nghệ Thông tin | University of Information Technology">
  </a>
</p>

---

# **VIETNAMESE SPELLING CORRECTION**

## **CS221 — NATURAL LANGUAGE PROCESSING**

---

## **📝 Giới thiệu**

Dự án xây dựng hệ thống **sửa lỗi chính tả tiếng Việt** dựa trên **Hidden Markov Model (HMM)** với **Viterbi decoding** và **mô hình ngôn ngữ trigram**.  
 Bài toán được mô hình hoá như quá trình giải mã: câu **quan sát** (có lỗi) → chuỗi **trạng thái ẩn** (câu đúng). Mô hình kết hợp:

- **Xác suất chuyển tiếp** (LM n-gram/trigram) để chọn từ hợp ngữ cảnh.

- **Xác suất phát xạ** (emission) mô tả **mẫu lỗi gõ**: dấu tiếng Việt (diacritics), **TELEX**, nhầm phím lân cận, **homophone**, tách/ghép âm tiết…

**Ứng dụng**: làm sạch dữ liệu đầu vào cho pipeline NLP (dịch máy, QA, sentiment), giảm lỗi nhập liệu trong biểu mẫu/app, hỗ trợ soát chính tả tiếng Việt.

---

## **✨ Tính năng chính**

- 🔤 Sửa lỗi chính tả tiếng Việt theo ngữ cảnh bằng **HMM trigram \+ Viterbi**.

- 📦 **Artifacts** đã đóng gói (vocab, n-grams, transition, context totals, config) **để suy luận nhanh** không cần train lại.

- 🌐 **REST API** bằng **FastAPI** với **OpenAPI/Swagger** tại `/docs`.

- 🖥️ **Demo UI** nhanh bằng **Gradio** (nhập văn bản → xem kết quả \+ highlight khác biệt).

- 📓 Notebook đầy đủ: **thống kê dữ liệu**, **sinh lỗi → tạo tập train (5M câu)**, **tiền xử lý**, **huấn luyện & xuất artifacts**, **thử nghiệm biến thể (Laplace, Kneser–Ney, boosted emissions, seq2seq with attention, BARTpho syllable fine-tuning, Transformer)**.

---

## **🧰 Công nghệ & Thư viện sử dụng**

- **Ngôn ngữ & Môi trường:** Python, VS Code, Kaggle.

- **NLP/Modeling:** HMM \+ Viterbi, N-gram LM với Laplace & Kneser–Ney; emission dựa trên edit distance và quy tắc lỗi (diacritics/TELEX/adjacent-key/homophone/split-merge).

- **Học máy/Học sâu:** PyTorch (Seq2Seq \+ Attention), Hugging Face Transformers (BARTpho syllable).

- **Xử lý dữ liệu:** pandas, numpy, regex/re, unicodedata; tqdm; matplotlib/plotly.

- **Serving/Demo:** FastAPI \+ Uvicorn (REST API, OpenAPI/Swagger), Gradio (UI demo).

## **📂 Cấu trúc thư mục**
Dưới đây là cấu trúc thư mục chính của dự án, giúp bạn dễ dàng điều hướng và hiểu rõ các thành phần:
```bash
Vietnamese_Spelling_Correction/
├── artifacts/                      # Artifacts phục vụ suy luận
│   ├── vocab.json                  # Từ vựng tiếng Việt
│   ├── ngrams.pkl                  # Thống kê n-gram (trigram)
│   ├── transition_prob.pkl         # Xác suất chuyển P(w_t | w_{t-1}, w_{t-2})
│   ├── context_totals.pkl          # Tổng đếm/chuẩn hoá (cho smoothing)
│   └── config.json                 # Tham số decode (ví dụ: alpha/emission_weight)
│
├── dataset/                        # Dữ liệu huấn luyện/đánh giá
│   ├── test.csv                    # Tập kiểm thử
│   ├── unique_correct_texts.csv    # Nguồn câu/từ chuẩn để sinh lỗi
│   └── NguLieuPhanTich.xlsx        # Phân tích một số ngữ liệu
│   # Ghi chú: train.csv được tạo bằng script sinh lỗi (errors-generator.ipynb)
│
├── data_stats/                     # Thống kê dữ liệu (EDA)
│   ├── test_dataset_stats.ipynb
│   └── train_dataset_stats.ipynb
│
├── preprocessing/                  # Tiền xử lý dữ liệu
│   └── Data-Preprocessing.ipynb    # Chuẩn hoá Unicode, tokenize, trie/grouping
│
├── training/                       # Huấn luyện & các biến thể mô hình
│   ├── hmm-trigram-laplace-*.ipynb
│   ├── hmm-trigram-kneser-ney-smoothing.ipynb
│   ├── hmm-trigram-boosted-emission-probabilities.ipynb
│   ├── hmm-trigram-reordered-context.ipynb
│   ├── seq2seq_with_attention.ipynb
│   └── bartpho-syllable-finetuning.ipynb
│
├── src/                            # Mã nguồn chính
│   ├── hmm_decoder.py              # HMMSpellChecker + Viterbi + load artifacts
│   └── diff.py                     # Highlight khác biệt trước/sau
│
├── api.py                          # FastAPI: POST /correct (OpenAPI tại /docs)
├── app.py                          # Gradio demo UI
├── errors-generator.ipynb          # Sinh lỗi → tạo train.csv (~5M câu)
├── requirements.txt                # Thư viện Python cần thiết
└── README.md                       # Tài liệu dự án
```
---

## **🔬 Quy trình huấn luyện & tái tạo**

1. **EDA** (`data_stats/*.ipynb`)  
   Khảo sát phân bố độ dài câu, tần suất token/loại lỗi, ký tự/dấu/TELEX để thiết kế emission & smoothing.

2. **Sinh lỗi → tạo train (mục tiêu \~5,000,000 câu)** (`errors-generator.ipynb`)

   - Nguồn chuẩn: `dataset/unique_correct_texts.csv`

   - Quy tắc: diacritics/TELEX, adjacent-key, homophone, split/merge syllables…

   - Xuất: `dataset/train.csv`

3. **Tiền xử lý** (`preprocessing/Data-Preprocessing.ipynb`)  
   Chuẩn hoá Unicode (NFC), tokenize theo âm tiết/từ, trie/grouping (nếu dùng), chuẩn hoá case/punctuations.

4. **Huấn luyện & Export artifacts** (`training/*.ipynb`)

   - Trigram LM \+ **Laplace/Kneser–Ney**; **boosted emissions**; **reordered context**

   - Export: `artifacts/{vocab.json, ngrams.pkl, transition_prob.pkl, context_totals.pkl, config.json}`

5. **Suy luận**

   - `src/hmm_decoder.py` nạp artifacts và thực hiện Viterb

   - Được gọi từ **api** (REST) và **gradio** (UI)

---

## **⚙️ Cài đặt & Sử dụng**

Để chạy dự án, hãy làm theo các bước sau:

### 1. Tạo môi trường ảo

Việc tạo môi trường ảo sẽ giúp bạn dễ dàng quản lí các phiên bản thư viện, giúp dễ cài đặt và sửa chữa, tránh lỗi phiên bản.

```bash
python -m venv venv
```

Tạo môi trường ảo với tên venv. Sau khi khởi tạo thành công, tiến hành kích hoạt môi trường ảo:

```bash
venv\Scripts\activate
```

Môi trường khởi tạo thành công sẽ hiển thị tên (venv) màu xanh trước đường dẫn.

### 2. Clone Repository

```bash
git clone https://github.com/hongphat13/Vietnamese_Spelling_Correction.git
cd Vietnamese_Spelling_Correction
```

### 3. Cài đặt các thư viện cần thiết

```bash
pip install -r requirements.txt
```

### 4. REST API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
# mở http://localhost:8000/docs
```

### 4. Gradio (demo UI nhanh)

```bash
python app.py
# mở http://127.0.0.1:7860
```

---

## 🎥 **Demo**

Dưới đây là demo minh họa giao diện sửa chính tả: nhập câu, nhấn Sửa chính tả; ứng dụng trả về câu đã chỉnh, nêu tổng số lỗi phát hiện và đánh dấu rõ từng từ sai/đã sửa

<img src="assets\demoHD.png" width="100%">
