# LAB 03 — Word Representations and Embeddings
Môn: Xử lý ngôn ngữ tự nhiên và ứng dụng | Học kỳ I - 2026

## 1. Cấu trúc nộp
```
lab03/
├── README.md
├── calculations.md      # Bài 1-4, CBOW/Skip-gram, analogy (tính tay)
├── prediction.md        # Prediction trước experiment
├── cooccurrence.py      # Core: build_vocabulary, build_cooccurrence_matrix, cosine_similarity, most_similar
├── w2v_numpy.py         # Skip-gram NEG numpy-only (thay gensim vì Python 3.14 chưa có wheel, cần MSVC)
├── word_embedding.ipynb # Exp1 co-occurrence, Exp2 Word2Vec, window/dim, eval, semantic search
├── results.csv          # similarity + analogy scores (số thật từ chạy code)
├── error_analysis.md    # 3 đúng + 3 sai
└── reflection.md        # static -> contextual
```

## 2. Cách chạy
```bash
python lab03/cooccurrence.py
# Mở word_embedding.ipynb để chạy từng experiment (mỗi model ~40s trên CPU)
```
Mở `word_embedding.ipynb` để xem từng bước có giải thích.

## 3. Corpus & hyperparams (Exp2)
- Corpus tự xây ~600 câu (repeat=12 trên 32 templates + câu shuffle), phủ
  medical / pets / vehicles / computers / sports / fruit / royalty / geography.
  Ví dụ: `the doctor treated the patient in the hospital` song song với
  `the physician ...` để tạo synonymy có kiểm chứng.
- Vocab: V=76 (min_count=2). Embedding: Skip-gram + negative sampling k=5,
  lr=0.05 (decay 0.95), epochs=8, seed=42, sg=1.
- Chính: vector_size=50, window=5, min_count=2, epochs=8.
- So sánh window ∈ {2,5,10}, dim ∈ {50,100,300}.

## 4. Kết quả chính (xem results.csv)
- Co-occurrence (corpus 4 câu, V=8): matrix 8×8, nnz 18/30/36 với window 1/2/5;
  cos(cat,dog)=1.0/0.875/0.875.
- Word2Vec main (dim50/win5): doctor-physician 0.980 (rank1), cat-dog 0.965,
  car-automobile 0.929, doctor-hospital 0.707 (related), computer-banana 0.532 (cao bất ngờ).
- Window: win2 tách domain kém (computer-banana 0.812), win5/10 tốt hơn (~0.52);
  analogy king-man+woman: win2 queen rank4, win5/10 rank1.
- Dim: 50→100→300 cải thiện marginal (+0.005), size 30KB→59KB→178KB.
  → capacity cần data tương xứng; corpus nhỏ không lấp nổi 300 chiều.
- Analogy Paris-France+Italy≈Rome: corpus có đủ pattern nhưng tần suất thấp hơn
  royalty nên kết quả yếu hơn king-queen (ghi trong notebook).
- Semantic search `medical treatment` trả về docs chứa therapy/clinical/patient
  lên top nhờ mean-embedding + cosine (chi tiết trong notebook).

## 5. Hạn chế static embedding
Xem error_analysis.md + reflection.md: 1 vector/từ trộn synonymy với topical
relatedness, không tách polysemy (`bank`), nhạy cảm window/frequency/corpus size.

## 6. AI assistance statement
- Tool: Muse Spark (opencode agent) + sklearn/matplotlib.
- Purpose: giải thích API, debug build gensim thất bại, dựng khung code/thí nghiệm,
  visualization gợi ý, kiểm tra tính toán cosine.
- Generated vs modified: calculations/predictions/interpretations trong các file .md
  đã được đối chiếu với số liệu chạy thật từ `cooccurrence.py` / `run_exp.py`
  (xem results.csv) và hiệu chỉnh thủ công.
- Verification: mọi con số trong results.csv đều từ output thực tế của code
  (copy từ log chạy ngày 06/10/2026 với seed=42, repeat=12, epochs=8); công thức cosine tính tay đã check bằng numpy.
