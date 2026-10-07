# Reflection — Từ Word2Vec đến Transformer

## Bảng so sánh

| Representation | Context-dependent? | Sparse / Dense | Một từ có nhiều vector? |
|----------------|--------------------|----------------|--------------------------|
| TF-IDF (LAB01) | Không. 1 doc = 1 vector đếm có trọng số, thứ tự từ bị mất, `doctor` trong mọi doc đều đóng góp cùng một chiều. | Sparse (V≈10k-100k, hầu hết 0). | Không. Mỗi từ là 1 chiều cố định, không có vector ngữ nghĩa riêng. |
| Co-occurrence count (LAB03 Exp1) | Một phần. Vector = phân bố context trong window, nhưng gộp toàn corpus thành 1 vector duy nhất. | Sparse (V×V, sparsity 0.44-0.72 trong demo 8 từ; với V=100k → 10^10 entries, không lưu nổi). | Không. `bank` chỉ có 1 hàng duy nhất dù xuất hiện ở 2 nghĩa. |
| Word2Vec static (LAB03 Exp2) | Không (static). Học từ context nhưng sau train, mỗi từ type chỉ có 1 vector, tra cứu là lookup. | Dense (50-300D, hầu hết khác 0). V×d = 76×50 ≈ 3.8k params trong demo; V=100k,d=300 → 30M params, vẫn rẻ hơn 10^10. | Không. Vẫn 1 vector cho `bank`. |
| Contextual embedding (Transformer) | Có. Vector = hàm của cả câu (self-attention), cùng từ type khác câu → khác vector. | Dense (768D+), thậm chí contextualized theo từng token occurrence. | Có (mỗi occurrence 1 vector; có thể cluster thành nhiều nghĩa). |

## Tại sao `bank` cần contextual representation?

Hai câu:
- `I deposited money in the bank.` (ngân hàng)
- `We sat on the river bank.` (bờ sông)

Static Word2Vec chỉ có 1 vector `v(bank)` = trung bình có trọng số của cả hai nghĩa
(phụ thuộc nghĩa nào frequent hơn trong corpus tài chính vs địa lý).
Hệ quả: similarity(bank, money) và similarity(bank, river) đều lưng chừng,
không câu nào được biểu diễn đúng; tìm kiếm ngữ nghĩa / phân loại dễ sai.

Contextual representation giải quyết bằng cách tính `v(bank | câu)`:
attention gom `money/deposited` ở câu 1 → vector nghiêng về finance;
gom `river/sat` ở câu 2 → vector nghiêng về geography.
Nhờ đó cùng một từ type có nhiều điểm trong không gian, có thể phân biệt nghĩa
theo ngữ cảnh (WSD ngầm), làm tiền đề cho ELMo → BERT → Transformer → RAG.

## Mạch 3 lab
LAB01 (sparse lexical) → LAB02 (probability/next-word) → LAB03 (dense semantic).
LAB04 tự nhiên tiếp nối: TF-IDF / mean-embedding → document representation → classification,
rồi tới contextual encoders.
