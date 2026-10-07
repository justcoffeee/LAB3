# LAB 03 — Calculations (Theory & hand computation, no Python)

## Bài 1 — Co-occurrence, window k=1

Corpus (lowercased, tokenized by space):
1. the cat eats fish
2. the dog eats fish
3. the cat likes milk
4. the dog likes meat

Quy ước: window k=1 nghĩa là 1 từ bên trái + 1 từ bên phải của target (trong cùng câu).
Đếm cộng dồn trên cả 4 câu. Kết quả kiểm chứng bằng `cooccurrence.py`:

| target | context counts (k=1) |
|--------|----------------------|
| cat   | the:2, eats:1, likes:1 |
| dog   | the:2, eats:1, likes:1 |
| eats  | cat:1, dog:1, fish:2 |
| likes | cat:1, dog:1, milk:1, meat:1 |

Chi tiết từng câu:
- S1 `the|cat|eats|fish`: cat->{the:1,eats:1}; eats->{cat:1,fish:1}
- S2 `the|dog|eats|fish`: dog->{the:1,eats:1}; eats->{dog:1,fish:1} (cộng dồn eats->{cat:1,dog:1,fish:2})
- S3 `the|cat|likes|milk`: cat+= {the:1,likes:1} => cat={the:2,eats:1,likes:1}; likes->{cat:1,milk:1}
- S4 `the|dog|likes|meat`: dog+= {the:1,likes:1} => dog={the:2,eats:1,likes:1}; likes+= {dog:1,meat:1}

Nếu loại stopword `the` (như bảng minh họa ở mục 5 của đề, vocab = {cat,dog,eats,likes,fish,milk,meat}):

- vec(cat)   = [eats:1, likes:1, rest 0]
- vec(dog)   = [eats:1, likes:1, rest 0]  → **giống hệt vec(cat)**
- vec(eats)  = [cat:1, dog:1, fish:2]
- vec(likes) = [cat:1, dog:1, milk:1, meat:1]

Nhận xét (LO1/LO2): `cat` và `dog` có phân bố context giống nhau → vector giống nhau
→ cosine(cat,dog)=1.0 (đã verify bằng code). Đây chính là distributional hypothesis:
từ xuất hiện trong context tương tự thì vector gần nhau.

Ma trận đầy đủ (giữ `the`, V=8, window=1, thứ tự vocab
`[cat,dog,eats,fish,likes,meat,milk,the]`):

- cat  = [0,0,1,0,1,0,0,2]
- dog  = [0,0,1,0,1,0,0,2]
- eats = [1,1,0,2,0,0,0,0]
- likes= [1,1,0,0,0,1,1,0]

## Bài 2 — Cosine similarity

Cho x=[1,2,1], y=[2,4,2].

```
dot = 1*2 + 2*4 + 1*2 = 2+8+2 = 12
||x|| = sqrt(1+4+1) = sqrt(6)
||y|| = sqrt(4+16+4) = sqrt(24) = 2*sqrt(6)
cos = 12 / (sqrt(6)*2*sqrt(6)) = 12 / (2*6) = 1.0
```

Ý nghĩa: y = 2x, cùng hướng, khác độ lớn. Cosine chỉ đo **góc**, không đo độ lớn.
Trong NLP điều này tốt vì văn bản dài/ngắn khác nhau (tần suất cao hơn) vẫn có thể
cùng chủ đề → cùng hướng vector. Ngược lại, nhược điểm là cosine bỏ qua thông tin
magnitude (ví dụ confidence/frequency) nếu magnitude có ý nghĩa.

## Bài 3 — Semantic similarity

Dự đoán trước khi tính: `doctor` gần `physician` hơn (cùng lĩnh vực y tế),
`banana` khác lĩnh vực (thực phẩm) nên xa. Tức là cos(doctor,physician) >> cos(doctor,banana).

Tính:

```
v_doctor    = [0.8, 0.1, 0.7]
v_physician = [0.7, 0.2, 0.8]
v_banana    = [-0.2, 0.9, -0.1]

dot(d,p) = 0.8*0.7 + 0.1*0.2 + 0.7*0.8 = 0.56+0.02+0.56 = 1.14
||d|| = sqrt(0.64+0.01+0.49)=sqrt(1.14)≈1.0677
||p|| = sqrt(0.49+0.04+0.64)=sqrt(1.17)≈1.0817
cos(d,p) = 1.14/(1.0677*1.0817) ≈ 0.9871

dot(d,b) = 0.8*(-0.2)+0.1*0.9+0.7*(-0.1) = -0.16+0.09-0.07 = -0.14
||b|| = sqrt(0.04+0.81+0.01)=sqrt(0.86)≈0.9274
cos(d,b) = -0.14/(1.0677*0.9274) ≈ -0.1414
```

Kết luận: cos(doctor,physician)≈**0.987** (gần 1, cùng hướng),
cos(doctor,banana)≈**-0.14** (gần trực giao / hơi ngược hướng).
Khớp dự đoán. Đây là cách embedding mã hóa synonymy: từ cùng domain/context
có vector cùng hướng.

## Bài 4 — Sparse vs Dense (V=10000)

1. Biểu diễn nào sparse? → **word-context 10000 chiều với 30 non-zero** là sparse
   (99.7% entries = 0).
2. Biểu diễn nào dense? → **embedding 300 chiều, hầu hết khác 0** là dense.
3. Vì sao dense thuận lợi cho semantic similarity?
   - Mỗi chiều không còn là "một từ context cụ thể" mà là latent feature
     (tổng hợp từ nhiều context).
   - Hai từ chưa từng đồng xuất hiện trực tiếp vẫn có thể gần nhau nếu context
     của chúng tương tự (generalization, chống sparsity).
   - Cosine trong không gian 300D ổn định hơn so với 10000D thưa (nơi 2 vector
     ngẫu nhiên hầu như trực giao, norm bị chi phối bởi vài từ frequent như `the`).
   - Tính toán nhẹ: 300 mults vs 10000 mults; dễ đưa vào neural net downstream.
4. Dense có chắc chắn tốt hơn mọi bài toán?
   - **Không.** Sparse giữ tính diễn giải (mỗi chiều = 1 từ), tốt cho retrieval
     chính xác từ khóa (lexical match), ít cần data, không cần train.
     Dense cần nhiều data, khó interpret, có thể mất thông tin hiếm (rare words),
     tốn train/tune, và trên corpus nhỏ dễ overfit / học bias.
     Chọn sparse vs dense phụ thuộc task, data, yêu cầu interpretability.

## Bài tập prediction — CBOW vs Skip-gram

Câu: `the cat eats fish`, window=1.

### CBOW (context → target). Mỗi target + 2 context words (trừ biên):

| context (input) | target (output) |
|-----------------|-----------------|
| (the, eats)     | cat |
| (the, cat, fish) rút gọn window=1: (cat, fish) | eats |
| (eats) + biên? chi tiết: |  |
- target=the: context=[cat] (đầu câu, chỉ 1 bên)
- target=cat: context=[the, eats]
- target=eats: context=[cat, fish]
- target=fish: context=[eats]

### Skip-gram (target → từng context word). Mỗi cặp (target, 1 context):

- (the → cat), (cat → the), (cat → eats), (eats → cat),
  (eats → fish), (fish → eats)

Tổng 6 pairs. CBOW có 4 samples (mỗi sample nhiều inputs),
Skip-gram có 6 samples (mỗi sample 1 input → 1 output).
CBOW mượt hơn (average context), train nhanh; Skip-gram tốt hơn cho rare words
vì mỗi cặp context là 1 update riêng.

CBOW khác Skip-gram ở **chiều dự đoán**: CBOW gom context để đoán giữa,
Skip-gram lấy giữa để đoán xung quanh.

## Bài tập tính analogy

```
king  = [8,2,7]; man = [5,1,5]; woman = [5,3,5]
king - man + woman = [8-5+5, 2-1+3, 7-5+5] = [8,4,7]
```

Vector [8,4,7] giữ thành phần "royalty" của king (8,7) nhưng dịch chuyển chiều
thứ 2 (giới tính?) từ 2 → 4 (gần woman=3 hơn man=1).
Kỳ vọng nearest neighbor là `queen`. Giải thích: phép cộng/trừ vector mã hóa
quan hệ tuyến tính (gender direction ≈ woman-man). **Lưu ý:** đây chỉ là pattern
hình học trong không gian học được, không phải model "hiểu" quan hệ như người.
Trên corpus nhỏ / bias, analogy dễ fail (xem error_analysis.md).
