# Error analysis — 3 đúng + 3 sai/bất ngờ (model chính: Skip-gram NEG, dim=50, window=5, min_count=2, epochs=8)

Corpus: ~600 câu tự xây (medical, pets, vehicles, computers, sports, fruit, royalty, geography),
lặp templates + shuffle, vocab V=76. Training objective: Skip-gram negative sampling (k=5),
lr=0.05 decay 0.95. Evidence kiểm tra bằng tần suất câu chứa từ.

## 3 trường hợp ĐÚNG

### 1. doctor → physician (0.9802, rank 1/76)
- Observed: most_similar(doctor) = physician 0.9802, tiếp theo hospital 0.707, nurse 0.663.
- Expected: physician là synonym thay thế trực tiếp cho doctor.
- Explanation: hai từ chia sẻ gần như 100% mẫu câu (`_ treated the patient in the hospital`,
  `_ works in the hospital with the nurse`, `_ diagnosed the disease...`).
- Evidence: mỗi template xuất hiện 12 lần cho cả doctor và physician → phân bố context
  gần đồng nhất → cosine ≈ 1. Đúng theo distributional hypothesis.

### 2. cat → dog (0.9648)
- Observed: cat-dog 0.9648, top neighbors lẫn nhau.
- Expected: cùng pets, context `eats fish/meat`, `likes milk/fish`, `are pets`.
- Explanation: overlap context cao, tần suất cân bằng.
- Evidence: co-occurrence tay window=1 đã cho cos=1.0; Word2Vec giữ được quan hệ này
  dù có thêm nhiễu từ câu shuffle.

### 3. car → automobile (0.9292)
- Observed: car-automobile 0.9292.
- Expected: paraphrase, cùng mẫu `i drive my _ to the city`, `_ is fast`.
- Explanation: từ hiếm nhưng xuất hiện đối xứng (mỗi template lặp đều cho cả hai),
  Skip-gram tốt với rare words vì mỗi cặp (target,context) là 1 update.
- Evidence: min_count=2 giữ lại automobile; nếu min_count=5 từ này đã bị loại → OOV.

## 3 trường hợp SAI / BẤT NGỜ

### 1. computer → banana = 0.532 (quá cao)
- Observed: 0.53 dù khác domain hoàn toàn (tech vs fruit). Window=2 còn tệ hơn: 0.812.
- Expected: < 0.3 (như doctor-banana 0.22).
- Possible explanation: (a) câu shuffle `computer laptop software hardware` và
  `banana apple fruit eat` quá ngắn + trộn thứ tự, (b) cả hai đều xuất hiện trong
  mẫu `X is a ...` / `X runs ...`? Không — nguyên nhân chính là **corpus nhỏ + window lớn
  gom nhiễu + từ nối chung (`is,a,the,and`)** khiến mọi danh từ đều có một phần context chung.
- Evidence: window 2→5→10: 0.812 → 0.532 → 0.520. Window lớn tách domain tốt hơn
  (topical), window nhỏ chỉ thấy từ kề (`is,a`) nên lẫn. Vẫn còn cao → cần thêm data
  phân biệt, subsampling từ frequent, hoặc tăng negative samples.

### 2. Analogy king-man+woman → palace cạnh tranh queen (win2: palace 0.818 > queen 0.777)
- Observed: window=2 queen chỉ rank 4; window=5/10 queen rank 1 nhưng palace vẫn rank 2
  (0.624 vs 0.584, cách biệt nhỏ).
- Expected: queen vượt trội.
- Possible explanation: `palace` chỉ đi với king/queen (`live in the palace`) nên vector
  nó cũng nằm giữa cluster royalty; phép trừ `king-man` chưa tách sạch "gender direction"
  khỏi "royalty/place direction". Corpus không có đủ cặp tương phản
  (king:queen = man:woman = paris:rome?) với tần suất cân bằng.
- Evidence: `kingdom/village/live/palace` đều có cosine 0.58-0.81 với vector analogy
  → không gian học được **relatedness** (cùng chủ đề hoàng gia) chứ không phải
  quan hệ giới tính thuần túy. Đây là minh chứng: vector arithmetic = pattern hình học,
  không phải hiểu biết như người.

### 3. doctor → hospital (0.707) vs doctor → physician (0.980): relatedness lẫn với similarity
- Observed: hospital/nurse/treatment đều > 0.6 với doctor.
- Expected (nếu muốn synonym detector): related words phải thấp hơn synonym rõ rệt.
- Possible explanation: Skip-gram với window=5 gom cả câu → mọi từ y tế cùng câu đều
  thành context của nhau → related và similar bị trộn. Đây là giới hạn của static embedding:
  1 vector/một từ phải gánh cả synonymy, topical relatedness, cú pháp.
- Evidence: tăng window 5→10, doctor-hospital giảm 0.707→0.639 (bớt topical?) nhưng
  đồng thời king-queen cũng giảm 0.844→0.739 → không có window tối ưu cho mọi cặp.

## Tổng kết nguyên nhân (map với đề)
- corpus nhỏ (600 câu, V=76) → phương sai cao, từ hiếm noisy.
- domain bias: medical templates lặp nhiều nhất → cluster y tế quá chặt.
- frequency: `the/and/a` xuất hiện khắp nơi → mọi danh từ đều hơi gần nhau.
- polysemy: chưa test ở đây (xem reflection: `bank`), nhưng static vector duy nhất
  không tách nghĩa được.
- window: nhỏ → lẫn cú pháp; lớn → lẫn chủ đề. Không có giá trị luôn tốt.
- insufficient training: 8 epochs + dim 50 là đủ demo, chưa hội tụ hoàn toàn;
  tăng epochs/dim trên data nhỏ chỉ cải thiện marginal (xem dim 50→300: +0.005).
