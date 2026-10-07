# Prediction (viết TRƯỚC khi chạy model — không sửa sau khi thấy kết quả)

## Prediction 1 — Những từ nào gần nhau nhất?
- Prediction: (doctor, physician) > (cat, dog) > (car, automobile) >> (doctor, banana), (computer, football).
  Cụ thể: doctor-physician ≈ 0.9+, cat-dog ≈ 0.8+, car-automobile ≈ 0.8+, doctor-banana < 0.3.
- Reason: doctor/physician, cat/dog, car/automobile là synonym/paraphrase, xuất hiện thay thế nhau
  trong cùng mẫu câu (distributional hypothesis). hospital liên quan (related) nhưng không đồng nghĩa
  nên gần thứ hai sau physician. banana/computer/football khác domain.
- Confidence: cao (85%). Rủi ro: nếu corpus nhỏ/thiên lệch, từ hiếm (automobile) có thể vector noisy.

## Prediction 2 — Context window 2 → 5, similarity thay đổi không?
- Prediction: Có. Window=5 cho similarity synonym cao hơn và ổn định hơn window=2,
  nhưng phân biệt synonym vs relatedness kém đi (doctor-hospital cũng tăng).
  Window=2 nhạy với cú pháp vị trí, window lớn bắt topical/domain.
- Reason: window nhỏ = context cục bộ (từ kề), window lớn = gom cả câu → nhiều overlap hơn
  giữa các từ cùng domain. Trade-off: quá lớn sẽ trộn nhiễu (mọi từ y tế đều gần nhau).
- Confidence: trung bình (70%). Không chắc "luôn tốt hơn" — phụ thuộc cặp từ.

## Prediction 3 — Embedding dim 50 → 100 → 300, chất lượng có chắc tăng không?
- Prediction: Tăng nhẹ rồi bão hòa; 100 tốt hơn 50 một chút, 300 không hơn 100 đáng kể
  trên corpus nhỏ, trong khi model size (V×d) tăng 6x và dễ overfit.
- Reason: capacity ↔ data ↔ computation. Corpus vài nghìn câu không đủ để lấp 300 chiều;
  chiều thừa học nhiễu. Dim lớn chỉ phát huy khi corpus lớn + train đủ epochs + regularize.
- Confidence: trung bình (65%).

## Prediction 4 — doctor-physician có chắc gần nhau nếu corpus chỉ 100 câu?
- Prediction: Không chắc. Với 100 câu, mỗi từ xuất hiện vài lần → ước lượng co-occurrence
  phương sai lớn, negative sampling noisy, thứ tự init/random ảnh hưởng mạnh.
- Reason: embedding là thống kê; cần đủ lặp lại của pattern "X treated patient/hospital".
  Corpus nhỏ còn bị frequency bias (từ frequent như `the`, `and` chi phối) và OOV cao.
- Confidence: thấp cho khẳng định "chắc chắn" (90% tin là KHÔNG chắc chắn).
  Cần min_count thấp + nhiều epochs, nhưng vẫn không thay thế được data.
