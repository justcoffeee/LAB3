"""LAB 03 - Core implementation: word-context co-occurrence + cosine similarity.

Khong duoc su dung truc tiep Word2Vec(...) o phan core nay.
Muc tieu: context -> co-occurrence -> vector -> similarity
"""
import numpy as np
from collections import Counter


def tokenize(text):
    return text.lower().split()


def build_vocabulary(corpus_tokens, min_count=1):
    """corpus_tokens: list[list[str]]. Tra ve (vocab list, word2id dict)."""
    counter = Counter()
    for sent in corpus_tokens:
        counter.update(sent)
    vocab = sorted([w for w, c in counter.items() if c >= min_count])
    word2id = {w: i for i, w in enumerate(vocab)}
    return vocab, word2id


def build_cooccurrence_matrix(corpus_tokens, vocabulary, word2id, window=1):
    """X in R^{|V| x |V|}: rows = target, cols = context.
    window = so tu moi ben (left+right). Dem so lan dong xuat hien.
    """
    V = len(vocabulary)
    X = np.zeros((V, V), dtype=np.int64)
    for sent in corpus_tokens:
        n = len(sent)
        for i, target in enumerate(sent):
            if target not in word2id:
                continue
            ti = word2id[target]
            for j in range(max(0, i - window), min(n, i + window + 1)):
                if j == i:
                    continue
                ctx = sent[j]
                if ctx not in word2id:
                    continue
                X[ti, word2id[ctx]] += 1
    return X


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def most_similar(word, matrix, vocabulary, word2id, top_k=5):
    if word not in word2id:
        raise ValueError(f"'{word}' not in vocabulary")
    idx = word2id[word]
    target = matrix[idx].astype(float)
    scores = []
    for w, i in word2id.items():
        if w == word:
            continue
        scores.append((w, cosine_similarity(target, matrix[i].astype(float))))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def describe_matrix(X):
    return {
        "shape": X.shape,
        "vocab_size": X.shape[0],
        "nnz": int(np.count_nonzero(X)),
        "total": int(X.size),
        "sparsity": 1.0 - float(np.count_nonzero(X)) / float(X.size),
    }


if __name__ == "__main__":
    corpus = [
        "the cat eats fish",
        "the dog eats fish",
        "the cat likes milk",
        "the dog likes meat",
    ]
    corpus_tokens = [tokenize(s) for s in corpus]
    for window in [1, 2, 5]:
        vocab, word2id = build_vocabulary(corpus_tokens)
        X = build_cooccurrence_matrix(corpus_tokens, vocab, word2id, window=window)
        info = describe_matrix(X)
        print(f"=== window={window} ===")
        print("vocab:", vocab)
        print("matrix size:", info["shape"], "nnz:", info["nnz"], f"sparsity={info['sparsity']:.3f}")
        print("cat vec:", X[word2id["cat"]].tolist())
        print("dog vec:", X[word2id["dog"]].tolist())
        print("eats vec:", X[word2id["eats"]].tolist())
        print("likes vec:", X[word2id["likes"]].tolist())
        print("cos(cat,dog) =", round(cosine_similarity(X[word2id["cat"]], X[word2id["dog"]]), 4))
        print("cos(cat,fish)=", round(cosine_similarity(X[word2id["cat"]], X[word2id["fish"]]), 4))
        print("cos(eats,likes)=", round(cosine_similarity(X[word2id["eats"]], X[word2id["likes"]]), 4))
        print("most_similar(cat):", most_similar("cat", X, vocab, word2id, top_k=3))
        print()
