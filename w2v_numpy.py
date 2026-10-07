"""Word2Vec Skip-gram with Negative Sampling (numpy-only).
Ly do numpy-only: moi truong Python 3.14 chua co wheel gensim (can MSVC de build C-ext),
nen implement chuan Skip-gram + NEG de dam bao tai hien duoc thi nghiem window/dim.
API giu gan voi gensim: vector_size, window, min_count, epochs, sg=1, negative, seed.
"""
import numpy as np
from collections import Counter
import random


def build_vocab(tokenized, min_count=2):
    cnt = Counter()
    for s in tokenized:
        cnt.update(s)
    vocab = sorted([w for w, c in cnt.items() if c >= min_count])
    w2i = {w: i for i, w in enumerate(vocab)}
    freq = np.array([cnt[w] for w in vocab], dtype=float)
    return vocab, w2i, cnt, freq


def get_training_pairs(tokenized, w2i, window):
    pairs = []
    for sent in tokenized:
        ids = [w2i[w] for w in sent if w in w2i]
        for i, t in enumerate(ids):
            L = max(0, i - window)
            R = min(len(ids), i + window + 1)
            for j in range(L, R):
                if j == i:
                    continue
                pairs.append((t, ids[j]))
    return pairs


def sigmoid(x):
    # stable
    x = np.clip(x, -6, 6)
    return 1.0 / (1.0 + np.exp(-x))


class SkipGramNEG:
    def __init__(self, vocab_size, dim=100, seed=42, neg_pow=0.75):
        rng = np.random.default_rng(seed)
        self.W_in = (rng.random((vocab_size, dim)) - 0.5) / dim  # target embeddings
        self.W_out = np.zeros((vocab_size, dim))  # context embeddings
        self.neg_pow = neg_pow

    def train(self, pairs, freq, epochs=10, lr=0.05, negative=5, seed=42):
        rng = np.random.default_rng(seed)
        V = self.W_in.shape[0]
        unigram = freq ** self.neg_pow
        unigram /= unigram.sum()
        N = len(pairs)
        for ep in range(epochs):
            idx = rng.permutation(N)
            total_loss = 0.0
            for k in idx:
                t, c = pairs[k]
                # positive + negatives (avoid target itself as neg for stability)
                negs = rng.choice(V, size=negative, p=unigram).tolist()
                # forward
                h = self.W_in[t]  # (dim,)
                pos_score = float(h @ self.W_out[c])
                pos_sig = sigmoid(pos_score)
                g = lr * (1 - pos_sig)
                # accumulate grad for h
                grad_h = g * self.W_out[c].copy()
                self.W_out[c] += g * h
                loss = -np.log(pos_sig + 1e-10)
                for n in negs:
                    if n == c:
                        continue
                    s = float(h @ self.W_out[n])
                    sig = sigmoid(s)
                    g2 = -lr * sig
                    grad_h += g2 * self.W_out[n]
                    self.W_out[n] += g2 * h
                    loss += -np.log(1 - sig + 1e-10)
                self.W_in[t] += grad_h
                total_loss += loss
            # decay lr nhe
            lr = max(0.005, lr * 0.95)
        return self

    @property
    def vectors(self):
        return self.W_in


def cosine(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


def most_similar(word, vocab, w2i, vecs, top_k=5):
    i = w2i[word]
    v = vecs[i]
    out = []
    for w, j in w2i.items():
        if w == word:
            continue
        out.append((w, cosine(v, vecs[j])))
    out.sort(key=lambda x: x[1], reverse=True)
    return out[:top_k]


def analogy(pos, neg, vocab, w2i, vecs, top_k=5):
    v = np.zeros(vecs.shape[1])
    for w in pos:
        v += vecs[w2i[w]]
    for w in neg:
        v -= vecs[w2i[w]]
    out = []
    excl = set(pos) | set(neg)
    for w, j in w2i.items():
        if w in excl:
            continue
        out.append((w, cosine(v, vecs[j])))
    out.sort(key=lambda x: x[1], reverse=True)
    return out[:top_k], v


def build_demo_corpus(repeat=40, seed=42):
    """Corpus du lon de hoc duoc doctor~physician, car~automobile, king~queen..."""
    rng = random.Random(seed)
    base = [
        "the doctor treated the patient in the hospital",
        "the physician treated the patient in the hospital",
        "the doctor works in the hospital with the nurse",
        "the physician works in the hospital with the nurse",
        "the patient has a disease and needs treatment",
        "the nurse helps the doctor with clinical therapy",
        "the nurse helps the physician with clinical therapy",
        "medical treatment helps the patient recover",
        "the cat eats fish and likes milk",
        "the dog eats meat and likes fish",
        "the cat and the dog are pets",
        "the car is fast and the automobile is fast",
        "i drive my car to the city",
        "i drive my automobile to the city",
        "the computer runs software and hardware",
        "the laptop runs software like the computer",
        "football is a sport with a ball",
        "soccer is a sport like football",
        "the banana is a fruit",
        "the apple is a fruit like the banana",
        "the king rules the kingdom with the queen",
        "the man walks with the woman in the kingdom",
        "the king and the queen live in the palace",
        "the man and the woman live in the village",
        "paris is the capital of france",
        "rome is the capital of italy",
        "france and italy are countries in europe",
        "the king is a man and the queen is a woman",
        "the doctor diagnosed the disease of the patient",
        "the physician diagnosed the disease of the patient",
        "hospital therapy treats disease",
        "clinical hospital treats patient",
    ]
    sents = []
    for _ in range(repeat):
        for s in base:
            sents.append(s)
    # them nhieu bien the dao context de tang overlap
    extra = [
        "doctor physician hospital patient nurse disease treatment therapy clinical",
        "car automobile drive city fast road",
        "king queen man woman palace kingdom village",
        "paris france rome italy europe capital country",
        "cat dog pet fish milk meat",
        "computer laptop software hardware",
        "football soccer sport ball game",
        "banana apple fruit eat",
    ]
    for _ in range(repeat * 2):
        s = rng.choice(extra)
        toks = s.split()
        rng.shuffle(toks)
        sents.append(" ".join(toks))
    rng.shuffle(sents)
    return [s.split() for s in sents]
