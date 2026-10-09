"""Tiny embedding similarity search using hashed bag-of-words vectors."""

import math
import zlib

DIM = 128


def embed(text, dim=DIM):
    """Return an L2-normalized hashed bag-of-words vector for text."""
    vec = [0.0] * dim
    for token in text.lower().split():
        vec[zlib.crc32(token.encode()) % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec


def cosine(a, b):
    """Cosine similarity between two equal-length vectors."""
    return sum(x * y for x, y in zip(a, b))


class Index:
    """In-memory index supporting top-k cosine similarity queries."""

    def __init__(self, dim=DIM):
        self.dim = dim
        self.items = []

    def add(self, text):
        self.items.append((text, embed(text, self.dim)))

    def search(self, query, k=3):
        q = embed(query, self.dim)
        scored = [(cosine(q, v), t) for t, v in self.items]
        scored.sort(reverse=True)
        return scored[:k]


def main():
    docs = [
        "the cat sat on the warm mat",
        "a dog barked at the mail carrier",
        "python is a programming language",
        "machine learning models produce embeddings",
        "the kitten napped in the sun",
    ]
    index = Index()
    for doc in docs:
        index.add(doc)

    query = "a small cat sleeping"
    print(f"query: {query!r}")
    for score, text in index.search(query, k=3):
        print(f"  {score:.3f}  {text}")


if __name__ == "__main__":
    main()