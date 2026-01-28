import numpy as np
import matplotlib.pyplot as plt


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def main() -> None:
    np.random.seed(0)

    seq_len = 10
    d_k = 16

    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)
    V = np.random.randn(seq_len, d_k)

    scores = (Q @ K.T) / np.sqrt(d_k)
    attention = softmax(scores, axis=-1)
    _output = attention @ V  # computed for completeness

    plt.figure()
    plt.imshow(attention)
    plt.colorbar()
    plt.title("Toy Self-Attention Heatmap")
    plt.xlabel("Key positions")
    plt.ylabel("Query positions")
    plt.tight_layout()

    plt.savefig("figures/attention_heatmap.png", dpi=200)
    plt.close()

    print("Saved figures/attention_heatmap.png")


if __name__ == "__main__":
    main()
