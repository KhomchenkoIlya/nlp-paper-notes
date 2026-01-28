import numpy as np
import matplotlib.pyplot as plt


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def dot_product_attention(Q, K):
    return softmax(Q @ K.T)


def scaled_dot_product_attention(Q, K):
    d_k = Q.shape[1]
    return softmax((Q @ K.T) / np.sqrt(d_k))


def additive_attention(Q, K):
    # Simple additive-style scoring (toy, not trained)
    Wq = np.random.randn(Q.shape[1], Q.shape[1])
    Wk = np.random.randn(K.shape[1], K.shape[1])
    v = np.random.randn(Q.shape[1])

    scores = np.zeros((Q.shape[0], K.shape[0]))
    for i in range(Q.shape[0]):
        for j in range(K.shape[0]):
            scores[i, j] = np.dot(v, np.tanh(Q[i] @ Wq + K[j] @ Wk))
    return softmax(scores)


def main():
    np.random.seed(42)

    seq_len = 12
    d_k = 16

    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)

    attn_dot = dot_product_attention(Q, K)
    attn_scaled = scaled_dot_product_attention(Q, K)
    attn_add = additive_attention(Q, K)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(attn_dot)
    axes[0].set_title("Dot-Product")

    axes[1].imshow(attn_scaled)
    axes[1].set_title("Scaled Dot-Product")

    axes[2].imshow(attn_add)
    axes[2].set_title("Additive (toy)")

    for ax in axes:
        ax.set_xlabel("Key positions")
        ax.set_ylabel("Query positions")

    plt.tight_layout()
    plt.savefig("figures/attention_mechanisms_comparison.png", dpi=200)
    plt.close()

    print("Saved: figures/attention_mechanisms_comparison.png")


if __name__ == "__main__":
    main()
