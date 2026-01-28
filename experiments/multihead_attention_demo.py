import numpy as np
import matplotlib.pyplot as plt


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def scaled_dot_attention(Q, K, V):
    d_k = Q.shape[-1]
    A = softmax((Q @ K.T) / np.sqrt(d_k))
    return A, A @ V


def multihead_attention(X, num_heads=4, d_k=16, d_v=16, seed=0):
    """
    Toy MHA: random projection matrices (not trained).
    X: [seq_len, d_model]
    Returns:
      head_attentions: list of [seq_len, seq_len]
      out: [seq_len, num_heads * d_v]
    """
    rng = np.random.default_rng(seed)
    seq_len, d_model = X.shape

    head_attentions = []
    head_outputs = []

    for h in range(num_heads):
        Wq = rng.standard_normal((d_model, d_k))
        Wk = rng.standard_normal((d_model, d_k))
        Wv = rng.standard_normal((d_model, d_v))

        Q = X @ Wq
        K = X @ Wk
        V = X @ Wv

        A, O = scaled_dot_attention(Q, K, V)
        head_attentions.append(A)
        head_outputs.append(O)

    out = np.concatenate(head_outputs, axis=-1)
    return head_attentions, out


def main():
    np.random.seed(42)
    seq_len = 14
    d_model = 32

    X = np.random.randn(seq_len, d_model)

    heads, out = multihead_attention(X, num_heads=4, d_k=16, d_v=16, seed=42)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for i, ax in enumerate(axes):
        ax.imshow(heads[i])
        ax.set_title(f"Head {i+1}")
        ax.set_xlabel("Key pos")
        ax.set_ylabel("Query pos")

    plt.tight_layout()
    plt.savefig("figures/multihead_attention_heads.png", dpi=200)
    plt.close()

    print("Saved: figures/multihead_attention_heads.png")
    print("Output shape:", out.shape)


if __name__ == "__main__":
    main()
