# Notes: Transformer Attention Mechanism

## Paper
"Attention Is All You Need" — Vaswani et al.

## Core idea
The paper proposes replacing recurrence and convolution with self-attention,
allowing models to process sequences in parallel and efficiently model long-range dependencies.

## Self-attention
The attention mechanism is defined as:

Attention(Q, K, V) = softmax(QKᵀ / √dₖ) V

where queries, keys, and values are learned linear projections of the input.

## Practical observations
- Computational complexity scales quadratically with sequence length
- Positional encodings are required to inject order information
- Multi-head attention improves representational capacity

## Relevance to modern LLMs
Most modern large language models are based on stacked Transformer blocks,
making self-attention a core component of current NLP systems.
