# NLP Paper Notes & Toy Experiments

Небольшой исследовательский репозиторий с конспектами статей по NLP / LLM
и воспроизведением ключевых идей в виде простых экспериментов.

## 📄 Papers

- `papers/icml_transformer_attention.md`  
  Краткий разбор self-attention и scaled dot-product attention
  (по мотивам Transformer-архитектур, ICML / NeurIPS).
- `papers/icml_llm_scaling_laws.md`  
  Краткий обзор scaling laws для больших языковых моделей (ICML).
- `papers/attention_mechanisms_comparison.md`  
  Обзор и сравнение различных механизмов внимания.

## 🧪 Experiments

- `experiments/toy_attention_experiment.py`  
  Минимальная реализация self-attention (NumPy) и визуализация
  attention-матрицы.
- `experiments/softmax_temperature_experiment.py`  
  Визуализация влияния температуры на softmax-распределение.
- `experiments/compare_attention_mechanisms.py`  
  Сравнение dot-product, scaled и additive attention с визуализацией.

Пример результата:

<p align="center">
  <img src="figures/attention_heatmap.png" width="420">
</p>

## 🛠 How to run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 experiments/toy_attention_experiment.py
