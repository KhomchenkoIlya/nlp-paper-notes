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
## Attention

### Attention mechanisms comparison
![Attention mechanisms](figures/attention_mechanisms_comparison.png)

### Multi-head attention (toy)
![Multi-head heads](figures/multihead_attention_heads.png)
---

## Gated Attention (toy study)

Ноутбук: `notebooks/gated_attention_toy.ipynb`

Идея: attention отвечает за *куда смотреть*, а **gate** — за *сколько информации пропускать*.  
Это добавляет управляемость и селективность, что используется в современных архитектурах (gated blocks, mixture-of-experts и др.).

<p align="center">
  <img src="figures/gated_attention_gate_values.png" width="420">
</p>

<p align="center">
  <img src="figures/gated_attention_output_magnitude.png" width="420">
</p>


---

## Positional Encoding as Inductive Bias (toy study)

Ноутбук: `notebooks/positional_encoding_and_inductive_bias.ipynb`

Идея: показать, почему self-attention без позиции “не знает геометрию”, и как absolute PE / relative bias вводят зависимость от расстояния `|i-j|` (физическая интуиция: “взаимодействие зависит от дистанции”).

<p align="center">
  <img src="figures/attention_comparison.png" width="900">
</p>

<p align="center">
  <img src="figures/attention_vs_distance.png" width="650">
</p>


Заметка (paper note): `papers/positional_encoding_bias.md`
---

## Research-style summary

В рамках toy-исследования я сформулировал гипотезу, что relative positional bias в self-attention можно интерпретировать как дискретный аналог потенциального взаимодействия, зависящего от расстояния между элементами последовательности. 
Я ввёл простую количественную метрику — “энергию внимания” — и показал, что форма positional bias напрямую влияет на локальность и энергетический режим attention.
Подход мотивирован аналогиями с физическими моделями взаимодействующих частиц и демонстрирует перенос исследовательского мышления между физикой и NLP.
