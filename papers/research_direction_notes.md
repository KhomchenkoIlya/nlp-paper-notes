# Research Direction Notes: Attention & Tokenization Inductive Biases

## Мотивация

Современные LLM основаны на self-attention, однако на практике его поведение
сильно определяется **индуктивными байасами**, встроенными в архитектуру и preprocessing:

- позиционные кодировки и relative bias,
- приближения attention (linear / kernel),
- механизмы управления потоком информации (gating),
- выбор токенизации и предобработки текста.

Цель данного репозитория — **на toy-экспериментах показать**, как эти решения
влияют на:
- геометрию внимания,
- вычислительную сложность,
- локальность и селективность,
- масштабируемость по длине контекста.

---

## 1. Attention без индуктивных байасов

Self-attention без positional information:
- не различает порядок токенов,
- даёт почти равномерные карты внимания,
- не имеет понятия “дистанции” между токенами.

📓 notebook:
- `notebooks/positional_encoding_and_inductive_bias.ipynb`

Вывод: **архитектура нуждается во внешних предположениях о структуре данных.**

---

## 2. Positional Encoding как inductive bias

Мы сравнили:
- отсутствие позиции,
- absolute positional encoding,
- relative positional bias `f(|i - j|)`.

Relative bias интерпретируется как **дискретный потенциал взаимодействия**:
близкие токены взаимодействуют сильнее, дальние — слабее.

📓 notebook:
- `notebooks/positional_encoding_and_inductive_bias.ipynb`

Ключевая идея:  
> *attention ≈ взаимодействие, управляемое расстоянием.*

---

## 3. Linear / Kernel Attention: эффективность vs точность

Vanilla attention имеет сложность `O(n²)`, что ограничивает длину контекста.

Мы реализовали toy-версию kernel / linear attention:
- без явной матрицы `n × n`,
- с приближением softmax через feature map.

📓 notebook:
- `notebooks/linear_kernel_attention_toy.ipynb`

Мы сравнили:
- карты внимания,
- MSE ошибки аппроксимации,
- масштабирование по длине последовательности.

Вывод:
- kernel attention выигрывает по скорости,
- цена — аппроксимационная ошибка,
- реальная применимость зависит от feature map.

---

## 4. Gated Attention: управляемость и селективность

Gating отделяет два вопроса:
- *куда смотреть* (attention),
- *сколько информации пропускать* (gate).

Toy-модель показывает:
- подавление шума,
- контроль амплитуды выходов,
- потенциальную стабилизацию обучения.

📓 notebook:
- `notebooks/gated_attention_toy.ipynb`

Вывод:
> gating добавляет управляемость без изменения геометрии внимания.

---

## 5. Tokenization & Preprocessing как источник сложности

Длина последовательности `n` напрямую влияет на стоимость attention `O(n²)`.

Мы сравнили:
- word-level,
- char-level,
- toy-BPE,
- stemming / lemmatization.

📓 notebook:
- `notebooks/tokenization_and_preprocessing_toy.ipynb`

Наблюдения:
- char-level резко увеличивает `n²`,
- stemming / lemmatization уменьшают словарь,
- subword-токенизация — компромисс между длиной и выразимостью.

---

## Общий вывод

Эффективность и поведение LLM определяются не только масштабом,
но и **набором индуктивных байасов**, заложенных в:

- attention-механизм,
- positional information,
- tokenization,
- preprocessing.

Даже простые toy-эксперименты позволяют:
- изолировать влияние отдельных компонентов,
- понимать архитектурные компромиссы,
- осмысленно проектировать новые модели.

