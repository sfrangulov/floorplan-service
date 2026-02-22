# Floorplan SegFormer

Семантическая сегментация планировок квартир на основе SegFormer-B2.

**Вход:** изображение планировки → **Выход:** JSON с полигонами комнат и их типами.

## Как работает

### Модель

**SegFormer-B2** (`nvidia/segformer-b2-finetuned-ade-512-512`) — трансформер для семантической сегментации. Претрейн на ADE20K (150 классов), голова переинициализируется на 10 классов.

### Классы

| ID | Класс | Описание |
|----|-------|----------|
| 0 | background | Всё остальное |
| 1 | wall | Стены |
| 2 | door | Двери |
| 3 | window | Окна |
| 4 | balcony | Балкон |
| 5 | balcony_window | Стеклянная перегородка |
| 6 | bedroom | Спальня |
| 7 | living_room | Гостиная |
| 8 | kitchen | Кухня |
| 9 | bathroom | Санузел |

### Данные

- **Синтетика** (2000 train / 500 val) — `generate_synthetic.py` генерирует планировки программно через `floorplan_generator.py`
- **CubiCasa5K** (~4500 train / ~500 val) — реальные планировки с [Zenodo](https://zenodo.org/record/2613548), SVG-аннотации конвертируются в маски через `prepare_cubicasa.py`

Все сэмплы приводятся к 512x512 с сохранением пропорций (белый padding).

### Обучение (`train.py`)

- Оптимизатор: AdamW + PolynomialLR scheduler
- Loss: Combined (CrossEntropy + Tversky)
- Аугментации: flip, rotate, color jitter, noise, elastic
- Early stopping по mIoU на валидации
- Сохраняет `best/` и `latest/` чекпоинты

### Инференс (`predict.py`)

1. Загрузка картинки → ресайз 512x512 с padding
2. Прогон через модель → логиты 128x128 (1/4 разрешения)
3. Upsample до 512x512 → argmax → маска классов
4. Обратный маппинг на оригинальный размер (убираем padding)
5. Векторизация маски в полигоны (`vectorize.py`) → JSON с координатами комнат

## Quick Start

### 1. Setup

```bash
cd training
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Генерация синтетических данных

```bash
python generate_synthetic.py --count 2000 --output-dir data/prepared/synthetic/train
python generate_synthetic.py --count 500 --output-dir data/prepared/synthetic/val --seed 999
```

### 3. Подготовка CubiCasa5K

```bash
wget -O cubicasa5k.zip https://zenodo.org/record/2613548/files/cubicasa5k.zip
unzip -q cubicasa5k.zip -d data/
python prepare_cubicasa.py --data-dir data/cubicasa5k --output-dir data/prepared/cubicasa
```

### 4. Обучение

```bash
python train.py \
  --data-dirs data/prepared/cubicasa/train data/prepared/synthetic/train \
  --val-dirs data/prepared/cubicasa/val data/prepared/synthetic/val \
  --output-dir checkpoints/segformer-floorplan-v3 \
  --batch-size 8 \
  --epochs 50
```

### 5. Тестирование

```bash
python predict.py \
  --model checkpoints/segformer-floorplan-v3/best \
  --image ../test-data/U3laAgYJ.jpg \
  --output result.json \
  --save-mask
```

### 6. Обучение в Google Colab

Загрузить `training-code-v3.zip` в Colab и запустить `train_colab.ipynb`.

Версия задаётся один раз в cell 1 через `VERSION = "v3"`.

## Файлы

| Файл | Назначение |
|------|-----------|
| `config.py` | Классы, гиперпараметры, конфигурация модели |
| `prepare_cubicasa.py` | Подготовка датасета CubiCasa5K (SVG → маски) |
| `floorplan_generator.py` | Процедурная генерация планировок |
| `generate_synthetic.py` | Рендер синтетических изображений + масок |
| `dataset.py` | PyTorch Dataset с аугментациями |
| `loss.py` | Tversky + Combined loss |
| `train.py` | Обучение с early stopping |
| `vectorize.py` | Маска → полигоны |
| `predict.py` | Полный пайплайн инференса |
| `serve.py` | HTTP-сервер для инференса |
| `cog_predict.py` | Replicate predictor |
| `train_colab.ipynb` | Ноутбук для обучения в Colab |
| `test_vectorize.py` | Тесты векторизации |
