# Floor Plan Analysis APIs — Полное исследование (2026-02-16)

## 1. КОММЕРЧЕСКИЕ API (Специализированные)

### 1.1 RasterScan
- **URL**: https://www.rasterscan.com/
- **RapidAPI**: https://rapidapi.com/akashdev2016/api/floor-plan-digitalization
- **HuggingFace Demo**: https://huggingface.co/spaces/RasterScan/Automated-Floor-Plan-Digitalization
- **GitHub (On-Premise)**: https://github.com/RasterScan/Floor-Plan-Recognition
- **Что делает**: Конвертирует сканированные/нарисованные floor plan изображения в структурированные векторные данные. Извлекает стены, двери, окна и символы. Рендерит интерактивные 3D модели.
- **Вход**: Base64 images или multipart form-data (JPG, PNG, PDF, TIFF)
- **Выход**: DXF, IFC, GLTF, SVG, JSON (списки дверей, стен, комнат)
- **Цена**: 6 тарифов от $49/мес (Starter) до $950/мес (Ultimate). Free tier: 10,000 API вызовов/день. 30-day money-back guarantee.
- **API**: REST API с API key + secret, rate limiting, key rotation. Cloud и on-premise.
- **Точность**: AI-based, "high precision" — зависит от качества входного изображения.

### 1.2 CubiCasa
- **URL**: https://www.cubi.casa/developers/
- **API Docs**: https://integrate.docs.cubi.casa/
- **Что делает**: Mobile SDK сканирует комнату; Conversion API конвертирует сканы в floor plan модели. 80+ категорий объектов.
- **Вход**: Мобильный скан через CubiCasa SDK или загрузка изображения
- **Выход**: SVG (векторные полигоны с аннотациями), JPG, PNG, PDF, OBJ, IFC (BIM)
- **Цена**: Первый скан бесплатно. $9.99-$30 за план. Скидки: 25% от 20+/мес, 30% от 200+, 35% от 1000+.
- **API**: REST API (v3). Base URL: `https://app.cubi.casa/api/integrate/v3`
- **Точность**: Deep learning semantic segmentation. CubiCasa5k датасет — benchmark в индустрии.
- **Минус**: Заточен под мобильное сканирование, не под "отправь любое изображение".

### 1.3 Archilogic
- **URL**: https://www.archilogic.com/
- **Developer Docs**: https://developers.archilogic.com/
- **Что делает**: Spatial data платформа. Принимает PDF/JPG/DWG → структурированные пространственные данные. White-glove (human QA).
- **Вход**: JPG, PNG, PDF, DWG, DXF
- **Выход**: JPG, PNG, SVG, DXF, GeoJSON, IFC, GLTF, IMDF
- **Цена**: Starter (Free): 5 планов, $329/план конвертация. Plus: $24.90/план/мес. Professional: $59/план/мес. Enterprise: custom.
- **API**: GraphQL (Space API), REST (V2), Floor Plan Engine JS/TS SDK
- **Точность**: Human review = высшее качество. Fortune 500 клиенты.

### 1.4 Kreo Software
- **URL**: https://www.kreo.net/
- **API**: https://www.kreo.net/features/api
- **Что делает**: AI takeoff — стены (с типами и длинами), двери, окна, сантехника, розетки. IFC 3D из 2D. OCR для текста.
- **Вход**: 2D floor plan images/PDFs
- **Выход**: Элементы с координатами и типами. IFC для 3D.
- **Цена**: Flat fee + per-request. Контакт: info@kreo.net
- **API**: REST API

### 1.5 MagicPlan
- **URL**: https://magicplan.app/
- **API Docs**: https://apidocs.magicplan.app/
- **Что делает**: REST API для управления проектами floor plan, загрузки PDF, получения данных.
- **Вход**: PDF, images, мобильные сканы
- **Выход**: JSON через REST endpoints
- **Цена**: от $149/мес (PRO Estimator) до $600/мес (PRO Flex)
- **Минус**: Больше workflow management, чем image→vector конвертация.

### 1.6 GetFloorPlan
- **URL**: https://getfloorplan.com/
- **Что делает**: AI создаёт 2D/3D планы, виртуальные туры из загруженных изображений. До 1000 планов за 24 часа.
- **Выход**: 2D, 3D планы, виртуальные туры, рендеры
- **Цена**: Basic $20, Plus $35, Pro $45, Render $45, Max $60 (за план)
- **Минус**: Фокус на визуализацию, не на извлечение координат.

### 1.7 Floorplanner
- **URL**: https://floorplanner.com/
- **API Docs**: https://floorplanner.readme.io/reference/getting-started
- **Что делает**: 2D/3D creation и rendering с API для встраивания редактора.
- **Цена**: от $5/мес. Enterprise для API.
- **Минус**: Инструмент создания, не анализа изображений.

### 1.8 MeasureSquare
- **URL**: https://measuresquare.com/floor-plan-sdk-api/
- **Что делает**: M2Diagram SDK для интерактивных floor plan моделей. AI takeoff.
- **Цена**: от $149/мес
- **Минус**: Фокус на flooring/construction estimation.

### 1.9 Matterport
- **URL**: https://matterport.github.io/showcase-sdk/modelapi_property_intelligence.html
- **Что делает**: Извлекает комнаты, стены, размеры из 3D сканов. CubiCasa интеграция.
- **Минус**: Требует Matterport 3D камеру. Не image-analysis API.

---

## 2. LLM / VISION API (Общего назначения)

### 2.1 Google Gemini API (2.5 Pro / 3 Flash)
- **URL**: https://ai.google.dev/gemini-api/docs/vision
- **Что делает**: Мультимодальная модель. С Gemini 2.5+ поддерживает **object detection с bounding boxes** [y0,x0,y1,x1] normalized 0-1000, и **segmentation с contour masks**.
- **Цена**: Gemini 2.5 Pro: $1.25/M input, $10.00/M output. ~$0.0007/image input. **Free tier через AI Studio**.
- **API**: REST, Python/Node.js SDK
- **Наш опыт**: Используем сейчас. Считает элементы правильно, но координаты неточные (±5-10% изображения).

### 2.2 OpenAI GPT-4o / GPT-4.1 Vision
- **URL**: https://platform.openai.com/docs/guides/vision
- **Что делает**: Vision с structured JSON output. НЕ имеет native bounding box как Gemini.
- **Цена**: GPT-4o: $2.50/M input, $10.00/M output.
- **Минус**: Координаты только через prompting, не native.

### 2.3 Anthropic Claude (Opus 4 / Sonnet 4)
- **URL**: https://docs.anthropic.com/en/docs/build-with-claude/vision
- **Цена**: Sonnet 4: $3.00/M input, $15.00/M output.
- **Минус**: Документация Anthropic сама отмечает "limited spatial reasoning".

### 2.4 Microsoft Florence-2
- **URL**: https://huggingface.co/microsoft/Florence-2-large
- **Что делает**: Lightweight vision-language model: captioning, object detection, grounding, segmentation.
- **Цена**: Free (MIT license). Self-hosted.
- **Плюс**: Можно дообучить на floor plan данных.

---

## 3. OPEN SOURCE (GitHub)

### 3.1 Grounding DINO + SAM (Zero-Shot Pipeline)
- **Grounding DINO**: https://github.com/IDEA-Research/GroundingDINO
- **SAM**: Meta's Segment Anything Model
- **Что делает**: Промпт "wall. door. window." → bounding boxes → SAM даёт точные маски сегментации → полигоны.
- **Цена**: Free, Apache 2.0
- **Точность**: 52.5 AP на COCO zero-shot. ECCV 2024.
- **Важно**: Не требует обучения, работает out-of-the-box.

### 3.2 Raster-to-Graph (SOTA 2024, Eurographics)
- **GitHub**: https://github.com/SizheHu/Raster-to-Graph
- **Что делает**: Attention transformer авторегрессивно предсказывает wall junctions (узлы) и wall segments (рёбра).
- **Выход**: Граф с координатами стыков стен + категории комнат
- **Цена**: Free, open source
- **Датасет**: 10,000+ реальных floor plans

### 3.3 FloorplanTransformation (ICCV 2017)
- **GitHub**: https://github.com/art-programmer/FloorplanTransformation
- **Что делает**: Растр → вектор. Детектирует junctions, integer programming агрегирует в wall lines, door lines, icon boxes.
- **Выход**: Текстовый файл: стены `x1,y1,x2,y2,room_type_left,room_type_right`, двери, иконки.
- **Цена**: Free
- **Минус**: Lua/Torch (старый стек)

### 3.4 TF2DeepFloorplan
- **GitHub**: https://github.com/zcemycl/TF2DeepFloorplan
- **Что делает**: TensorFlow 2 реимплементация DeepFloorplan. Семантическая сегментация комнат и границ.
- **Плюс**: Flask server + Docker + TFLite + Google Colab

### 3.5 DeepFloorplan (оригинал)
- **GitHub**: https://github.com/zlzeng/DeepFloorplan
- **Что делает**: Multi-task network с room-boundary-guided attention.
- **Точность**: IEEE published. Базовая работа, но новые модели лучше (GMFS в 22.7x точнее).

### 3.6 Floor Plan Object Detection (YOLOv8)
- **GitHub**: https://github.com/sanatladkat/floor-plan-object-detection
- **Что делает**: YOLOv8 для columns, walls, doors, windows.
- **Выход**: Bounding boxes с классами и confidence

### 3.7 Floor Plan Segmentation API (Flask)
- **GitHub**: https://github.com/ozturkoktay/floor-plan-segmentation-api
- **Что делает**: Flask REST API для PDF анализа, AI сегментации, object detection.
- **Плюс**: Ready-to-deploy REST API

### 3.8 CubiCasa5k (Датасет + Модель)
- **GitHub**: https://github.com/CubiCasa/CubiCasa5k
- **Что делает**: 5,000 floor plan images с SVG аннотациями, 80+ категорий. Включает trained модель.
- **Плюс**: Industry-standard benchmark

### 3.9 MLSTRUCT-FP (Датасет)
- **GitHub**: https://github.com/MLSTRUCT/MLSTRUCT-FP
- **PyPI**: https://pypi.org/project/MLStructFP/
- **Что делает**: 954 floor plan images, 70,873 wall rectangles в JSON. Python library.

---

## 4. ROBOFLOW UNIVERSE (Готовые модели + Hosted API)

| Модель | Датасет | Классы | Ссылка |
|--------|---------|--------|--------|
| Floor Plan AI Object Detection | 6,031 imgs | Multiple | https://universe.roboflow.com/floor-plan-rendering/floor-plan-ai-object-detection |
| Floor Plan Walls | 3,395 imgs | Walls | https://universe.roboflow.com/testing-daidy/floor-plan-walls |
| Window Detection | 4,000 imgs | Windows | https://universe.roboflow.com/bytetrooper/window-detection-in-floor-plans |
| Floor Plans (Doors/Windows) | 100 imgs | Doors, Windows | https://universe.roboflow.com/muhammad-anas-i2dav/floor-plans-njqjm |
| Floor-Plan Segmentation | 148 imgs | Wall, Door, Window | https://universe.roboflow.com/iiitbangalore/floor-plan-segmentation-dtr4r |
| Floor Plan Instance Seg | 1,096 imgs | Elements | https://universe.roboflow.com/3dfloorplan-nytxc/floor-plan-nnoub-bk4vn-czy3i |

**Цена Roboflow**: Free (public). Starter: $49/мес (10K hosted inference). Growth: $299/мес.

---

## 5. HUGGINGFACE

| Ресурс | Тип | Описание |
|--------|-----|----------|
| segformer-b0-finetuned-floorplan | Model | SegFormer fine-tuned |
| FloorPlanVisionAIAdaptor | Model | VLM для floor plans |
| Automated Floor Plan Digitalization | Space | RasterScan демо |
| FloorPlanCAD | Dataset | 15,000+ планов, 30 категорий |
| pseudo-floor-plan-12k | Dataset | 12,000 pseudo планов |

---

## 6. RESEARCH PAPERS (2024-2025)

| Название | Год | Инновация |
|----------|-----|-----------|
| **WAFFLE** | WACV 2025 | 20K multimodal датасет, semantic segmentation benchmark |
| **FloorSAM** | 2025 | SAM-guided reconstruction from LiDAR |
| **GMFS** | 2024 | GPT-4 + SAM few-shot, 22.7x precision vs DeepFloorplan |
| **DiffPlanner** | 2025 | Vector floor plan generation via diffusion |
| **ArchCAD-400k** | 2025 | 413,062 annotated chunks, 26x > FloorPlanCAD |
| **MDA-Unet** | 2024 | 95.2% wall detection precision, 92%+ recall |
| **BFA-YOLO** | 2024 | Best YOLO для door/window detection |
| **FloorPlan-LLaMa** | ACL 2025 | LLM-based с architect feedback |

---

## 7. СВОДНАЯ ТАБЛИЦА

| Сервис | Тип | Координаты? | Free tier? | Цена/image | Лучше для |
|--------|-----|-------------|------------|------------|-----------|
| **RasterScan** | Dedicated API | ✅ DXF/IFC/JSON | ✅ 10K/day | $49-950/мес | Production digitization |
| **Archilogic** | SaaS | ✅ GeoJSON/SVG | ✅ 5 планов | $24.90-329/план | Portfolio management |
| **CubiCasa** | Mobile SDK+API | ✅ SVG polygons | ✅ 1 скан | $10-30/скан | Mobile scanning |
| **Kreo** | Construction | ✅ coords+IFC | ❌ | Volume-based | Construction takeoff |
| **Roboflow** | Hosted CV | ✅ bboxes | ✅ | $49/мес starter | Quick prototyping |
| **Grounding DINO+SAM** | Open Source | ✅ bbox+masks | ✅ | Self-hosted | Zero-shot, no training |
| **Raster-to-Graph** | Open Source | ✅ graph nodes | ✅ | Self-hosted | SOTA vectorization |
| **TF2DeepFloorplan** | Open Source | Masks | ✅ | Self-hosted | Research baseline |
| **Gemini** | LLM Vision | ⚠️ ~±5-10% | ✅ | ~$0.001 | Semantic understanding |
| **GPT-4o** | LLM Vision | ⚠️ approximate | ❌ | ~$0.003 | Semantic understanding |

---

## 8. РЕКОМЕНДАЦИИ

1. **RasterScan** — самый быстрый путь: REST API, бесплатный tier, координаты в JSON/DXF/SVG
2. **Grounding DINO + SAM** — лучший долгосрочный: бесплатно, точная сегментация, без вендора
3. **Roboflow** — быстрый прототип с hosted inference API
4. **Raster-to-Graph** — SOTA для стен, но нужен Python-сервер
