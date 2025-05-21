# Design Doc — **Lesson-Audio Kit v0.9**

---

## 1. Цель

Сократить время подготовки и повысить качество ESL-материалов, автоматизировав полный цикл:

```
PROMPT (тема / диалог) → Текст      →   Аудио  →   Готовый ресурс для урока
                         (LLM)          (TTS)       (ZIP / mp3 / share-link)
```

* **MVP-фокус:** генерация диалога / монолога + озвучка с кэшем и выбором TTS.
* **Расширение:** конструктор целого урока (Lesson Builder) и Umbrella-Hub для внешних текстов.

---

## 2. Scope v1.0 (MVP)

| Входит                                                      | Не входит                      |
| ----------------------------------------------------------- | ------------------------------ |
| ✔ Генерация текста (OpenAI / Fal-LLM)                       | ✘ Авто-грамматика / упражнения |
| ✔ TTS-озвучка (OpenAI mini, ElevenLabs Creator, Fal PlayAI) | ✘ Распознавание речи           |
| ✔ Кэш реплик (SHA-256 → mp3)                                | ✘ Хранение прогресса ученика   |
| ✔ Подсчёт стоимости и лимитов                               | ✘ Платёжная интеграция         |

---

## 3. Архитектура (вариант С — монорепо)

```text
dialogue_tts_core/     # библиотека: парсер, LLM-клиенты, TTS-клиенты, кэш
│
├─ gradio_frontend/    # существующий HF-Space (UI + /api/tts)
└─ streamlit_frontend/ # новый быстрый UI для учителя
```

### 3.1 Компоненты core

| Модуль             | Функция                                    | Ключевые зависимости              |
| ------------------ | ------------------------------------------ | --------------------------------- |
| **parser.py**      | split\_script → \[chunks]                  | regex / spaCy (optional)          |
| **llm\_client.py** | `generate_text(prompt, provider)`          | openai, requests (Fal)            |
| **tts\_client.py** | `synthesize(text, engine, voice, emotion)` | openai, requests (ElevenLabs/Fal) |
| **cache.py**       | `get_or_create(chunk_hash)`                | sqlite, disk (mp3)                |
| **usage.py**       | лог расходов, лимиты                       | sqlite                            |

### 3.2 Деплой

| Сервис                                   | Хостинг         | Содержимое                    |
| ---------------------------------------- | --------------- | ----------------------------- |
| **dialogue-tts-core**                    | PyPI (optional) | для CLI/скриптов              |
| **HF Space** (`abocha/esl-dialogue-tts`) | 2 CPU / 16 GB   | Gradio UI + `/api/tts`        |
| **Streamlit Community**                  | 1 CPU           | Учительский UI (fast wake-up) |

---

## 4. Потоки данных

```mermaid
graph TD
  A[Prompt / Script] -->|generate_text()| B[LLM (OpenAI / Fal)]
  B --> C[Script w/ speaker tags]
  C -->|chunk| D{Cache?}
  D --hit--> E1[mp3 from disk]
  D --miss--> F[TTS engine]
  F -->|audio bytes| G[Save to cache + return]
  E1 & G --> H[Assembler (pydub)]
  H --> I[Download / Player UI]
```

---

## 5. Техстек

| Слой   | Библиотеки                                |
| ------ | ----------------------------------------- |
| Core   | Python 3.11, pydub, uvicorn (API), sqlite |
| LLM    | openai-python ≥ 1.14, requests → Fal      |
| TTS    | openai.audio, ElevenLabs REST, Fal REST   |
| UI     | Gradio 4.x, Streamlit 1.33                |
| DevOps | GitHub Actions, pytest+vcr, ruff/black    |

---

## 6. Квоты & маршрутизация

| Правило                                          | Engine          |
| ------------------------------------------------ | --------------- |
| default ≤ 2 k симв.                              | OpenAI mini-TTS |
| premium + симв. ≤ 2 k + credits > 20 k           | ElevenLabs      |
| multi-voice / спец.эмоция                        | Fal PlayAI      |
| LLM: use GPT-4o (grant) → fallback any-LLM (Fal) |                 |

---

## 7. Роадмап (6 недель)

| Неделя | Milestone                                                  |
| ------ | ---------------------------------------------------------- |
| 1      | Fork → `dialogue_tts_core`; unit-тесты; перенёс кэш        |
| 2      | Gradio UI переключён на core; добавлен `/api/tts`          |
| 3      | MVP Streamlit UI (prompt → audio); подсчёт цены            |
| 4      | Fal any-LLM, ElevenLabs интеграция; кнопка “Premium voice” |
| 5      | Кэш реплик, монитор расходов, лимиты UI                    |
| 6      | Документация, Loom-демо, тест с реальными уроками          |

---

## 8. Риски & минимизация

| Риск                   | Митигейшн                             |
| ---------------------- | ------------------------------------- |
| Траты > бюджета        | кэш реплик + пороговые алерты         |
| Cold-start HF          | Streamlit-UI вынесен отдельно         |
| API - breaking changes | обёртка-adapter + pytest CI           |
| Сложность монорепо     | чёткая папочная структура, pre-commit |

---

## 9. Success Criteria

* 3-минутный listening-snippet генерируется ≤ 30 сек.
* > 95 % кэш-хит при повторном редактировании.
* UX-опрос: «время подготовки снизилось ×5» (с 15 мин → 3 мин).

---

**Готов двигаться к детализации API core-модуля или расписанию задач первого спринта.**

**Уровень уверенности: 92 % 🙂**
