## 🚀 Bootstrap-гайд для **ИИ-архитектора**

*(цель — развернуть рабочий скелет проекта в VS Code, чтобы ИИ-кодер мог сразу приступать к рефактору)*

---

### 1. Локальное окружение

1. **Cloning**

   ```bash
   git clone https://huggingface.co/spaces/abocha/esl-dialogue-tts lesson-audio-kit
   cd lesson-audio-kit
   ```
2. **Python & venv**

   ```bash
   pyenv install 3.13.3                 # если нужен менеджер версий
   python -m venv .venv
   source .venv/bin/activate            # Win: .venv\Scripts\activate
   ```
3. **Poetry / pip-tools (выбери одно)**

   ```bash
   pip install --upgrade pip poetry
   poetry init  # использовать существующий requirements.txt как подсказку
   ```
4. **VS Code settings** (`.vscode/settings.json`)

   ```json
   {
     "python.defaultInterpreterPath": ".venv/bin/python",
     "python.testing.pytestEnabled": true,
     "python.formatting.provider": "black",
     "python.linting.ruffEnabled": true
   }
   ```

---

### 2. Базовая структура репо

```text
lesson-audio-kit/
│
├─ dialogue_tts_core/          # ➊ будет выделен из существующего кода
│   ├─ __init__.py
│   ├─ parser.py
│   ├─ llm_client.py
│   ├─ tts_client.py
│   ├─ cache.py
│   └─ usage.py
│
├─ gradio_frontend/            # ➋ копия нынешнего HF UI
│   └─ app.py
│
├─ streamlit_frontend/         # ➌ новый UI
│   └─ app.py
│
├─ tests/                      # pytest + vcr
│
├─ scripts/                    # helper CLI, DB migrations
├─ .github/workflows/ci.yml    # Ruff → PyTest → Build
├─ .pre-commit-config.yaml     # ruff, black, isort
└─ README.md
```

> **ИИ-архитектор** формализует API для `dialogue_tts_core` и спецификацию REST-эндпоинта `/api/tts` до того, как ИИ-кодер начнёт правку.

---

### 3. Пакеты к установке (poetry / requirements)

```toml
# pyproject.toml (фрагмент)
[tool.poetry.dependencies]
python = "^3.13.3"
openai = ">=1.14"
requests = "*"
pydub = "*"
uvicorn = "*"
fastapi = "*"
gradio = "^4.0"
streamlit = "^1.33"
# TTS providers
elevenlabs = "*"
falclient = "*"        # обёртка для Fal API (либо raw requests)
# Dev / tests
pytest = "*"
pytest-vcr = "*"
ruff = "*"
black = "*"
```

---

### 4. DevOps / CI

1. **GitHub Actions** (`ci.yml`)

   * lint: `ruff .`
   * format-check: `black --check .`
   * tests: `pytest -q`
2. **Pre-commit**

   ```yaml
   repos:
     - repo: https://github.com/astral-sh/ruff-pre-commit
       rev: v0.4.4
       hooks: [id: ruff]
     - repo: https://github.com/psf/black
       rev: 24.4.2
       hooks: [id: black]
   ```

---

### 5. Secrets & конфиги (для удобства ИИ-кодера)

```
.env.example
OPENAI_API_KEY=
ELEVEN_KEY=
FAL_KEY=
DB_PATH=usage.db
```

*Для монетизации:*
добавь placeholders `STRIPE_KEY=`, `SUPABASE_URL=`, `SUPABASE_ANON_KEY=`.

---

### 6. План разборки существующего кода

| Очередь | Файл-источник       | Что выдёргиваем в `dialogue_tts_core`  |
| ------- | ------------------- | -------------------------------------- |
| 1       | `event_handlers.py` | `synthesize_audio()`, `split_script()` |
| 2       | `utils/speaker.py`  | весь парсинг ролей/эмоций              |
| 3       | `app.py`            | TTS-клиенты (OpenAI mini)              |

*(ИИ-архитектор оформит таблицу “source → destination → refactor notes”, а ИИ-кодер выполняет.)*

---

### 7. Шаблоны интерфейсов

```python
# dialogue_tts_core/tts_client.py
def synthesize(text: str,
               engine: Literal["openai", "eleven", "fal"],
               voice: str,
               emotion: str | None = None) -> bytes: ...

# dialogue_tts_core/llm_client.py
def generate(prompt: str,
             provider: Literal["openai", "claude", "gemini", "llama"],
             **kwargs) -> str: ...
```

ИИ-архитектор в дальнейшем зафиксирует сигнатуры и опишет, какие kwargs нужны.

---

### 8. База расходов (SQLite)

```sql
CREATE TABLE usage (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  user TEXT,
  provider TEXT,
  chars INTEGER,
  tokens INTEGER,
  cost REAL
);
```

---

### 9. “Монетизация-Ready”

* Подготовить модуль `billing.py` с заглушками:

  ```python
  def check_credits(user: str, cost: int) -> bool: ...
  def add_credits(user: str, amount: int): ...
  ```
* ИИ-архитектор опишет интеграцию Stripe → Supabase в отдельном дизайн-спеке, ИИ-кодер потом подменит заглушки.

---

## 📌 Что должен сделать ИИ-архитектор дальше

1. **Финализировать API blue-prints** (core-функции, REST-эндпоинт).
2. **Составить список тасков для ИИ-кодера** по модулю/файлу с приоритетами и тест-кейсами.
3. Описать, как мигрировать существующий `cache/` и настроить shared storage (если нужно).
4. Специфицировать правила кэш-хеша (SHA-256 текста без пробелов, lower-case?).

---

После утверждения архитектуры ИИ-кодер берёт этот репо, выполняет таски и пушит MR / PR.

**Уровень уверенности: 86 % 😊**
