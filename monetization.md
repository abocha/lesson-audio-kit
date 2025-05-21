### 💸 Монетизация Lesson-Audio Kit — от идеи до продакшена

| Шаг                                   | Что конкретно делаем                                                                                                                                                                                                          | Почему это работает                                                                   |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| **1. Модель “credits-as-fuel”**       | 1 кредит = 1 символ TTS + 50 токенов LLM. <br>• **Free-tier:** 5 000 кредитов / месяц (≈ 5 мин аудио) <br>• **Creator:** 100 000 кредитов / мес → 9 \$/мес <br>• **Pro-Studio:** 500 000 кредитов + Voice-Cloning → 39 \$/мес | ✔ Прозрачно: учителю понятно, за что платит <br>✔ Легко держать расходы под контролем |
| **2. Счётчик usage в core**           | `usage.log_event(user_id, provider, chars, tokens, cost_est)` → SQLite / Supabase                                                                                                                                             | ✔ Одна таблица → любые дашборды <br>✔ Используем для отсека (paywall)                 |
| **3. Stripe + Billing micro-service** | • Планы и one-off top-up.<br>• Webhook `/stripe/webhook` → `users.update(credits += purchased)`<br>• При откате платежа — вычитаем                                                                                            | ✔ Stripe в VN работает через Wise/Techcom <br>✔ Webhook ≈ 50 строк FastAPI            |
| **4. Paywall в Streamlit**            | `if user.credits < 0: st.stop()` + модалка “Пополнить” → /billing                                                                                                                                                             | ✔ Streamlit session\_state хранит JWT (Supabase auth)                                 |
| **5. Referral 10 % bonus**            | Таблица `referrals` + купон `?ref=code` → +10 % кредитов обоим                                                                                                                                                                | ✔ Простой growth-loop без расходов на ads                                             |
| **6. Лимиты провайдерам**             | • OpenAI: держим отдельный ключ per-tenant (optional) <br>• Elevent: per-tenant API-key → не рискуем глобальной блокировкой                                                                                                   | ✔ Pay-as-you-go, никто не сожрёт твой лимит                                           |
| **7. Premium-фичи за paywall**        | • ElevenLabs HQ-звук и Voice-clone<br>• Batch-export целого курса<br>• Telegram-бот “скачай-в-одно-касание”                                                                                                                   | ✔ Платят те, кому реально нужна скорость / качество                                   |

---

#### ⚙️ Минимальный tech-стек

```text
Supabase Auth  ──>  JWT in Streamlit session
Supabase DB (credits, usage, referrals)
Stripe Checkout + Webhook (FastAPI on HF)
```

* **auth**: Google → сразу e-mail verified
* **credits**: INT, never negative; after each request `credits -= estimated`
* **webhook**: `POST /stripe/webhook` validates signature → `db.rpc('add_credits', user_id, amount)`

---

#### 📐 Флоу пользователя

1. **Sign-up** → 5 000 free-credits.
2. Работает, видит счётчик (“3 412 credits left”).
3. Ноль → модалка “Need more?” → Stripe Checkout в том же окне.
4. Возврат + web-socket message “Credits updated: 105 000”.
5. Пользователь продолжает без перезахода.

---

#### 🪙 Ценообразование (пример)

| Провайдер       | Себестоимость | К-во символов / 1 \$ | Наценка | В кредитах (1 симв. = 1 кр.)    |
| --------------- | ------------- | -------------------- | ------- | ------------------------------- |
| OpenAI mini-TTS | \$0.015 / 1 k | 66 666               | ×3      | 1 симв. = 1 кр. (≈ \$0.045 / к) |
| ElevenLabs      | \$0.30 / 1 k  | 3 333                | ×2      | 1 симв. = 2 кр.                 |
| Fal PlayAI      | \$0.005 / 1 k | 200 000              | ×4      | 1 симв. = 1 кр.                 |

*В UI просто пишем: “ElevenLabs HQ = 2 credits/char”*.

---

#### 🚀 Roll-out план

| Неделя | Что выкатываем                                                   |
| ------ | ---------------------------------------------------------------- |
| 0-1    | Supabase Auth + Stripe sandbox + credits decrement               |
| 2      | Production Stripe + Free-tier лимит + Paywall                    |
| 3      | Referral код + Telegram бот оплаты (Qiwi / USDT fallback для RU) |
| 4      | Premium-features toggle (ElevenLabs HQ, batch export)            |

---

#### 🛑 Риски

| Риск                    | Митигейшн                                      |
| ----------------------- | ---------------------------------------------- |
| Карты РФ не платят      | USDT/crypto invoice через NowPayments          |
| Abuse free-tier         | Captcha + rate-limit 10 requests / час         |
| Перерасход OpenAI ключа | Верхний hard-cap per-user и global-monthly-cap |

---

### TL;DR

1. **Credits-based freemium** (free 5k, Creator 100k, Pro 500k).
2. **Stripe + Supabase** — лёгкий, 100 % сценариев закрывает.
3. **Paywall** прямо в Streamlit; premium аудио и batch-фишки мотивируют платить.
4. Расходы прогнозируем, риски контроля держим через глобальные лимиты и кэш.

**Уровень уверенности: 85 % 😎**
