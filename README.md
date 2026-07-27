# LetsHyre Chatbot API

A self-contained, role-aware chatbot API for the LetsHyre platform. Handles:

- **Visitors** — public FAQ answers via RAG over `data/faq_content.json`. No tools, no account data.
- **Candidates** — live status on their own application/interview flow (`get_candidate_status`), scoped to their session only.
- **Recruiters** — candidate search (returns a filtered dashboard redirect link, never PII in chat), plus employer profile and job listing lookups, scoped to their own company.

Responses stream token-by-token over Server-Sent Events. Chat history is stored per-session (Redis, 24h TTL by default) and identity is always resolved server-side from your auth — never from anything the client or the model claims.

Everything (config, models, rate limiter, session store, RAG, tools, LLM client, auth) lives in a single file: **`letshyre_chatbot_api.py`**. That's intentional — it's meant to drop into an existing backend with minimal ceremony.

## 1. What's in this package

```
letshyre-chatbot-api/
├── letshyre_chatbot_api.py   # the whole service
├── requirements.txt
├── .env.example
├── data/
│   └── faq_content.json      # visitor FAQ knowledge base (RAG source)
└── README.md
```

## 2. Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then fill in real values
```

Requires Python 3.11+.

## 3. Configuration

All config is via environment variables (see `.env.example` for the full list and defaults). The ones you must set for a real deployment:

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | LLM calls |
| `JWT_SECRET`, `JWT_ALGORITHM` | Must match the JWT your main LetsHyre API issues |
| `REDIS_URL` | Session store. If Redis is unreachable, the service logs a warning and falls back automatically to an in-process store — fine for local dev, **not safe for multi-worker/production** (state won't be shared across processes) |
| `CORE_API_BASE_URL`, `CORE_API_INTERNAL_TOKEN` | Where `get_candidate_status` / `get_employer_profile` / `get_employer_jobs` fetch real data from — point these at your Core API |
| `ALLOWED_ORIGINS` | CORS allow-list |
| `ENABLE_TEST_HEADERS` | **Must stay `false` outside local dev/QA** — see Security section below |

## 4. Integration: two ways to run it

### Option A — mount into your existing FastAPI app (recommended)

```python
from letshyre_chatbot_api import router as chatbot_router

app.include_router(chatbot_router)
```

That adds `POST /api/v1/chat`, `GET /health`, and `GET /` (a built-in test playground UI — safe to leave mounted, or drop that one route if you don't want it exposed) to your existing app. Make sure `data/faq_content.json` is deployed alongside the module (or set `RAG_CONTENT_PATH` to wherever you place it).

### Option B — run it standalone

```bash
python letshyre_chatbot_api.py
# or
uvicorn letshyre_chatbot_api:app --host 0.0.0.0 --port 8005
```

Useful for running the chatbot as its own microservice behind a reverse proxy/API gateway, or for the backend team to test it in isolation before wiring it in.

## 5. API

### `POST /api/v1/chat`

Request body:
```json
{ "message": "How do I sign up as a candidate?", "session_id": null }
```
Omit or pass `null` for `session_id` on the first message; the server creates one and returns it in the stream. Reuse it on subsequent turns to continue the conversation. `message` is limited to 1–4000 chars.

Response is `text/event-stream` (SSE) with three event types:
- `event: session` → `{"session_id": "..."}` — sent once, immediately
- `event: token` → `{"text": "..."}` — one per streamed chunk of the reply
- `event: done` → `{}` — end of turn (an `event: error` may appear instead if the LLM call fails)

Rate-limited per identity/IP (`RATE_LIMIT_MAX_REQUESTS` per `RATE_LIMIT_WINDOW_SECONDS`, sliding window); returns `429` when exceeded.

### `GET /health`
Returns `{"status": "ok"}`.

### `GET /`
Built-in browser playground for manually testing all three roles against the running server. Not meant for end users of the real product — feel free to exclude this route in production if you'd rather it not be reachable.

## 6. Identity & auth

Identity (role, user_id, company_id, candidate_session_id) is resolved **only** inside `get_identity()`, from one of:
1. `Authorization: Bearer <JWT>` — decoded with `JWT_SECRET`/`JWT_ALGORITHM`; `role` claim must be `recruiter` (requires `company_id` claim) or `candidate`
2. `letshyre_session` cookie — candidate mid-flow (upload → confirm-role → interview → scorecard), no full account yet
3. Neither present → treated as an anonymous `visitor`

Every tool handler scopes its data access from this server-resolved identity — a recruiter's `company_id` and a candidate's `candidate_session_id` are never taken from the request body or from anything the model outputs. This is what prevents cross-tenant data leakage.

**Wire `get_identity()` to your real auth** — as shipped it expects a JWT with `role`/`company_id`/`sub` claims and/or a `letshyre_session` cookie matching your existing candidate flow. Adjust the claim names/cookie name if yours differ.

## 7. Security notes for the backend team

- **`ENABLE_TEST_HEADERS`** gates a dev-only identity override (`x-test-role`, `x-test-company-id`, `x-test-candidate-session-id` headers) used for local testing without real tokens. It defaults to `false`, and while `false` those headers are ignored entirely. **Do not set it to `true` in any environment reachable by real users** — it lets a caller impersonate any role by just sending a header.
- No candidate PII is ever returned to a recruiter in chat — `find_candidates` only ever returns a redirect link to a filtered dashboard view.
- Visitors get zero tools — FAQ answers only, sourced from `data/faq_content.json`.
- Rotate `OPENAI_API_KEY`, `JWT_SECRET`, and `CORE_API_INTERNAL_TOKEN` before deploying — don't reuse whatever was in this package's `.env.example` (which ships empty/placeholder on purpose).

## 8. Dependencies

Everything the module imports is pinned in `requirements.txt`:

```
fastapi, uvicorn[standard], pydantic, openai, redis, httpx, pyjwt, scikit-learn, python-dotenv
```

`redis` here is used via its built-in `redis.asyncio` client — no separate `aioredis` package needed. `scikit-learn` powers the TF-IDF/RAG matching over the FAQ file.

## 9. What the Core API needs to expose

For candidate/recruiter tool calls to return real data (instead of errors), the Core API at `CORE_API_BASE_URL` should expose, authenticated via `CORE_API_INTERNAL_TOKEN` as a bearer token:

- `GET /api/v1/candidate-sessions/{candidate_session_id}/status`
- `GET /api/v1/employers/{company_id}/profile`
- `GET /api/v1/employers/{company_id}/jobs`

Until those exist, the chatbot degrades gracefully — tool calls return a friendly error message instead of a 500.
