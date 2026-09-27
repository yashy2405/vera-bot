# Vera AI Challenge — Submission

**Team:** Yash
**Contact:** yashyadav120905@gmail.com
**Live endpoint:** https://vera-bot-v11y.onrender.com

## Approach

This bot is a **deterministic, rule-based composer** — there is no LLM in the decision-making
or message-generation path. Every message is built by dispatching on `trigger.kind` to a
dedicated Python function that slot-fills a template using *only* fields present in the
`CategoryContext`, `MerchantContext`, `TriggerContext`, and (when present) `CustomerContext`
payloads.

### Why no LLM in the core path

The rubric weighs groundedness and zero-hallucination heavily, and the brief explicitly
recommends `temperature=0` with a rule-based fallback for exactly this reason. Rather than
build an LLM-generation-plus-validation-layer, I went straight to the validation layer as
the primary path: a template composer *cannot* invent a price, a discount, or a competitor
name that isn't already in the context, because it never has the freedom to write freeform
text in the first place. This trades away some of the surface-level naturalness an LLM
rephrasing pass could add, in exchange for:

- **Zero hallucination risk**, by construction, not by hoping a validator catches it after the fact
- **Sub-10ms response times**, well inside the 30s timeout even under load
- **Full reliability** with no dependency on a third-party LLM API's uptime or rate limits — a
  real concern I ran into personally while testing with a free-tier key

An optional `/v1/compose` endpoint is also exposed as a direct alias to the same composer,
in case any tooling expects that endpoint name — it does not replace or interfere with the
`/v1/context` + `/v1/tick` + `/v1/reply` flow that the judge harness actually calls.

## Architecture

Five required endpoints, all implemented in a single FastAPI app (`bot.py`), with in-memory
state (no external DB — appropriate for a hackathon-scale evaluation window):

- `POST /v1/context` — versioned, idempotent context ingestion. Re-posting the same version
  is a no-op; a higher version replaces atomically. Supports `category`, `merchant`,
  `customer`, and `trigger` scopes.
- `POST /v1/tick` — for each available trigger, resolves category + merchant + (optional)
  customer context, checks suppression state, and calls the composer. Returns up to 20
  actions per call. Never lets one bad trigger break the whole batch (wrapped in try/except).
- `POST /v1/reply` — a small deterministic state machine handling:
  - **Auto-reply detection**, tracked *per merchant identity* (not per conversation_id),
    since a real merchant's auto-reply account may be contacted across multiple threads.
    Escalates send → wait (24h) → end across repeated canned replies.
  - **Hard opt-out** phrases → polite exit + 30-day suppression flag on that merchant.
  - **Intent transition** phrases (e.g. "kar do", "let's do it") → skips straight to an
    execution/confirmation message rather than re-qualifying.
  - **Hard 5-turn cap**, enforced regardless of content.
- `GET /v1/healthz`, `GET /v1/metadata` — as specified.

A `GET /v1/_debug/reset` endpoint is also included, purely to make repeated manual testing
against a long-running deployment easier (clears all in-memory state). It is not part of the
graded contract and is never called by `judge_simulator.py`.

## Key design decisions

- **Grounding over generation.** Every composer function only reads fields that exist in the
  payload it's given. Missing fields degrade gracefully to a safer, more generic phrasing —
  they never get invented.
- **Category-aware salutation.** The bot reads `category.voice.salutation_examples` (e.g.
  `"Dr. {first_name}"` for dentists) rather than hardcoding a tone per vertical, so the same
  code path stays correct if new categories are added.
- **Customer-scope safety.** If a trigger is customer-scoped but no `CustomerContext` has
  been pushed for that customer, the bot skips sending entirely rather than guessing a name
  or preference — a message is worse than no message if it isn't grounded.
- **A generic, safe fallback for unrecognized `trigger.kind` values.** Since the real
  evaluation injects fresh triggers mid-test, the fallback composer is built to degrade
  gracefully on anything unseen rather than throwing an error or fabricating content.

## Known limitations

- Follow-up replies in `/v1/reply` that aren't an opt-out, auto-reply, or clear "go ahead"
  signal now name the actual topic of the originating trigger when it's known (e.g. "should
  I go ahead with the kids yoga summer camp then?"), by looking up the trigger that started
  the conversation. This is still a fixed template, not a full understanding of what the
  merchant said — it doesn't yet parse the content of an ambiguous reply itself, only knows
  what topic the conversation was originally about.
- Off-topic messages (e.g. "who is this?", "wrong number") are now recognized against a
  curated phrase list and get a short, friendly, in-character redirect rather than being
  treated as real engagement. This is a heuristic phrase match, not a general classifier, so
  it will not catch every possible off-topic message — only the more common patterns.

## Testing

Verified locally against the real seed dataset and the provided `judge_simulator.py`
(with a free Gemini key) across multiple iterations, in addition to manual endpoint testing
covering context versioning, auto-reply escalation, opt-out suppression, and the 5-turn cap.
