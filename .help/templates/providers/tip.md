---
type: tip
name: providers-tip
feature: providers
depth: tip
generated_at: 2026-05-20T03:27:22.851988+00:00
source_hash: fbe19e4accf7e90a0aec29d23dcdabe3822e7458ce7ff5e4412a81d42aae02f9
status: generated
---

# Tip: working effectively with providers

## Recommendation

Call `list_available()` before calling `get_provider()` to avoid a `ValueError` at runtime.

`list_available()` inspects which provider SDKs are currently importable and returns only their names. If you call `get_provider("claude")` without first installing `attune-rag[claude]`, the call raises:

```
ValueError: Unknown provider {claude}. Known providers: {...}.
```

Checking `list_available()` first makes that failure visible before it can block a request.

**Why it sticks:** the providers module lazy-imports each SDK, so the core package installs cleanly regardless of which extras you include — but that also means a provider can silently be absent until you ask for it.

## Tradeoff

`list_available()` reflects the environment at import time. If you install an extra after the process starts, the result won't update until you restart the interpreter. Don't cache the output across long-running processes where the environment might change.
