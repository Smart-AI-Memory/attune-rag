# Distinguish local errors from provider outages

R5 permits one bounded retry for a typed transient provider failure, but requires
configuration and local failures to remain blocking. Before this change,
`_transient_primary_provider_error` accepted every Anthropic `APIConnectionError`.
The SDK wraps arbitrary request exceptions in that class.

An October 19 offline probe using CI’s Anthropic 0.125.0 reproduced malformed
dummy header values as `APIConnectionError -> httpx.LocalProtocolError` and an
unsupported URL scheme as `APIConnectionError -> httpx.UnsupportedProtocol`.
Both produced benchmark exit 3 and unavailable faithfulness. The malformed
headers wrote zero HTTP bytes. A valid dummy control completed against a memory
backend. This proves a classifier defect; it does not identify the cause of the
actual CI connection failure.

## Change and validation

- Keep rate limits and HTTP 5xx retryable. Preserve bare typed SDK connection
  errors for compatibility with providers that omit an underlying cause.
- When a connection error includes a cause, require a recognized HTTPX network,
  timeout, or remote-protocol failure. Local protocol/configuration failures and
  unknown wrapped exceptions fail validation.
- Reject certificate-verification failures even when nested in network errors.
  Detect cause-chain cycles conservatively without rendering exception messages.
- Emit only fixed diagnostic labels for rejected connection wrappers: invalid
  request headers/protocol, unsupported or invalid endpoint, or certificate
  verification failure when the cause chain identifies one. Never include
  credential values, exception messages, or response content. For a local protocol
  error, inspect only the already-prepared request in memory to distinguish an
  API-key header containing boundary whitespace, CR/LF, or NUL from another
  malformed header. Emit only a fixed category; never return a header name or
  value. A valid API-key header does not imply valid authentication.
- Exercise the benchmark boundary: preserve retrieval, return exit 2 and failed
  faithfulness for local errors, and preserve exit 3 for genuine transport errors.
  Re-run the real SDK offline probe and the serial suite. Prove the regression
  tests fail with the old classifier.

Reject treating the outer SDK type as proof of an outage: the offline probe
shows it cannot distinguish malformed local requests. Reject silently trimming
or replacing credentials: that changes user configuration and hides the cause.
No provider switch, threshold change, new live-call trigger, or dependency change
is part of this repair.
