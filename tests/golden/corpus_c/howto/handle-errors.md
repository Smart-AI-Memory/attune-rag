# Handle errors

Non-success status codes raise an ApiError carrying the code and message.
Transient server-side problems (5xx) are retried automatically with
backoff; client mistakes (4xx) are not. Wrap calls to catch ApiError.
