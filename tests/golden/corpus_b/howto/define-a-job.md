# Define a job

Decorate a function with `@job` to register it. The function body is
the unit of work; its arguments are serialized when enqueued. Return
values are discarded. Keep a job idempotent so a re-attempt is safe.
