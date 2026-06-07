# Retries and backoff

When a job raises an exception, the scheduler can re-attempt it.
Each attempt waits longer than the last using exponential backoff,
capped at a maximum delay. After the attempt budget is exhausted the
job is marked failed and moved to the dead-letter queue.
