# Persistence

Jobs and their state are written to a durable store so that pending
work survives a process restart or crash. On startup the scheduler
reloads incomplete jobs from the store and resumes them. The store
backend is pluggable (SQLite by default, Redis optional).
