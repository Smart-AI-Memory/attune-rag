# Handle failures

Configure the attempt budget and backoff policy per job. A job that
keeps raising errors is eventually parked in the dead-letter queue
for inspection. You can requeue a dead-lettered job manually after
fixing the underlying cause.
