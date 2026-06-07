# Rate limiting

The server caps how many calls you may make per minute. Exceeding the
cap returns a 429 with a Retry-After header indicating how long to wait
before the next call. Stay under the quota by spacing calls out.
