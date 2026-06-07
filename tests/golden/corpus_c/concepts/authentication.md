# Authentication

Requests are authorized with a bearer token supplied in the
Authorization header. Obtain a key from the dashboard and pass it when
constructing the client. Tokens can be rotated; an expired token yields
a 401 and must be replaced.
