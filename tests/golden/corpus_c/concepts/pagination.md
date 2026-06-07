# Pagination

List endpoints return one page at a time. Each response carries a cursor
pointing at the next page; keep following it until the cursor is empty to
walk the whole result set. Page size is adjustable up to a limit.
