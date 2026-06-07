# Upload files

To send a binary attachment, use a multipart form body instead of JSON.
Pass a file handle; the client streams it so large attachments do not
load fully into memory.
