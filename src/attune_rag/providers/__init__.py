"""Optional LLM provider adapters. Implementation in task 1.7.

Each adapter is behind a pip extra:

- attune-rag[claude]  -> ClaudeProvider
- attune-rag[openai]  -> OpenAIProvider
- attune-rag[gemini]  -> GeminiProvider

Adapters lazy-import their SDKs so attune-rag installs
cleanly without any provider SDK.
"""
