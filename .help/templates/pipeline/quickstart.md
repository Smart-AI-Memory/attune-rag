---
type: quickstart
name: pipeline-quickstart
feature: pipeline
depth: quickstart
generated_at: 2026-05-20T03:20:40.538198+00:00
source_hash: f5cc845ee3957a76674338c9a162ce4a86e404c42291f721ed77a3b4c3b27569
status: verified
---

# Run a strict query without a model call

## Run the example

Install `attune-rag`, then run this example with Python. The empty directory gives the keyword retriever a corpus with no matches.

```python
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from attune_rag import DirectoryCorpus, RagPipeline

with TemporaryDirectory() as directory:
    pipeline = RagPipeline(corpus=DirectoryCorpus(Path(directory)))
    answer, result = asyncio.run(
        pipeline.run_and_generate(
            "How do I configure a workflow?",
            provider="claude",
            prompt_variant="strict",
        )
    )
    print(answer)
    print(result.fallback_used, result.abstained)
```

## Check the output

```text
The provided context does not cover this question.
True True
```

Strict mode returns this refusal locally before initializing the answer provider. To inspect retrieved evidence for your own documents, pass their directory to `DirectoryCorpus` and call `pipeline.run(query)`.

**Tags:** `pipeline`, `orchestration`, `rag`, `result`
