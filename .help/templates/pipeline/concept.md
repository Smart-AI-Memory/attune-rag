---
type: concept
name: pipeline-concept
feature: pipeline
depth: concept
generated_at: 2026-05-15T20:01:28.739702+00:00
source_hash: f5cc845ee3957a76674338c9a162ce4a86e404c42291f721ed77a3b4c3b27569
status: generated
---

# Pipeline

`RagPipeline` is an LLM-agnostic orchestrator that takes a query, retrieves relevant context from a corpus, assembles a grounded prompt, and optionally calls an LLM — all in a single method call.

## How the pieces fit together

A pipeline run flows through four cooperating components:

| Component | Role |
|-----------|------|
| **Corpus** (`CorpusProtocol`) | Stores the documents to search |
| **Retriever** (`RetrieverProtocol`) | Selects the most relevant passages for a query |
| **Prompt builder** | Assembles the retrieved context and query into a prompt variant |
| **LLM provider** | Generates a response (only required for `run_and_generate`) |

You wire these together when you construct `RagPipeline`, then call `run()` or `run_and_generate()` to execute the full sequence. Optional components — `QueryExpander` and `LLMReranker` — slot in between retrieval and prompt assembly to broaden or reorder results before the prompt is built.

## `RagPipeline` — the orchestrator

`RagPipeline.__init__` accepts a corpus, retriever, expander, and reranker, all optional so you can start with defaults and add components incrementally.

Two execution paths are available:

- **`run(query, k, prompt_variant)`** — retrieves `k` passages, builds an augmented prompt, and returns a `RagResult` without calling any LLM. Use this when you supply your own LLM or want to inspect the prompt before generation.
- **`run_and_generate(query, provider, ...)`** — does everything `run` does, then calls the specified `LLMProvider` and returns both the generated text and the `RagResult`.

If no relevant context is found in the corpus, the pipeline substitutes a fallback prompt that instructs the LLM to answer honestly rather than invent information.

## `RagResult` — the output record

Every `run()` call returns a `RagResult` dataclass. Its fields give you the full picture of what happened:

| Field | Type | What it tells you |
|-------|------|-------------------|
| `augmented_prompt` | `str` | The exact prompt sent to the LLM |
| `citation` | `CitationRecord` | Provenance for the retrieved context |
| `confidence` | `float` | Retrieval confidence score |
| `fallback_used` | `bool` | Whether the corpus returned no usable context |
| `elapsed_ms` | `float` | Wall-clock time for the full run |
| `context` | `str` | The raw retrieved context |
| `claim_citations` | `tuple[ClaimCitation, ...]` | Per-claim source attribution |
| `used_native_citations` | `bool` | Whether the LLM's own citation format was used |

`fallback_used: True` is a useful signal in production: it means the query fell outside the corpus's scope and the answer came from the model's weights rather than your documents.

## When the pipeline matters

Use `RagPipeline` whenever you need retrieval-augmented answers with traceable provenance. The citation fields on `RagResult` let you show users exactly which source passages informed each answer — something a plain LLM call cannot provide.
