"""Corpus protocol + implementations.

- CorpusProtocol, RetrievalEntry (task 1.2)
- DirectoryCorpus (task 1.2)
- AttuneHelpCorpus (task 1.3, opt. dep)
- AliasInfo, DuplicateAliasError (template-editor M1 task #2)
"""

from .base import (
    AliasInfo,
    CorpusProtocol,
    DuplicateAliasError,
    RetrievalEntry,
)
from .directory import DirectoryCorpus

__all__ = [
    "AliasInfo",
    "CorpusProtocol",
    "DirectoryCorpus",
    "DuplicateAliasError",
    "RetrievalEntry",
]
