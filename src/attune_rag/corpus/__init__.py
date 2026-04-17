"""Corpus protocol + implementations.

- CorpusProtocol, RetrievalEntry (task 1.2)
- DirectoryCorpus (task 1.2)
- AttuneHelpCorpus (task 1.3, opt. dep)
"""

from .base import CorpusProtocol, RetrievalEntry
from .directory import DirectoryCorpus

__all__ = ["CorpusProtocol", "RetrievalEntry", "DirectoryCorpus"]
