# attune-rag — maintainer shortcuts
#
# These targets wrap common dev workflows so future-you (and future-me)
# don't have to re-derive the steps every time. Run `make help` for a
# list.
#
# Convention: targets call `uv run <tool>` so they use the project venv
# (.venv) regardless of which Python is on PATH. `uv` itself is assumed
# to be installed globally (e.g. via pyenv shim).

.DEFAULT_GOAL := help

UV := uv
ATTUNE_AUTHOR := $(UV) run attune-author

# --------------------------------------------------------------------
# Help

.PHONY: help
help:  ## Show this help.
	@awk 'BEGIN {FS = ":.*##"; printf "Targets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  %-28s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# --------------------------------------------------------------------
# Tests

.PHONY: test
test:  ## Run the unit test suite.
	$(UV) run pytest tests/unit -q

.PHONY: test-cov
test-cov:  ## Run unit tests with coverage report.
	$(UV) run pytest tests/unit --cov

# --------------------------------------------------------------------
# Help-template maintenance (.help/templates/)
#
# Background: .help/templates/<feature>/{concept,reference,task}.md
# are AI-authored docs. A pre/post hook auto-regenerates them when
# source changes, but the auto-regen SKIPS the polish pass — the
# result is accurate-but-skeletal. The proper regen runs through
# attune-author's polish + fact-check, which spends Anthropic API
# tokens. Use the targets below for that proper path.
#
# Gotcha: the 0.13.0 wheel on PyPI ships with a stale __version__
# string ("0.11.1"). `attune-author --version` therefore lies. The
# `_check-author-polish` target probes for the `--fact-check` flag
# instead, which only exists from 0.13.0 onward, so the version-string
# bug doesn't trip the check.

.PHONY: _check-author-polish
_check-author-polish:
	@$(ATTUNE_AUTHOR) regenerate --help 2>/dev/null | grep -q -- '--fact-check' || { \
	  echo "ERROR: attune-author in the venv lacks the polish step (--fact-check flag missing)."; \
	  echo "       Fix: $(UV) pip install --upgrade 'attune-author>=0.13.0'"; \
	  echo "       (Note: 'attune-author --version' may report 0.11.1 even when 0.13.0 is installed —"; \
	  echo "        the PyPI wheel has a stale version string. The --fact-check probe is authoritative.)"; \
	  exit 1; \
	}

.PHONY: help-status
help-status:  ## Show which .help/templates features are stale.
	$(ATTUNE_AUTHOR) status

.PHONY: help-regen-preview
help-regen-preview: _check-author-polish  ## Preview which features would be regenerated (free, no API calls).
	$(ATTUNE_AUTHOR) regenerate --dry-run

.PHONY: help-regen
help-regen: _check-author-polish  ## Regenerate stale .help/templates with polish + soft fact-check (spends API tokens).
	@echo "Spending Anthropic API tokens for the polish + soft fact-check pass."
	@echo "Requires ANTHROPIC_API_KEY in the environment."
	$(ATTUNE_AUTHOR) regenerate

.PHONY: help-regen-batch
help-regen-batch: _check-author-polish  ## Submit a batch regen (~50% cost; detaches — run `help-regen-resume` later).
	$(ATTUNE_AUTHOR) regenerate --batch

.PHONY: help-regen-resume
help-regen-resume: _check-author-polish  ## Resume the pending batch regen and write its templates.
	$(ATTUNE_AUTHOR) regenerate --resume

.PHONY: help-regen-status
help-regen-status:  ## Show one-shot status of the pending batch (no polling).
	$(ATTUNE_AUTHOR) regenerate --status

# --------------------------------------------------------------------
# Claude Code session hooks (.claude/hooks/) — vendored from attune-ai
#
# attune-ai's plugin/hooks/ is the canonical source. Each sibling repo
# (attune-rag, attune-author, attune-help, attune-gui) carries a
# byte-identical copy of the portable hook closure plus a drift-guard
# test that asserts the copy matches the canonical hash.
#
# See specs/sibling-claude-hooks/ in the attune umbrella workspace.

ATTUNE_AI_ROOT ?= ../attune-ai
HOOK_FILES = security_guard.py format_on_save.py compact_warning.py spec_orient.py _state.py _resume_prompt.py _transcript_size.py _sdk_gate.py spec_audit.py

.PHONY: sync-hooks
sync-hooks:  ## Re-copy session hooks from attune-ai canonical + refresh checksums.
	@if [ ! -d "$(ATTUNE_AI_ROOT)/plugin/hooks" ]; then \
		echo "Error: $(ATTUNE_AI_ROOT)/plugin/hooks not found. Set ATTUNE_AI_ROOT=<path>"; \
		exit 1; \
	fi
	@mkdir -p .claude/hooks
	@for f in $(HOOK_FILES); do \
		cp "$(ATTUNE_AI_ROOT)/plugin/hooks/$$f" ".claude/hooks/$$f"; \
		echo "  synced: $$f"; \
	done
	@(cd .claude/hooks && shasum -a 256 $(HOOK_FILES) > .canonical-sha256)
	@echo "✓ .claude/hooks/.canonical-sha256 refreshed"
