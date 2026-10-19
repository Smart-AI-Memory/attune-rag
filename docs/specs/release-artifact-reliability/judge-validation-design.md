# Reject incomplete faithfulness verdicts

The 1.2.1 repair must distinguish an explicit zero-claim verdict from a
missing verdict. The October 19 review reproduced a nonempty answer scoring
1.0 when either a tool block or text JSON contained `{}`. A partial payload
containing only supported claims also scored 1.0. A complete unsupported
verdict scored 0.0, and a complete zero-claim verdict scored 1.0 as intended.
These were injected-client experiments with network connections forbidden.

## Cases and change

- Require `supported_claims`, `unsupported_claims`, and `reasoning` before
  computing a score, for tool output, text JSON, and subscription output.
- Reject every incomplete combination with a diagnostic `RuntimeError`.
  Report missing field names, without including answer or source content.
- Keep complete zero-claim verdicts and empty-answer short circuits valid.
- Preserve existing list/type checks and intentional claim-item string
  conversion. Tightening that compatibility behavior is outside this repair.
- Keep the exception a validation failure: the benchmark must not classify
  it as transient provider unavailability or turn it into a passing score.

Validate all seven incomplete field combinations through each route, retain
positive controls, and run the new tests against the previous parser to prove
they detect the defect. Then run the full offline suite and artifact parity
check. Live provider response quality requires the separately authorized CI
benchmark; deterministic tests do not establish that quality.

## Alternatives rejected

- Default missing fields to empty values: this creates the reproduced false
  perfect score and cannot distinguish absence from a deliberate refusal.
- Return zero for malformed output: this invents a measured faithfulness
  result. Raising preserves the distinction between invalid evidence and a
  valid unsupported-claim verdict.
- Replace the parser with full JSON Schema validation in this patch: that
  would also change established claim-item coercion and unknown-field behavior,
  expanding the release beyond the confirmed missing-field defect.
