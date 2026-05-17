---
name: staged-implementation-protocol
description: Use this skill whenever the user asks for a prompt to send to an implementer or reviewer of a stage from a staged-implementation checklist in this project. Trigger phrases include "crie/escreve um prompt para o codex/claude implementar", "prompt para implementar [stage X / task X.Y]", "prompt para revisar [stage X]", "comando para nova sessão revisar", "como mando isso para o codex", "prompt para o claude revisar", "review prompt", "implementation prompt". Also trigger when the user mentions a Stage from PHASE_B_IMPLEMENTATION_CHECKLIST.md (or any future *_IMPLEMENTATION_CHECKLIST.md) and is about to delegate it to another session. Skill captures the project-specific protocol for: extracting actionable scope from a concept doc (e.g., audit report, design spec) via a staged checklist; deciding whether canonical doc PR precedes code PR; authoring self-contained implementation prompts with verified refs and protected-file lists; authoring self-contained review prompts with active gap detection; enforcing implementer/reviewer separation across sessions; and recording decisions in checklist Notas de revisão. Lean slightly pushy: when in doubt about whether a request involves stage-based delegation, load this skill — under-triggering wastes the protocol's value.
---

# Staged Implementation Protocol

Protocol for delivering work that originates from a **concept doc** (audit report, design spec, requirements document) via a **staged checklist** where each stage = 1 PR and each task = 1 commit. The protocol covers the full cycle: authoring prompts for an isolated implementer session → implementer commits → authoring prompts for an isolated reviewer session → reviewer audits → iterate or merge → register decisions.

This skill is **parametric**: it does not encode a specific project phase. It encodes the patterns learned from operating this delivery model across multiple stages. The current concrete instance in the project is the Phase B engineering work driven from `docs/07_reports/phase-gates/A_code_audit.md` via `docs/05_checklists/PHASE_B_IMPLEMENTATION_CHECKLIST.md`, but the protocol applies equally to future design-spec-driven or requirement-driven staged work.

## When to use

- User asks for an implementation prompt for a specific stage/task to send to Codex or a new Claude session.
- User asks for a review prompt for a stage already committed on a branch.
- User asks how to delegate a stage's execution or audit to another session.
- User is about to write either prompt from scratch — intercept and apply this protocol.
- User mentions discrepancies between checklist text and code, or asks about updating checklist Notas de revisão after implementation.

Do **not** use this skill for:
- Authoring the concept doc itself (use documentation-authoring).
- Implementing code directly in the current session (use module-implementation-clean-arch + testing-implementation).
- Statistical analysis decisions (use model-performance-and-research-advisor).
- Writing TCC chapters (use tcc-*-writer).

## Core principles

1. **One stage = one PR; one task = one commit.** Convention enforced by `docs/08_governance/GOVERNANCE_AND_VERSIONING.md`. The protocol does not negotiate this.
2. **Implementer and reviewer must be different sessions.** Self-review is forbidden — independent verification is the point of the cycle.
3. **Notas de revisão are claims, not truth.** When auditing, treat them as the implementer's testimony to be verified against code.
4. **Canonical policy precedes code when policy is new.** If the stage introduces a new concept (categorization, semantic rule, academic methodology), the canonical doc PR ships first, code PR cites it. If the stage merely applies existing policy, code PR cites the canonical doc directly without preceding doc work.
5. **Cross-link over duplication.** Project documentation rule (`docs/INDEX.md`): one topic, one source of truth. Prompts should send the implementer to canonical docs; they should not re-state policy in the prompt body.
6. **Verify refs against current code.** File:line references in the checklist drift between stages. Always grep before citing.

## Workflow

### Phase A — Pre-flight (mandatory before authoring any prompt)

1. **Read the target stage in the staged checklist.** Extract: task list, Aceite criteria, severity, cross-link to concept doc, any existing Notas de revisão.

2. **Read the cross-linked §section in the concept doc.** Understand the original diagnosis, why this is RED/YELLOW, what evidence supports it.

3. **List dependencies on prior stages.** Which stages are merged? What patterns did they establish that this stage should replicate? Grep prior commits for the patterns the new prompt should cite ("siga o padrão do Stage X.Y").

4. **Verify file:line refs.** The checklist may cite `:1079,1082` but code shifted after prior stages. Open the file and confirm. Update the prompt with current line numbers, not stale ones.

5. **Classify the stage (CRITICAL):**

   - **Operational** (mechanical refactor; no new concept; pure pattern propagation): code PR directly. Doc updates only if a contract changes (e.g., new column shape).
   - **Policy-introducing** (new categorization, semantic rule, academic methodology — current example: Stage 8 introduced Cat A/B/C policy): **canonical doc PR ships first**. Identify which canonical docs are impacted (`docs/04_evaluation/METRICS_DEFINITIONS.md`, `docs/01_architecture/ANALYTICS_STORE_ARCHITECTURE.md`, `docs/00_overview/GLOSSARY.md`, `docs/07_reports/living-paper/20_method.md`, etc.). Author the doc PR first; only after merge, write the implementation prompt that cites the merged policy.
   - **Policy-applying** (filter, criterion, propagation deriving from existing canonical policy — current example: Stage 9 applies Cat C filter from Cat A/B/C policy already canonical): code PR cites existing canonical doc. No preceding doc PR. Optionally add a tight (~10-15 line) note to `20_method.md` after merge if methodologically defensible at paper level.

6. **List downstream consumers of any contract the stage modifies.** Grep for callsites. The implementer prompt must enumerate these so propagation is not silently missed.

7. **Decide implementer target.** Codex is good at mechanical pattern replication across N similar sites. Claude (new session) is needed when the task requires nuanced reasoning about classification, policy interpretation, or edge cases the checklist did not fully specify. Mixed stage may require multiple implementer sessions or a switch mid-stage.

### Phase B — Implementation prompt authoring

8. **Structure of an implementation prompt** (use this skeleton; adapt sections to stage classification):

   - **Context block**: project framing (Clean Architecture; analytics store silver/gold; pre-Phase B); current branch; which stages already merged; severity of this stage's bug.
   - **Required reading**: target stage in checklist; cross-link in concept doc; relevant canonical policy docs (only those actually used by the stage — do not over-cite); governance doc for branch/commit/PR conventions; AGENT_CORE for scope governance.
   - **Stage scope and diagnosis**: quote the audit/concept doc literally for the bug; cite severity; declare cross-links.
   - **Per-task guidance**: for each task in the stage, give: file:line (verified, not stale); pattern to replicate (cite prior stage if applicable); aceite literal; specific edge cases (legacy silver, NaN propagation, fallback behavior). Use `Local: [file:line](path)` markdown links so the implementer's IDE resolves them.
   - **Conventions block**: branch naming `<tipo>/<area>-<objetivo>`; commit format `<tipo>(<escopo>): <imperativo>`; "one task = one commit" rule; explicit "no PR — apenas commits" if review is delegated.
   - **Validation commands**: exact pytest paths; lint command if applicable; smoke command for CLI stages.
   - **Regras de execução (protected files list)**: enumerate files the implementer must NOT touch. This list is derived from prior stages' scope and from the categorization (Cat B post-only files; Cat C raw-only files; schemas; docs not synchronized in this stage). Without this list, escopo creep is the default failure mode.
   - **Final deliverable**: checklist Notas de revisão template with date, validation results, decisions, follow-ups.

9. **For Codex prompts**, be explicit about every file:line and every pattern to replicate — Codex performs better on mechanical specificity than abstract reasoning.

10. **For Claude (new session) prompts**, lean on referencing canonical policy and letting Claude reason. Claude can read the policy doc and infer the right classification; Codex would copy literally and miss nuance.

### Phase C — Implementer executes

11. **Out of scope of this skill** — implementer commits per conventions in isolated branch and fills Notas de revisão in the checklist with decisions made, validation evidence, and any follow-ups.

### Phase D — Review prompt authoring

12. **Structure of a review prompt** (use this skeleton):

    - **Context block including any trajectory atypical**: if the stage had unusual history (scope expanded post-implementation, policy mergeada in separate PR, fix defensivo pelo Codex), declare it upfront — reviewer needs to know what to extra-scrutinize.
    - **Required reading**: same as implementation prompt + the stage's Notas de revisão (which will be audited).
    - **Audit dimensions** — five mandatory:
      1. **Per-task coverage** — each task's aceite verified against code.
      2. **Declared-vs-implemented coherence** — Notas de revisão are claims; audit each decision against the diff.
      3. **Cross-cutting interaction with prior stages** — especially dual contracts (Stage 8), scope_spec (Stage 7), cohort awareness (Stages 4-6).
      4. **Regression checks** — full pytest baseline; protected files untouched.
      5. **Active gap detection** — list 6-10 specific gaps to investigate (gaps NOT in the checklist; things only a careful reviewer would catch).
    - **Report format**: Veredicto (APPROVE/REQUEST_CHANGES/BLOCK); Findings with severity RED/YELLOW/GREEN; Cobertura por task; Comandos verificados.
    - **Severity criteria**: RED for semantic break or contract violation; YELLOW for documented gap or pendência aceitável; GREEN for positive observation.
    - **Execution rules**: "audit-only, do not implement"; "do not trust Notas de revisão without grep-based verification"; "do not trust commit messages alone".

13. **Pre-build the protected-files list and gap-detection list** before writing the review prompt. These are stage-specific and derived from the implementation prompt's protected files plus the classification's nuances.

### Phase E — Iteration / merge

14. **If review finds issues**: follow-up commits on the same branch, then re-review (verification pass against original findings, not full re-audit — write a targeted re-review prompt that quotes the findings and asks "RESOLVED / PARTIAL / NOT_RESOLVED / REGRESSED" per finding).

15. **After merge**: ensure Notas de revisão final reflects implementation + review decisions; update task checkboxes from `[~]` to `[x]` (governance convention).

## Notas de revisão pattern

Every stage's Notas de revisão section in the staged checklist must register:

- **Date** of implementation and of review (separate entries).
- **Branch** name.
- **Validation commands** with exact results (`X passed, Y warnings`).
- **Implementation decisions** that the original checklist did not pre-specify (strategy A vs B; permissive vs strict definition; alias compatibility behavior; fallback behavior for legacy silver).
- **Confirmations** that protected files were not touched and that prior-stage invariants hold.
- **Follow-ups** (YELLOW gaps deferred to future stages or future PRs).
- **Cross-links** to merged dependent doc PRs.

The reviewer's audit cross-checks every claim above against the diff.

## Gotchas / Known failures

These are failures observed during the protocol's use that the skill exists to prevent:

- **Cargo-cult prompt copying.** Writing the new prompt by mutating a prior stage's prompt without verifying current file:line refs, without re-deciding implementer target, without rebuilding the protected-files list. **Fix:** Pre-flight steps 4-7 are non-skippable.

- **Policy-introducing stage treated as operational.** Implementer ships code that invents a new concept (e.g., Cat A/B/C-like categorization) without canonical doc to cite. The concept then lives only in code comments and the checklist Notas, drifts as future stages reinterpret. **Fix:** Pre-flight step 5 classification; if policy-introducing, doc PR first.

- **Cross-link replaced by duplication.** Implementation or review prompt re-states full policy in body instead of citing the canonical doc. Future drift between prompt copy and canonical source. **Fix:** Prompts cite, do not re-state. Length grows linearly with task count, not with policy complexity.

- **Doc PR overshoots.** Author adds ~80 lines to a canonical doc when the methodologically essential content is ~15 lines (because the rest already lives in another canonical doc). Violates "one topic, one source of truth". **Fix:** Before drafting any doc addition, grep for existing canonical coverage; add only what is genuinely new at the doc's level of abstraction.

- **Reviewer trusts Notas de revisão without audit.** Reviewer reads "validation: X passed" and treats it as verified, missing that the test population is weak or the decision diverges from the code. **Fix:** Review prompt mandates audit-against-code for every Notas claim; severity RED if declared diverges from implemented.

- **Self-review.** Implementer's session is reused to "audit". Loses independent verification entirely. **Fix:** Skill mandates separate sessions explicitly; if user asks for self-review, redirect.

- **Escopo creep documented as expansion.** Implementer adds tasks (e.g., 8.5+) ad-hoc during implementation without re-classifying or re-deciding doc-impact. **Fix:** Expansion mid-stage is acceptable only if classified as new task in checklist with separate aceite; otherwise it is creep and review must catch it.

- **Stale file:line refs propagated.** Checklist cites `:1079,1082` (state at time of checklist authoring); current code has the function at `:1102,1105` after prior stages. Implementer follows stale refs and edits wrong lines. **Fix:** Pre-flight step 4 mandatory grep verification.

- **Protected files list missing.** Implementer modifies a Stage-N file because the prompt did not say not to. Review catches it as escopo creep, requires revert + recommit. Two cycles of wasted work. **Fix:** Phase B step 8 protected-files section is non-optional.

- **Permissive vs strict decision left implicit.** Stage requires a binary classification (e.g., "is_quantile_genuine: ≥1 row or all rows?"). Implementer picks one without declaring why. Review cannot tell if decision is principled or arbitrary. **Fix:** Implementation prompt forces decision declaration in commit body and Notas; review prompt audits the rationale.

- **Downstream propagation missed.** Stage modifies a builder; 5 downstream tables consume it but were not updated. Tests pass because fixtures are isolated. Real refresh produces shape mismatch. **Fix:** Pre-flight step 6 enumerates consumers; implementation prompt lists them; review prompt confirms each was addressed or explicitly deferred.

## Outputs expected

- **For implementation request**: a self-contained markdown prompt ~200-400 lines (scale with stage complexity, not formula), structured per Phase B step 8.
- **For review request**: a self-contained markdown prompt ~150-300 lines, structured per Phase D step 12.
- **For iteration request**: a targeted re-review prompt ~50-100 lines listing prior findings with RESOLVED/PARTIAL/NOT_RESOLVED slots.

The prompt body must be self-contained because it will be pasted into a session with no memory of this conversation. Do not write "as we discussed" or "in the previous session" — write everything the implementer/reviewer needs.

## Decision shortcuts (use these to avoid over-thinking)

- **Stage touches métricas / contratos semânticos novos?** Doc PR first.
- **Stage aplica filtro de policy existente?** Code PR direto; optional tight note in `20_method.md` after merge.
- **Stage é mecânico (renomear, propagar coluna, adicionar groupby)?** Codex.
- **Stage exige raciocínio sobre categorização ou edge case?** Claude (new session).
- **Stage modifica contrato consumido downstream?** Lista de consumidores no prompt obrigatória.
- **Stage anterior é Stage 8 (dual contract)?** Review prompt incluir auditoria cruzada de ambas variantes em todo teste.
- **Stage é Cat C check (gate Stage 11, _pred_interval_negative)?** Implementation prompt blinda explicitamente contra "uniformização para post-guardrail".
- **Stage final do checklist** (smoke + pré-registro)? Protocolo muda — não é stage de código, é gate de entrega; este skill cobre o protocolo de stages de código, não o gate final.

## References

This skill operates over the project's canonical documents — it does not duplicate them:

- `docs/ai/AGENT_CORE.md` — baseline rules; scope governance; data safety.
- `docs/08_governance/GOVERNANCE_AND_VERSIONING.md` — branch/commit/PR conventions.
- `docs/ai/INDEX.md` — routing manifest for canonical docs.
- `docs/INDEX.md` — full documentation index.
- `docs/ai/skills/workflow-governance/SKILL.md` — inherited execution discipline.
- `docs/ai/skills/documentation-authoring/SKILL.md` — when canonical doc PR precedes code.
- The staged checklist of the current phase (currently `docs/05_checklists/PHASE_B_IMPLEMENTATION_CHECKLIST.md`).
- The concept doc of the current phase (currently `docs/07_reports/phase-gates/A_code_audit.md`).

## How to use this skill

When a user asks for a stage prompt:

1. **Confirm whether it is implementation or review.** Different workflows (Phase B vs Phase D).
2. **Identify concept doc and staged checklist paths.** If not stated, default to the current phase's docs but confirm if user is in a different context.
3. **Run pre-flight (Phase A).** Do not skip — this is where most failures originate.
4. **Author prompt per skeleton (Phase B or D).** Cross-link, do not duplicate.
5. **State the implementer/reviewer target explicitly** in the prompt header.
6. **Remind user about session separation** if they seem about to self-review.
7. **Offer to author follow-up prompts** (re-review after fixes, doc PR if classification demands it) without inflating the current prompt.
