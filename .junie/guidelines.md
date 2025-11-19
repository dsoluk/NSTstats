# Junie Pro Working Guidelines

These guidelines define how Junie Pro operates in your repository. They emphasize minimal, precise changes, clear communication, and respect for your project’s conventions.

## 1) Objectives and success criteria
- Deliver exactly what the latest Effective Issue requests (latest‑first interpretation).
- Prefer minimal, targeted changes that solve the issue and preserve existing behavior elsewhere.
- Communicate clearly, ask when something is ambiguous, and provide verifiable outcomes.

## 2) Task interpretation: Latest‑First and UserPlan
- Latest‑First Principle: the most recent issue update overrides prior context.
- If you provide a UserPlan, Junie follows it step‑by‑step. Any deviation requires your explicit approval.

## 3) Modes of operation (decision rules)
- CHAT: Quick Q&A; no repo reads or edits.
- ADVANCED_CHAT: Explain/design/assess; read‑only exploration (search/open files) allowed; no edits or runs.
- FAST_CODE: Truly trivial change (1–3 steps, single file). If it grows beyond that, switch to CODE.
- CODE: Non‑trivial edits, multi‑file, investigation, or tests. Provide status updates via `update_status`; finish with `submit`.
- RUN_VERIFY: Run short, safe commands/tests to gather evidence; no edits.

## 4) Tooling rules (strict)
- Use specialized tools over general shell commands.
- Match the project's OS and shell:
  - On Windows: use PowerShell semantics and backslashes in paths.
  - On macOS/Linux: use Bash/Zsh semantics and forward slashes.
- Do not mix tools with shell commands in a single call.
- Creation/edits: only via the provided patch tool; do not use shell to create/modify files.
- Renames: when renaming any code element, exclusively use the `rename_element` tool so all references are updated safely.
- Repo search: use `search_project` with short keywords; avoid broad/semantic descriptions.
- File exploration: prefer `get_file_structure` before opening; open only relevant slices.
- Long‑running/background: only when necessary; avoid interactive commands.

## 5) When to edit vs. read‑only
- Edit only if the Effective Issue truly requires changes.
- Keep diffs minimal and localized; match existing code style and patterns.
- If a requested change risks breaking established flows, propose an alternative and ask for confirmation.

## 6) Running code and tests
- Only run the app/tests when:
  - The issue asks for it; or
  - Reproduction/verification of a bug or validation of a fix is needed; or
  - Code was changed that has relevant tests.
- Prefer targeted runs (specific tests, scripts) over full‑suite runs.

## 7) Documentation updates (scoped)
- Update docs only when behavior, public interfaces, or user‑facing CLI changes.
- Keep doc edits succinct and close to the code change; add examples only where they clarify usage.
- Respect existing docs structure and tone.

## 8) Code style & readability
- Mirror the project’s existing style: naming, imports order, comments, and formatting.
- Aim for simple, readable logic; avoid premature optimization.
- Keep functions cohesive with clear inputs/outputs; avoid global state.
- Import only what is needed; remove unused imports or dead code introduced by the change.
- Comments: explain non‑obvious decisions, corner cases, and invariants; avoid restating the obvious.

## 9) Error handling & robustness
- Handle expected failure modes explicitly; prefer clear error messages.
- Fail early on invalid input; do not silently swallow exceptions.
- Preserve existing error semantics unless the task requires changing them.

## 10) Modular design
- Favor small, focused functions/classes that can be reused and tested.
- Keep boundaries clean between adapters, application logic, and CLI orchestration.

## 11) Performance considerations
- Optimize only where necessary or where the issue calls it out.
- If a change affects performance‑critical paths, include a short note on complexity/impact.

## 12) Testing expectations
- For code changes that alter behavior, add or adjust focused tests when appropriate.
- Prefer unit‑level coverage around the changed logic; avoid sprawling integration changes unless needed.

## 13) Communication & deliverables
- Ask concise clarification questions when requirements are ambiguous, conflicting, or incomplete.
- Status updates: only in CODE mode via `update_status`; final result via `submit` with a brief Markdown summary.
- Response formatting: use Markdown with code fenced blocks for multi‑line code and inline backticks for identifiers.

## 14) Data, secrets, and safety
- Never expose secrets or write outside the project directory.
- Be mindful of large files and generated artifacts; don’t commit or duplicate large data unless requested.
