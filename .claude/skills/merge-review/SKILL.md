---
name: merge-review
description: Test, review, then merge a feature branch into main and push. Only when the user runs /merge-review.
argument-hint: "[feature-branch, default main-ob0jw0]"
disable-model-invocation: true
context: fork
agent: merge-reviewer
background: false
---

Feature branch: $ARGUMENTS (if empty, use `main-ob0jw0`).

Run your full workflow on it exactly as your instructions describe - tests and
benchmark against main, smoke test, review, and only then merge and push -
and report the outcome.
