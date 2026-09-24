---
name: merge-review
description: Test, review, then merge a feature branch into main and push. Only when the user runs /merge-review.
argument-hint: "[feature-branch, default: the one unmerged remote branch]"
disable-model-invocation: true
context: fork
agent: merge-reviewer
background: false
---

Feature branch: $ARGUMENTS

If that is empty, `git fetch origin` and take the remote branches not yet
merged into main (`git branch -r --no-merged origin/main`, leaving out
`origin/HEAD`). Exactly one: that is the branch. None or several: stop, and
report them rather than guess.

Run your full workflow on it exactly as your instructions describe - tests and
benchmark against main, smoke test, review, and only then merge and push -
and report the outcome.
