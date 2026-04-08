#!/bin/bash
cd "$(git rev-parse --show-toplevel 2>/dev/null)" || exit 0

# Check for meaningful changes (staged + unstaged + untracked)
changed_files=$(git diff --name-only 2>/dev/null; git diff --cached --name-only 2>/dev/null; git ls-files --others --exclude-standard 2>/dev/null)
[ -z "$changed_files" ] && exit 0

# Filter out non-code changes (memory files, settings, etc.)
code_changes=$(echo "$changed_files" | grep -v -E '^\.(claude|planning)/' | grep -v 'MEMORY\.md' | sort -u)
[ -z "$code_changes" ] && exit 0

# Count meaningful lines changed (ignore blank-only diffs)
meaningful_diff=$(git diff -- $code_changes 2>/dev/null | grep -c '^[+-][^+-]')
untracked_code=$(echo "$code_changes" | while read f; do [ -f "$f" ] && ! git ls-files --error-unmatch "$f" >/dev/null 2>&1 && echo "$f"; done)

if [ "$meaningful_diff" -lt 5 ] && [ -z "$untracked_code" ]; then
  exit 0
fi

# Build commit message from changed files
file_summary=$(echo "$code_changes" | head -5 | xargs -I{} basename {} | paste -sd, -)
num_files=$(echo "$code_changes" | wc -l | tr -d ' ')

git add $code_changes 2>/dev/null
git commit -m "WIP: auto-save progress (${num_files} files: ${file_summary})

Auto-committed by Claude Code stop hook to preserve work in progress.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>" 2>/dev/null

if [ $? -eq 0 ]; then
  echo '{"systemMessage": "Auto-committed progress to git."}'
else
  exit 0
fi
