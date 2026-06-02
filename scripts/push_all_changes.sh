#!/usr/bin/env bash
# Push KOZ / DeepReach-MPC work to GitHub.
#
# Problem: deepreach_MPC/ often has a broken nested .git (incomplete clone), so the
# parent repo (aa276_proj) never tracks those files. This script fixes that and pushes.
#
# Usage (from repo root):
#   chmod +x scripts/push_all_changes.sh
#
#   # Recommended: vendor deepreach_MPC inside aa276_proj (one repo, one push)
#   ./scripts/push_all_changes.sh
#
#   # Optional: also push deepreach_MPC edits to YOUR fork (separate remote)
#   export DEEPREACH_MPC_REMOTE="https://github.com/YOUR_USER/deepreach.git"
#   MODE=vendor_and_fork ./scripts/push_all_changes.sh
#
# Env:
#   MODE              vendor (default) | vendor_and_fork
#   BRANCH            main
#   REMOTE            origin
#   COMMIT_MSG        commit message (prompted if unset and interactive)
#   DEEPREACH_MPC_REMOTE   your fork URL (required for vendor_and_fork)
#   DEEPREACH_MPC_BRANCH   DeepReach_MPC (branch name on fork)
#   PUSH              1 to git push (default 1); 0 = commit only

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODE="${MODE:-vendor}"
BRANCH="${BRANCH:-main}"
REMOTE="${REMOTE:-origin}"
PUSH="${PUSH:-1}"
DEEPREACH_MPC_BRANCH="${DEEPREACH_MPC_BRANCH:-DeepReach_MPC}"

die() { echo "ERROR: $*" >&2; exit 1; }

deepreach_mpc_git_ok() {
  [[ -d deepreach_MPC/.git ]] || return 1
  git -C deepreach_MPC rev-parse HEAD &>/dev/null
}

fix_deepreach_mpc_for_vendor() {
  if deepreach_mpc_git_ok; then
    echo "==> deepreach_MPC has a working nested .git; removing it so aa276_proj can track files"
    echo "    (Keeps source files; only deletes deepreach_MPC/.git metadata.)"
    rm -rf deepreach_MPC/.git
  elif [[ -e deepreach_MPC/.git ]]; then
    echo "==> Removing broken deepreach_MPC/.git (incomplete clone — no HEAD)"
    rm -rf deepreach_MPC/.git
  else
    echo "==> deepreach_MPC has no .git — OK to vendor into aa276_proj"
  fi
}

push_deepreach_mpc_fork() {
  [[ -n "${DEEPREACH_MPC_REMOTE:-}" ]] || die "Set DEEPREACH_MPC_REMOTE for vendor_and_fork mode"

  WORK="$ROOT/deepreach_MPC"
  BACKUP="$ROOT/.deepreach_MPC_push_backup"
  rm -rf "$BACKUP"
  cp -R "$WORK" "$BACKUP"

  fix_deepreach_mpc_for_vendor

  echo "==> Initializing deepreach_MPC git for fork push"
  git -C "$WORK" init -b "$DEEPREACH_MPC_BRANCH"
  git -C "$WORK" remote add origin "$DEEPREACH_MPC_REMOTE" 2>/dev/null || \
    git -C "$WORK" remote set-url origin "$DEEPREACH_MPC_REMOTE"

  git -C "$WORK" add -A
  if git -C "$WORK" diff --cached --quiet; then
    echo "==> No changes in deepreach_MPC to commit on fork"
  else
    git -C "$WORK" commit -m "KOZ Cw6DKoz (SI), exact pretrain loss fix, run_cw6d_koz.sh"
    if [[ "$PUSH" == "1" ]]; then
      git -C "$WORK" push -u origin "$DEEPREACH_MPC_BRANCH"
      echo "==> Pushed deepreach_MPC to $DEEPREACH_MPC_REMOTE ($DEEPREACH_MPC_BRANCH)"
    fi
  fi

  # Restore vendored tree without nested .git for parent commit
  rm -rf "$WORK"
  mv "$BACKUP" "$WORK"
  rm -rf "$WORK/.git"
}

stage_parent_repo() {
  echo "==> Staging aa276_proj changes"
  # Ensure .DS_Store never ships
  git restore --staged .DS_Store 2>/dev/null || true
  git add -A
  git restore --staged .DS_Store 2>/dev/null || true

  if git diff --cached --quiet; then
    echo "No staged changes in aa276_proj."
    return 1
  fi
  git status
  return 0
}

commit_parent_repo() {
  local msg="${COMMIT_MSG:-}"
  if [[ -z "$msg" ]]; then
    if [[ -t 0 ]]; then
      read -r -p "Commit message: " msg
    else
      msg="Switch KOZ BRT to DeepReach-MPC; SI Cw6DKoz; u_max=0.15 m/s^2"
    fi
  fi
  [[ -n "$msg" ]] || die "Empty commit message"
  git commit -m "$msg"
}

echo "Repo: $ROOT"
echo "Mode: $MODE | branch: $BRANCH | remote: $REMOTE"

case "$MODE" in
  vendor)
    fix_deepreach_mpc_for_vendor
    ;;
  vendor_and_fork)
    push_deepreach_mpc_fork
    ;;
  *)
    die "Unknown MODE=$MODE (use vendor or vendor_and_fork)"
    ;;
esac

if stage_parent_repo; then
  commit_parent_repo
  if [[ "$PUSH" == "1" ]]; then
    echo "==> Pushing $REMOTE/$BRANCH"
    git push "$REMOTE" "$BRANCH"
    echo "Done: $(git remote get-url "$REMOTE" 2>/dev/null || echo $REMOTE)"
  else
    echo "Commit created locally (PUSH=0)."
  fi
else
  echo "Nothing to commit on aa276_proj."
fi
