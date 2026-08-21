#!/usr/bin/env bash
# Install krzysztofdudek/ResearcherSkill into OpenRSD (Cursor + Codex).
# Source: https://github.com/krzysztofdudek/ResearcherSkill

set -euo pipefail

REPO_URL="${RESEARCHER_REPO_URL:-https://github.com/krzysztofdudek/ResearcherSkill.git}"
ZIP_URL="${RESEARCHER_ZIP_URL:-https://github.com/krzysztofdudek/ResearcherSkill/archive/refs/heads/main.zip}"
SRC_DIR="${RESEARCHER_SRC_DIR:-/tmp/ResearcherSkill}"
ZIP_FILE="${RESEARCHER_ZIP_FILE:-/tmp/ResearcherSkill.zip}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CURSOR_DST="$PROJECT_ROOT/.cursor/skills/researcher"
CODEX_SKILLS="${CODEX_SKILLS:-$HOME/.codex/skills}"
CODEX_DST="$CODEX_SKILLS/researcher"
INSTALL_CODEX="${INSTALL_CODEX:-1}"
VERSION="1.6.0"

log() { echo "[researcher-install] $*"; }

fetch_source() {
  local skill="$SRC_DIR/skills/researcher/SKILL.md"
  if [ -f "$skill" ]; then
    log "Using existing $SRC_DIR"
    return 0
  fi
  if GIT_SSL_NO_VERIFY=1 git clone --depth 1 "$REPO_URL" "$SRC_DIR" 2>/dev/null; then
    log "Cloned $REPO_URL"
    return 0
  fi
  log "git clone failed; downloading zip..."
  if command -v wget >/dev/null 2>&1; then
    wget -q --timeout=120 -O "$ZIP_FILE" "$ZIP_URL"
  elif command -v python3 >/dev/null 2>&1; then
    python3 - <<PY
import urllib.request
urllib.request.urlretrieve("$ZIP_URL", "$ZIP_FILE")
PY
  else
    log "ERROR: need git, wget, or python3 to fetch ResearcherSkill"
    exit 1
  fi
  rm -rf "${SRC_DIR}-main"
  unzip -q -o "$ZIP_FILE" -d /tmp
  rm -rf "$SRC_DIR"
  mv /tmp/ResearcherSkill-main "$SRC_DIR"
  log "Extracted zip -> $SRC_DIR"
}

install_skill() {
  local src="$SRC_DIR/skills/researcher/SKILL.md"
  [ -f "$src" ] || { log "ERROR: skills/researcher/SKILL.md not found in $SRC_DIR"; exit 1; }
  mkdir -p "$(dirname "$CURSOR_DST")"
  rm -rf "$CURSOR_DST"
  mkdir -p "$CURSOR_DST"
  cp -f "$src" "$CURSOR_DST/SKILL.md"
  log "Cursor skill -> $CURSOR_DST/SKILL.md"
  if [ "$INSTALL_CODEX" = "1" ]; then
    mkdir -p "$CODEX_SKILLS"
    rm -rf "$CODEX_DST"
    mkdir -p "$CODEX_DST"
    cp -f "$src" "$CODEX_DST/SKILL.md"
    log "Codex skill -> $CODEX_DST/SKILL.md"
  fi
}

ensure_gitignore() {
  local gi="$PROJECT_ROOT/.gitignore"
  touch "$gi"
  for entry in .lab/ run.log; do
    if ! grep -qxF "$entry" "$gi" 2>/dev/null; then
      printf '\n# ResearcherSkill experiment lab (untracked)\n%s\n' "$entry" >>"$gi"
      log "Added $entry to .gitignore"
    fi
  done
}

main() {
  fetch_source
  install_skill
  ensure_gitignore
  log "Done (v$VERSION). Trigger: ask the agent to enter researcher mode or optimize a measurable metric."
  log "Guide: https://github.com/krzysztofdudek/ResearcherSkill/blob/main/GUIDE.md"
}

main "$@"
