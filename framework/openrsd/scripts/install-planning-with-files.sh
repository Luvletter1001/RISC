#!/usr/bin/env bash
# Install OthmanAdi/planning-with-files into OpenRSD (Cursor skill + hooks + Codex skill).
# Source: https://github.com/OthmanAdi/planning-with-files

set -euo pipefail

REPO_URL="${PLANNING_REPO_URL:-https://github.com/OthmanAdi/planning-with-files.git}"
ZIP_URL="${PLANNING_ZIP_URL:-https://github.com/OthmanAdi/planning-with-files/archive/refs/heads/master.zip}"
SRC_DIR="${PLANNING_SRC_DIR:-/tmp/planning-with-files}"
ZIP_FILE="${PLANNING_ZIP_FILE:-/tmp/planning-with-files.zip}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CURSOR_DIR="$PROJECT_ROOT/.cursor"
CODEX_SKILLS="${CODEX_SKILLS:-$HOME/.codex/skills}"
INSTALL_CODEX="${INSTALL_CODEX:-1}"
VERSION="2.38.1"

log() { echo "[planning-install] $*"; }

fetch_source() {
  if [ -f "$SRC_DIR/.cursor/skills/planning-with-files/SKILL.md" ]; then
    log "Using existing $SRC_DIR"
    return 0
  fi
  if git clone --depth 1 "$REPO_URL" "$SRC_DIR" 2>/dev/null; then
    log "Cloned $REPO_URL"
    return 0
  fi
  log "git clone failed; downloading zip..."
  wget -q --timeout=120 -O "$ZIP_FILE" "$ZIP_URL"
  rm -rf "${SRC_DIR}-master"
  unzip -q -o "$ZIP_FILE" -d /tmp
  rm -rf "$SRC_DIR"
  mv /tmp/planning-with-files-master "$SRC_DIR"
  log "Extracted zip -> $SRC_DIR"
}

install_cursor_skill() {
  local src="$SRC_DIR/.cursor/skills/planning-with-files"
  local dst="$CURSOR_DIR/skills/planning-with-files"
  [ -f "$src/SKILL.md" ] || src="$SRC_DIR/skills/planning-with-files"
  [ -f "$src/SKILL.md" ] || { log "ERROR: planning-with-files SKILL.md not found"; exit 1; }
  rm -rf "$dst"
  cp -a "$src" "$dst"
  chmod +x "$dst"/scripts/*.sh 2>/dev/null || true
  mkdir -p "$CURSOR_DIR/planning-with-files"
  echo "$VERSION" >"$CURSOR_DIR/planning-with-files/VERSION"
  log "Skill -> $dst"
}

install_cursor_hooks() {
  mkdir -p "$CURSOR_DIR/hooks" "$CURSOR_DIR/planning-with-files"
  cp -f "$SRC_DIR/.cursor/hooks/"*.sh "$CURSOR_DIR/hooks/"
  cp -f "$SRC_DIR/.cursor/hooks/"*.ps1 "$CURSOR_DIR/hooks/" 2>/dev/null || true
  chmod +x "$CURSOR_DIR/hooks/"*.sh 2>/dev/null || true
  cp -f "$SRC_DIR/.cursor/hooks.windows.json" "$CURSOR_DIR/hooks.windows.json"

  cat >"$CURSOR_DIR/hooks.json" <<'JSON_EOF'
{
  "version": 1,
  "hooks": {
    "sessionStart": [
      {
        "command": ".cursor/superpowers/hooks/session-start"
      }
    ],
    "userPromptSubmit": [
      {
        "command": ".cursor/hooks/user-prompt-submit.sh",
        "timeout": 5
      }
    ],
    "preToolUse": [
      {
        "command": ".cursor/hooks/pre-tool-use.sh",
        "matcher": "Write|Edit|Shell|Read",
        "timeout": 5
      }
    ],
    "postToolUse": [
      {
        "command": ".cursor/hooks/post-tool-use.sh",
        "matcher": "Write|Edit",
        "timeout": 5
      }
    ],
    "stop": [
      {
        "command": ".cursor/hooks/stop.sh",
        "timeout": 10,
        "loop_limit": 3
      }
    ]
  }
}
JSON_EOF
  log "Merged hooks.json (superpowers sessionStart + planning-with-files lifecycle)"
}

install_codex_skill() {
  [ "$INSTALL_CODEX" = "1" ] || return 0
  local src="$SRC_DIR/.codex/skills/planning-with-files"
  local dst="$CODEX_SKILLS/planning-with-files"
  [ -d "$src" ] || src="$SRC_DIR/skills/planning-with-files"
  rm -rf "$dst"
  cp -a "$src" "$dst"
  chmod +x "$dst"/scripts/*.sh 2>/dev/null || true
  log "Codex skill -> $dst"
}

main() {
  fetch_source
  install_cursor_skill
  install_cursor_hooks
  install_codex_skill
  log "Done. Multi-step tasks use task_plan.md / findings.md / progress.md (or .planning/<slug>/)."
  log "Alternative: npx skills add OthmanAdi/planning-with-files --skill planning-with-files -g"
}

main "$@"
