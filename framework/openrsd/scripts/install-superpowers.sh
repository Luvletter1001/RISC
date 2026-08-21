#!/usr/bin/env bash
# Install obra/superpowers into OpenRSD (Cursor skills + session hook + Codex skills).
# Source: https://github.com/obra/superpowers

set -euo pipefail

REPO_URL="${SUPERPOWERS_REPO_URL:-https://github.com/obra/superpowers.git}"
CLONE_DIR="${SUPERPOWERS_CLONE_DIR:-/tmp/superpowers}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CURSOR_SKILLS="$PROJECT_ROOT/.cursor/skills"
CURSOR_SUPER="$PROJECT_ROOT/.cursor/superpowers"
CODEX_SKILLS="${CODEX_SKILLS:-$HOME/.codex/skills}"
INSTALL_CODEX="${INSTALL_CODEX:-1}"
VERSION="5.1.0"

log() { echo "[superpowers-install] $*"; }

clone_repo() {
  if [ -d "$CLONE_DIR/.git" ]; then
    log "Updating $CLONE_DIR"
    git -C "$CLONE_DIR" pull --ff-only || true
  else
    log "Cloning $REPO_URL -> $CLONE_DIR"
    git clone --depth 1 "$REPO_URL" "$CLONE_DIR"
  fi
}

install_cursor_skills() {
  mkdir -p "$CURSOR_SKILLS"
  local count=0
  while IFS= read -r skill_md; do
    local skill_dir name target
    skill_dir="$(dirname "$skill_md")"
    name="$(basename "$skill_dir")"
    target="$CURSOR_SKILLS/$name"
    rm -rf "$target"
    cp -a "$skill_dir" "$target"
    count=$((count + 1))
  done < <(find "$CLONE_DIR/skills" -maxdepth 2 -name SKILL.md | sort)
  log "Installed $count skills -> $CURSOR_SKILLS"
}

install_codex_skills() {
  [ "$INSTALL_CODEX" = "1" ] || return 0
  mkdir -p "$CODEX_SKILLS"
  local count=0
  while IFS= read -r skill_md; do
    local skill_dir name target
    skill_dir="$(dirname "$skill_md")"
    name="$(basename "$skill_dir")"
    target="$CODEX_SKILLS/$name"
    rm -rf "$target"
    cp -a "$skill_dir" "$target"
    count=$((count + 1))
  done < <(find "$CLONE_DIR/skills" -maxdepth 2 -name SKILL.md | sort)
  log "Installed $count skills -> $CODEX_SKILLS"
}

install_session_hook() {
  mkdir -p "$CURSOR_SUPER/hooks"
  cp -f "$CLONE_DIR/hooks/session-start" "$CURSOR_SUPER/hooks/session-start"
  chmod +x "$CURSOR_SUPER/hooks/session-start"

  # Point bootstrap at project .cursor/skills (flat layout).
  cat >"$CURSOR_SUPER/hooks/session-start" <<'HOOK_EOF'
#!/usr/bin/env bash
# Superpowers sessionStart for OpenRSD (adapted from obra/superpowers hooks/session-start)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CURSOR_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SKILL_FILE="${CURSOR_DIR}/skills/using-superpowers/SKILL.md"

warning_message=""
legacy_skills_dir="${HOME}/.config/superpowers/skills"
if [ -d "$legacy_skills_dir" ]; then
  warning_message="\n\n<important-reminder>IN YOUR FIRST REPLY AFTER SEEING THIS MESSAGE YOU MUST TELL THE USER: Superpowers legacy skills at ~/.config/superpowers/skills are ignored. Move custom skills to ~/.cursor/skills or .cursor/skills.</important-reminder>"
fi

using_superpowers_content=$(cat "$SKILL_FILE" 2>&1 || echo "Error reading using-superpowers at ${SKILL_FILE}")

escape_for_json() {
  local s="$1"
  s="${s//\\/\\\\}"
  s="${s//\"/\\\"}"
  s="${s//$'\n'/\\n}"
  s="${s//$'\r'/\\r}"
  s="${s//$'\t'/\\t}"
  printf '%s' "$s"
}

using_superpowers_escaped=$(escape_for_json "$using_superpowers_content")
warning_escaped=$(escape_for_json "$warning_message")
session_context="<EXTREMELY_IMPORTANT>\nYou have superpowers.\n\n**Below is the full content of your 'using-superpowers' skill. For other superpowers skills, read and follow .cursor/skills/<name>/SKILL.md when relevant:**\n\n${using_superpowers_escaped}\n\n${warning_escaped}\n</EXTREMELY_IMPORTANT>"

printf '{\n  "additional_context": "%s"\n}\n' "$session_context"
exit 0
HOOK_EOF
  chmod +x "$CURSOR_SUPER/hooks/session-start"

  cat >"$PROJECT_ROOT/.cursor/hooks.json" <<'JSON_EOF'
{
  "version": 1,
  "hooks": {
    "sessionStart": [
      {
        "command": ".cursor/superpowers/hooks/session-start"
      }
    ]
  }
}
JSON_EOF

  echo "$VERSION" >"$CURSOR_SUPER/VERSION"
  log "Session hook -> .cursor/hooks.json"
}

main() {
  clone_repo
  install_cursor_skills
  install_codex_skills
  install_session_hook
  log "Done. Open a new Cursor Agent chat to load sessionStart bootstrap."
  log "Official plugin alternative: /add-plugin superpowers in Cursor chat."
}

main "$@"
