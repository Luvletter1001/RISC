#!/usr/bin/env bash
# Install hermes-agent-rtk-caveman into OpenRSD (Cursor skills + optional home bin/templates).
# Source: https://github.com/adityahimaone/hermes-agent-rtk-caveman

set -euo pipefail

REPO_URL="${HERMES_REPO_URL:-https://github.com/adityahimaone/hermes-agent-rtk-caveman.git}"
CLONE_DIR="${HERMES_CLONE_DIR:-/tmp/hermes-agent-rtk-caveman}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CURSOR_SKILLS="$PROJECT_ROOT/.cursor/skills"
CODEX_SKILLS="${CODEX_SKILLS:-$HOME/.codex/skills}"
BIN_DIR="${BIN_DIR:-$HOME/bin}"
TEMPLATES_DIR="${TEMPLATES_DIR:-$HOME/templates}"
INSTALL_CODEX="${INSTALL_CODEX:-1}"

log() { echo "[hermes-install] $*"; }

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
  done < <(find "$CLONE_DIR/skills" -name SKILL.md | sort)
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
  done < <(find "$CLONE_DIR/skills" -name SKILL.md | sort)
  log "Installed $count skills -> $CODEX_SKILLS"
}

install_scripts_and_templates() {
  mkdir -p "$BIN_DIR" "$TEMPLATES_DIR"
  cp -f "$CLONE_DIR/scripts/"*.sh "$BIN_DIR/"
  chmod +x "$BIN_DIR/"*.sh
  cp -f "$CLONE_DIR/templates/"*.txt "$TEMPLATES_DIR/"
  log "Scripts -> $BIN_DIR, templates -> $TEMPLATES_DIR"
}

install_bash_aliases() {
  local bashrc="$HOME/.bashrc"
  local marker="# Hermes Agent RTK+Caveman (OpenRSD)"
  if grep -qF "$marker" "$bashrc" 2>/dev/null; then
    log "Bash aliases already present"
    return 0
  fi
  cat >>"$bashrc" <<'EOF'

# Hermes Agent RTK+Caveman (OpenRSD)
export PATH="$HOME/bin:$PATH"
alias cgs='$HOME/bin/caveman_wrapper.sh git-status'
alias cgl='$HOME/bin/caveman_wrapper.sh git-log'
alias clint='$HOME/bin/caveman_wrapper.sh lint'
alias ctest='$HOME/bin/caveman_wrapper.sh test-results'
alias gs='rtk git status'
alias gl='rtk git log --oneline -10'
alias gd='rtk git diff'
# End Hermes Agent RTK+Caveman
EOF
  log "Aliases appended to $bashrc"
}

maybe_install_caveman_npm() {
  if command -v caveman >/dev/null 2>&1; then
    log "caveman CLI already on PATH"
    return 0
  fi
  if command -v npm >/dev/null 2>&1; then
    log "Installing caveman via npm (global)..."
    npm install -g caveman || log "WARN: npm install -g caveman failed; Caveman wrappers may need manual setup"
  else
    log "WARN: npm not found; skip caveman install"
  fi
}

main() {
  clone_repo
  install_cursor_skills
  install_codex_skills
  install_scripts_and_templates
  install_bash_aliases
  maybe_install_caveman_npm
  log "Done. Reload shell or: source ~/.bashrc"
}

main "$@"
