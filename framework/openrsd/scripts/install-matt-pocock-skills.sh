#!/usr/bin/env bash
# Install selected mattpocock/skills into OpenRSD (.cursor/skills + ~/.codex/skills).
# Source: https://github.com/mattpocock/skills

set -euo pipefail

REPO_URL="${MATT_POCOCK_REPO_URL:-https://github.com/mattpocock/skills.git}"
CLONE_DIR="${MATT_POCOCK_CLONE_DIR:-/tmp/mattpocock-skills}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CURSOR_SKILLS="$PROJECT_ROOT/.cursor/skills"
CODEX_SKILLS="${CODEX_SKILLS:-$HOME/.codex/skills}"
INSTALL_CODEX="${INSTALL_CODEX:-1}"

# name -> path under repo skills/
declare -A SKILL_PATHS=(
  [zoom-out]="skills/engineering/zoom-out"
  [improve-codebase-architecture]="skills/engineering/improve-codebase-architecture"
  [scaffold-exercises]="skills/misc/scaffold-exercises"
)

log() { echo "[matt-pocock-install] $*"; }

clone_repo() {
  if [ -d "$CLONE_DIR/skills/engineering/zoom-out/SKILL.md" ]; then
    log "Using existing $CLONE_DIR"
    return 0
  fi
  if [ -d "$CLONE_DIR/.git" ]; then
    log "Updating $CLONE_DIR"
    git -C "$CLONE_DIR" pull --ff-only || true
    return 0
  fi
  log "Cloning $REPO_URL -> $CLONE_DIR"
  if GIT_SSL_NO_VERIFY="${GIT_SSL_NO_VERIFY:-}" git clone --depth 1 "$REPO_URL" "$CLONE_DIR" 2>/dev/null; then
    return 0
  fi
  log "git clone failed; downloading zip (wget --no-check-certificate)..."
  local zip="${CLONE_DIR}.zip"
  wget -q --no-check-certificate --timeout=120 -O "$zip" \
    "https://github.com/mattpocock/skills/archive/refs/heads/main.zip"
  rm -rf "${CLONE_DIR}-main" "$CLONE_DIR"
  unzip -q -o "$zip" -d /tmp
  mv /tmp/skills-main "$CLONE_DIR"
  rm -f "$zip"
}

install_skill() {
  local name="$1"
  local rel="${SKILL_PATHS[$name]}"
  local src="$CLONE_DIR/$rel"
  if [ ! -f "$src/SKILL.md" ]; then
    log "ERROR: missing $src/SKILL.md"
    exit 1
  fi
  local dst="$CURSOR_SKILLS/$name"
  rm -rf "$dst"
  cp -a "$src" "$dst"
  log "Cursor: $name -> $dst"

  if [ "$INSTALL_CODEX" = "1" ]; then
    mkdir -p "$CODEX_SKILLS"
    dst="$CODEX_SKILLS/$name"
    rm -rf "$dst"
    cp -a "$src" "$dst"
    log "Codex:  $name -> $dst"
  fi
}

main() {
  clone_repo
  for name in "${!SKILL_PATHS[@]}"; do
    install_skill "$name"
  done
  log "Done. Installed: zoom-out, improve-codebase-architecture, scaffold-exercises"
  log "Superpowers skills (already present): using-git-worktrees, requesting-code-review, finishing-a-development-branch"
  log "Reinstall superpowers: bash scripts/install-superpowers.sh"
}

main "$@"
