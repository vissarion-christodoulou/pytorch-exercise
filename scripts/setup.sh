#!/usr/bin/env bash
#
# Bootstrap the Linux/WSL2 environment for the distributed MLP training exercise.
#
# Idempotent - safe to re-run. It creates a Python 3.11 virtualenv, installs the
# pinned dependency set, builds hivemind from the git hash required by the
# assignment, and verifies the result end to end.
#
# Usage:
#   ./scripts/setup.sh              # full setup
#   ./scripts/setup.sh --skip-verify
#
# Override locations with environment variables:
#   VENV_DIR=/path/to/venv HIVEMIND_SRC=/path/to/clone ./scripts/setup.sh

set -euo pipefail

# The assignment mandates this exact hivemind revision.
HIVEMIND_COMMIT="4d5c41495be082490ea44cce4e9dd58f9926bb4e"
HIVEMIND_REPO="https://github.com/learning-at-home/hivemind.git"

PYTHON_VERSION="3.11"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/pluralis}"
HIVEMIND_SRC="${HIVEMIND_SRC:-$HOME/src/hivemind}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCKFILE="$REPO_ROOT/requirements.lock.txt"

SKIP_VERIFY=0
[[ "${1:-}" == "--skip-verify" ]] && SKIP_VERIFY=1

log() { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\n\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# --- preflight ---------------------------------------------------------------
# hivemind needs the libp2p `p2pd` daemon, which has no native Windows build.
# Running this from Git Bash / MSYS would fail much later and far less clearly.
[[ "$(uname -s)" == "Linux" ]] || die "This must run on Linux (use WSL2), not $(uname -s)."
[[ -f "$LOCKFILE" ]] || die "Missing $LOCKFILE"

# The venv must live on the Linux filesystem. Virtualenvs on the /mnt/c DrvFs
# mount are slow and hit file-locking errors during installs.
case "$VENV_DIR" in
    /mnt/*) die "VENV_DIR is on the Windows mount ($VENV_DIR). Use a path under \$HOME." ;;
esac

# --- uv ----------------------------------------------------------------------
# uv provides the interpreter as well as the installer: Ubuntu 26.04 ships only
# Python 3.14, which hivemind does not support (setup.py caps out at 3.12).
UV_BIN="$HOME/.local/bin/uv"
if ! command -v uv >/dev/null 2>&1 && [[ ! -x "$UV_BIN" ]]; then
    log "Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
command -v uv >/dev/null 2>&1 && UV_BIN="$(command -v uv)"

log "Installing CPython $PYTHON_VERSION"
"$UV_BIN" python install "$PYTHON_VERSION"

# --- virtualenv --------------------------------------------------------------
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    log "Creating virtualenv at $VENV_DIR"
    "$UV_BIN" venv --python "$PYTHON_VERSION" "$VENV_DIR"
else
    log "Reusing virtualenv at $VENV_DIR"
fi
VENV_PY="$VENV_DIR/bin/python"

# --- dependencies ------------------------------------------------------------
# The lockfile carries torch's CPU wheel index, so this pulls torch 2.x+cpu
# rather than the default PyPI build and its ~2.5GB of unused CUDA libraries.
# It also supplies hivemind's build-time requirements (setuptools<81, wheel,
# grpcio-tools), which the next step relies on.
log "Installing pinned dependencies"
"$UV_BIN" pip install --python "$VENV_PY" -r "$LOCKFILE"

# --- hivemind ----------------------------------------------------------------
if [[ ! -d "$HIVEMIND_SRC/.git" ]]; then
    log "Cloning hivemind into $HIVEMIND_SRC"
    mkdir -p "$(dirname "$HIVEMIND_SRC")"
    git clone "$HIVEMIND_REPO" "$HIVEMIND_SRC"
fi

log "Checking out hivemind @ $HIVEMIND_COMMIT"
git -C "$HIVEMIND_SRC" fetch --quiet origin
git -C "$HIVEMIND_SRC" checkout --quiet "$HIVEMIND_COMMIT"
ACTUAL="$(git -C "$HIVEMIND_SRC" rev-parse HEAD)"
[[ "$ACTUAL" == "$HIVEMIND_COMMIT" ]] || die "hivemind is at $ACTUAL, expected $HIVEMIND_COMMIT"

# --no-build-isolation: hivemind's setup.py imports grpc_tools.protoc and
#   pkg_resources at build time but declares no [build-system] requires, so an
#   isolated build environment would not have them.
# --no-deps: the lockfile above already pins hivemind's full dependency set.
#   If hivemind's requirements.txt ever changes, regenerate the lockfile.
# The build downloads a checksummed prebuilt p2pd binary; no Go toolchain needed.
log "Installing hivemind from source"
"$UV_BIN" pip install --python "$VENV_PY" --no-build-isolation --no-deps "$HIVEMIND_SRC"

# --- verify ------------------------------------------------------------------
if [[ "$SKIP_VERIFY" -eq 1 ]]; then
    log "Skipping verification (--skip-verify)"
else
    log "Verifying the installation"
    "$VENV_PY" "$REPO_ROOT/scripts/verify_env.py"
fi

cat <<EOF

Setup complete. Activate the environment with:

    source $VENV_DIR/bin/activate

EOF
