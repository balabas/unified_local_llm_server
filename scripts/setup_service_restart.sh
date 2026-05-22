#!/usr/bin/env bash
# Sets up:
#   1. Systemd service files for LLM providers  (idempotent — skips existing)
#   2. Polkit rule for passwordless service restart
#
# Run once with sudo: sudo bash scripts/setup_service_restart.sh
#
# Ollama is skipped — its service is managed by the Ollama installer.
# For other providers the service file is created only when the required
# binary / working directory is found; otherwise that provider is skipped
# with a notice.
#
# Path overrides (env vars):
#   LLAMA_CPP_DIR      default: $HOME/Documents/projects/llama_cpp
#   UNSLOTH_WORKDIR    default: $HOME/Documents/projects/unsloth_server
#   HF_HOME            default: /mnt/M/llms_models/hg_fc

set -euo pipefail

USER="${SUDO_USER:-$(whoami)}"
USERHOME=$(getent passwd "$USER" | cut -d: -f6)
UID_NUM=$(id -u "$USER")
XDG_RUNTIME_DIR="/run/user/$UID_NUM"
HF_HOME="${HF_HOME:-/mnt/M/llms_models/hg_fc}"

SYSTEMD_DIR="/etc/systemd/system"
POLKIT_DIR="/etc/polkit-1/rules.d"
POLKIT_FILE="$POLKIT_DIR/10-llm-restart.rules"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROVIDERS_YAML="$SCRIPT_DIR/../providers.yaml"
[ ! -f "$PROVIDERS_YAML" ] && PROVIDERS_YAML="$SCRIPT_DIR/../providers.example.yaml"

# ---------------------------------------------------------------------------
# Service file templates
# Each function prints the unit file to stdout, or prints a SKIP notice and
# returns 1 when a required binary / directory is missing.
# ---------------------------------------------------------------------------

_service_lm_studio_serv() {
    local lms_bin="$USERHOME/.lmstudio/bin/lms"
    if [ ! -x "$lms_bin" ]; then
        echo "  SKIP lm_studio_serv.service — lms not found at $lms_bin"
        return 1
    fi
    cat <<EOF
[Unit]
Description=LM Studio Server
After=network.target graphical.target

[Service]
ExecStart=$lms_bin server start --port 1234
ExecStop=$lms_bin server stop
User=$USER
Environment=HOME=$USERHOME
Environment=XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR
Type=oneshot
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
}

_service_unsloth_studio() {
    local unsloth_bin="$USERHOME/.local/bin/unsloth"
    local workdir="${UNSLOTH_WORKDIR:-$USERHOME/Documents/projects/unsloth_server}"
    if [ ! -x "$unsloth_bin" ]; then
        echo "  SKIP unsloth_studio.service — unsloth not found at $unsloth_bin"
        return 1
    fi
    cat <<EOF
[Unit]
Description=Unsloth Studio Server
After=network.target

[Service]
ExecStart=$unsloth_bin studio -H 0.0.0.0 -p 8899
WorkingDirectory=$workdir
User=$USER
Environment=HF_HOME=$HF_HOME

[Install]
WantedBy=multi-user.target
EOF
}

_service_llama_cpp() {
    local workdir="${LLAMA_CPP_DIR:-$USERHOME/Documents/projects/llama_cpp}"
    local python="$workdir/.venv/bin/python3"
    if [ ! -x "$python" ]; then
        echo "  SKIP llama_cpp.service — venv not found at $workdir/.venv"
        return 1
    fi
    cat <<EOF
[Unit]
Description=llama.cpp HTTP Server
After=network.target

[Service]
ExecStart=$python llama_cpp_service.py
WorkingDirectory=$workdir
User=$USER

[Install]
WantedBy=multi-user.target
EOF
}

# Map service filename (stem without .service) to template function
_template_for() {
    case "$1" in
        lm_studio_serv.service)  echo "_service_lm_studio_serv" ;;
        unsloth_studio.service)  echo "_service_unsloth_studio" ;;
        llama_cpp.service)       echo "_service_llama_cpp" ;;
        *)                       echo "" ;;
    esac
}

# ---------------------------------------------------------------------------
# Read service names from providers.yaml
# ---------------------------------------------------------------------------
SERVICES=()
if [ -f "$PROVIDERS_YAML" ]; then
    while IFS= read -r line; do
        svc=$(echo "$line" | sed -n 's/.*systemd_service:[[:space:]]*//p' | tr -d '"'"'" | xargs)
        [ -n "$svc" ] && SERVICES+=("$svc")
    done < "$PROVIDERS_YAML"
fi

if [ ${#SERVICES[@]} -eq 0 ]; then
    echo "No systemd_service entries found in $PROVIDERS_YAML"
    exit 1
fi

# ---------------------------------------------------------------------------
# Create missing service files (idempotent, skip ollama)
# ---------------------------------------------------------------------------
echo "=== Service files ==="
created=0
for svc in "${SERVICES[@]}"; do
    if [ "$svc" = "ollama.service" ]; then
        echo "SKIP    $svc — managed by the Ollama installer"
        continue
    fi

    target="$SYSTEMD_DIR/$svc"
    if [ -f "$target" ]; then
        echo "EXISTS  $svc"
        continue
    fi

    fn=$(_template_for "$svc")
    if [ -z "$fn" ]; then
        echo "UNKNOWN $svc — no template available, skipping"
        continue
    fi

    # Capture output; if function returns 1 it printed a SKIP notice
    if content=$("$fn" 2>&1); then
        printf '%s\n' "$content" > "$target"
        chmod 644 "$target"
        echo "CREATED $target"
        created=$((created + 1))
    else
        echo "$content"
    fi
done

if [ "$created" -gt 0 ]; then
    systemctl daemon-reload
    echo "daemon-reload done"
fi

# ---------------------------------------------------------------------------
# Polkit rule — passwordless restart for all listed services
# ---------------------------------------------------------------------------
echo ""
echo "=== Polkit rule ==="
echo "Configuring restart permissions for user '$USER':"
for svc in "${SERVICES[@]}"; do echo "  $svc"; done

UNIT_CONDITIONS=""
for svc in "${SERVICES[@]}"; do
    UNIT_CONDITIONS+="        action.lookup(\"unit\") == \"$svc\" ||\n"
done
UNIT_CONDITIONS=$(printf '%b' "$UNIT_CONDITIONS" | sed '$ s/ ||$//')

mkdir -p "$POLKIT_DIR"
tee "$POLKIT_FILE" > /dev/null <<EOF
polkit.addRule(function(action, subject) {
    if (action.id == "org.freedesktop.systemd1.manage-units" &&
        subject.user == "$USER" &&
        (
$UNIT_CONDITIONS
        ) &&
        action.lookup("verb") == "restart") {
        return polkit.Result.YES;
    }
});
EOF

chmod 644 "$POLKIT_FILE"
echo "Written: $POLKIT_FILE"
echo ""
echo "Verify (should succeed without password):"
for svc in "${SERVICES[@]}"; do echo "  systemctl restart $svc"; done
