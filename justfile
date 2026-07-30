IMAGE := "anomalica-assimilator:development"

# Run the test suite inside the container.
test:
    #!/usr/bin/env bash
    set -euo pipefail
    docker run --rm \
        -v "$(pwd)/workspace:/home/nonroot/workspace" \
        -v "$HOME/repos/anomalica/anomalica-common/src:/opt/anomalica-common:ro" \
        -e PYTHONPATH=/opt/anomalica-common \
        --user "$(id -u):$(id -g)" \
        -w /home/nonroot/workspace \
        {{IMAGE}} \
        python -m pytest

# Run the embedding endpoint (127.0.0.1:8077) in the foreground. The model lives
# only in this container; consumers (the workbench audit view) reach it over
# localhost via anomalica_common.embedding_client. Runs foreground so a
# supervisor (systemd, `install-embed-service`) owns the lifecycle; Ctrl-C stops.
# --rm + a fixed name so a restart can't collide on the profile.
embed-service:
    #!/usr/bin/env bash
    set -euo pipefail
    docker rm -f anomalica-embed >/dev/null 2>&1 || true
    exec docker run --rm --name anomalica-embed --network host \
        -v "$(pwd)/workspace:/home/nonroot/workspace" \
        -v "$HOME/repos/anomalica/anomalica-common/src:/opt/anomalica-common:ro" \
        -v "$HOME/.local/share/assimilator:/data" \
        -e PYTHONPATH=/opt/anomalica-common \
        -e ANOMALICA_TEXT_EMBEDDINGS_DB=/data/text-embeddings.db \
        --user "$(id -u):$(id -g)" \
        -w /home/nonroot/workspace \
        {{IMAGE}} \
        python -m assimilator.embed_service

# Install + start the embedding endpoint as a systemd user service, so it
# survives a reboot and restarts on failure. Idempotent.
install-embed-service:
    #!/usr/bin/env bash
    set -euo pipefail
    unit="$HOME/.config/systemd/user/anomalica-embed.service"
    mkdir -p "$(dirname "$unit")"
    cat > "$unit" <<UNIT
    [Unit]
    Description=Anomalica embedding endpoint (assimilator vector space)
    After=docker.service

    [Service]
    Type=simple
    WorkingDirectory=$(pwd)
    ExecStart=$(command -v just) embed-service
    ExecStop=/usr/bin/docker stop anomalica-embed
    Restart=on-failure
    RestartSec=5

    [Install]
    WantedBy=default.target
    UNIT
    sed -i 's/^    //' "$unit"
    systemctl --user daemon-reload
    systemctl --user enable --now anomalica-embed.service
    echo "started; check: curl -s http://127.0.0.1:8077/health"

# Fold every digest on disk into the graph, then re-apply the curation ledger.
# Deterministic and local - no Claude, no allowance. Safe to run repeatedly: a
# record already in the graph is reconciled by claim_hash rather than duplicated.
assimilate:
    #!/usr/bin/env bash
    set -euo pipefail
    export ANOMALICA_INGESTS_DIR="${ANOMALICA_INGESTS_DIR:-$HOME/repos/anomalica/ingests}"
    digests="${ANOMALICA_DIGESTS_DIR:-$HOME/repos/anomalica/digests}"
    cd workspace
    python3 -m assimilator.cli assimilate "$digests/records"
    python3 -m assimilator.cli replay-curation
    python3 -m assimilator.cli link-works
    python3 -m assimilator.cli propose-pages

# Run `just assimilate` hourly. A STOPGAP: the scheduler generates import jobs
# correctly but /api/schedule surfaces none, so nothing folds new digests into
# the graph on its own and it drifts behind the digester (22 records, once).
# Remove this once import-job dispatch works: `just uninstall-assimilate-timer`.
install-assimilate-timer:
    #!/usr/bin/env bash
    set -euo pipefail
    dir="$HOME/.config/systemd/user"
    mkdir -p "$dir"
    cat > "$dir/anomalica-assimilate.service" <<UNIT
    [Unit]
    Description=Anomalica: fold digests into the knowledge graph (local, no AI)

    [Service]
    Type=oneshot
    WorkingDirectory=$(pwd)
    ExecStart=$(command -v just) assimilate
    UNIT
    cat > "$dir/anomalica-assimilate.timer" <<UNIT
    [Unit]
    Description=Hourly digest import - stopgap while import jobs are not dispatched

    [Timer]
    OnBootSec=10min
    OnUnitActiveSec=1h
    AccuracySec=1min
    Persistent=true

    [Install]
    WantedBy=timers.target
    UNIT
    sed -i 's/^    //' "$dir/anomalica-assimilate.service" "$dir/anomalica-assimilate.timer"
    systemctl --user daemon-reload
    systemctl --user enable --now anomalica-assimilate.timer
    systemctl --user list-timers anomalica-assimilate.timer --no-pager

uninstall-assimilate-timer:
    #!/usr/bin/env bash
    set -euo pipefail
    systemctl --user disable --now anomalica-assimilate.timer 2>/dev/null || true
    rm -f "$HOME/.config/systemd/user/anomalica-assimilate."{service,timer}
    systemctl --user daemon-reload
    echo "timer removed"
