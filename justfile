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
