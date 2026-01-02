#!/bin/bash

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CLUSTER_NAME_FILE="${SCRIPT_DIR}/../.cluster_name"

# Check if cluster name file exists
if [ ! -f "$CLUSTER_NAME_FILE" ]; then
    echo "Error: .cluster_name file not found. Have you launched a cluster?"
    exit 1
fi

CLUSTER_NAME=$(cat "$CLUSTER_NAME_FILE")

if [ -z "$CLUSTER_NAME" ]; then
    echo "Error: Cluster name is empty."
    exit 1
fi

echo "Syncing code to cluster: $CLUSTER_NAME..."
# Use --workdir to explicitly sync the project root
sky exec "$CLUSTER_NAME" --workdir "$PROJECT_ROOT" "echo 'Code synced successfully.'"

if [ $? -eq 0 ]; then
    echo "Sync complete."
else
    echo "Error: Sync failed."
    exit 1
fi
