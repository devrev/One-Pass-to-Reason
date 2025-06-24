#!/bin/bash
set -e

# Check if LLamaFactory directory is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path-to-llamafactory>"
    exit 1
fi

LLAMAFACTORY_PATH=$1
PATCH_FILE="$(pwd)/one-pass-to-reason.patch"

# Check if the directory exists
if [ ! -d "$LLAMAFACTORY_PATH" ]; then
    echo "Error: $LLAMAFACTORY_PATH directory does not exist"
    echo "Please clone LLamaFactory first:"
    echo "git clone https://github.com/hiyouga/LLaMA-Factory.git"
    exit 1
fi

# Apply patch with correct path handling
echo "Applying patches to $LLAMAFACTORY_PATH..."
(cd "$LLAMAFACTORY_PATH" && git apply -p1 "$PATCH_FILE")

echo "Patch applied successfully!"
echo "Your modified version of LLamaFactory is ready to use."