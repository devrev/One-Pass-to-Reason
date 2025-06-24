#!/bin/bash
set -e

# Create a directory for LLamaFactory
mkdir -p llamafactory
cd llamafactory

# Clone LLamaFactory
echo "Cloning LLamaFactory..."
git clone https://github.com/hiyouga/LLaMA-Factory.git .

# Checkout the specific version you used
git checkout a9211a730eb3fc7fe0d008107a0a135c3a8734d8

# Return to the root directory
cd ..

# Apply the patches
echo "Applying our modifications..."
bash apply_patches.sh $(pwd)/llamafactory
