#!/bin/bash
set -euo pipefail  # Exit on error, undefined variables, and pipe failures

echo ">>>>>>>>>>> Running <<<<<<<<<<<"
uv run python src/routing.py
echo ">>>>>>>>>>> Script completed successfully <<<<<<<<<<<"