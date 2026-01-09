#!/bin/bash
set -euo pipefail  # Exit on error, undefined variables, and pipe failures

echo ">>>>>>>>>>> Running <<<<<<<<<<<"
uv run python src/chain_prompt.py
echo ">>>>>>>>>>> Script completed successfully <<<<<<<<<<<"