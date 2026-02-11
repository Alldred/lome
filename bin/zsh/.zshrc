# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred.

# Jump to LOME_ROOT
cd $LOME_ROOT

# Custom prompt to make it clear this is the Lome environment
PROMPT="[LOME]:$PROMPT"

# Inherit the user history location
export HISTFILE=$USER_HISTFILE

# Incrementally append to history file
setopt INC_APPEND_HISTORY

# Ensure uv environment is installed
echo "# Checking Python environment is up-to-date"
if [ ! -d ".venv" ] || [ ! -f "uv.lock" ]; then
    uv lock
    uv sync --extra dev
fi

# Activate the uv virtual environment
echo "# Activating virtual environment"
export VIRTUAL_ENV_DISABLE_PROMPT=1
source .venv/bin/activate

# Install pre-commit
echo "# Setting up pre-commit hooks"
pre-commit install > /dev/null
