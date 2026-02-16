# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred.

# Jump to LOME_ROOT
cd $LOME_ROOT

# Update git submodules when present
if [ -f .gitmodules ]; then
    echo "# Updating git submodules"
    git submodule update --init --recursive
fi

# Custom prompt to make it clear this is the Lome environment
PROMPT="[LOME]:$PROMPT"

# Inherit the user history location
export HISTFILE=$USER_HISTFILE

# Incrementally append to history file
setopt INC_APPEND_HISTORY

# Ensure uv environment is installed and eumos is up-to-date
echo "# Checking Python environment is up-to-date"
if [ ! -d ".venv" ] || [ ! -f "uv.lock" ]; then
    uv lock
    uv sync --extra dev
else
    uv sync --extra dev --upgrade-package eumos
fi

# Activate the uv virtual environment
echo "# Activating virtual environment"
export VIRTUAL_ENV_DISABLE_PROMPT=1
source .venv/bin/activate

# Install pre-commit
echo "# Setting up pre-commit hooks"
pre-commit install > /dev/null

# Alias to refresh submodules on demand
alias gsu='git submodule update --init --recursive'
