#!/bin/bash
# Sync script to update the coach-bot repository
# Works for both git submodules and regular git clones

set -e

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

REPO_DIR="coach-bot-repo"

if [ ! -d "$REPO_DIR" ]; then
    echo "❌ $REPO_DIR not found"
    echo "Run ./setup_repo_without_token.sh first"
    exit 1
fi

echo "🔄 Syncing coach-bot repository..."

cd "$REPO_DIR"

# Check if it's a git repository
if [ -d ".git" ]; then
    # Check if it's a submodule
    if [ -f "../.git/modules/$REPO_DIR/config" ]; then
        echo "   Detected as git submodule"
        cd ..
        git submodule update --remote "$REPO_DIR"
    else
        echo "   Detected as regular git repository"
        git pull origin main || git pull origin master
    fi
    echo ""
    echo "✅ Coach-bot repository synced to latest version!"
else
    echo "❌ Not a git repository"
    echo "   Cannot sync automatically"
    exit 1
fi

cd ..

echo ""
echo "The auto-generation will now check both:"
echo "  - Local drawing_functions/ folder"
echo "  - $REPO_DIR/.../drawing_functions/ folder"
