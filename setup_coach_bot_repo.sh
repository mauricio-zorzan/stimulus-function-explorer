#!/bin/bash
# Setup script to link the coach-bot repository as a submodule

set -e

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if .env file exists and get token
if [ -f .env ]; then
    GITHUB_TOKEN=$(grep "^GITHUB_TOKEN=" .env | cut -d '=' -f2 | tr -d '"' | tr -d "'" | xargs)
else
    echo "⚠️  .env file not found"
    echo "Please set GITHUB_TOKEN in .env file first"
    exit 1
fi

if [ -z "$GITHUB_TOKEN" ]; then
    echo "⚠️  GITHUB_TOKEN not found in .env file"
    echo "Please add GITHUB_TOKEN=your_token to .env file"
    exit 1
fi

echo "🔗 Setting up coach-bot repository as submodule..."

# Remove existing submodule if it exists
if [ -d "coach-bot-repo" ]; then
    echo "⚠️  coach-bot-repo directory already exists"
    read -p "Remove and re-add? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git submodule deinit -f coach-bot-repo 2>/dev/null || true
        rm -rf coach-bot-repo
        git rm -f coach-bot-repo 2>/dev/null || true
    else
        echo "Keeping existing submodule"
        exit 0
    fi
fi

# Add submodule with token in URL
REPO_URL="https://${GITHUB_TOKEN}@github.com/trilogy-group/coach-bot-external-content-generators.git"

echo "📥 Cloning repository..."
git submodule add "$REPO_URL" coach-bot-repo

# Initialize and update
echo "🔄 Initializing submodule..."
git submodule update --init --recursive

echo "✅ Coach-bot repository linked successfully!"
echo ""
echo "To sync updates in the future, run:"
echo "  git submodule update --remote coach-bot-repo"
echo ""
echo "Or use the sync script:"
echo "  ./sync_coach_bot.sh"
