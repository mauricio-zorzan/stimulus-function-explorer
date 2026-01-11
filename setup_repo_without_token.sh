#!/bin/bash
# Setup script to clone and sync the coach-bot repository without using a token
# Uses git credentials (SSH keys or cached HTTPS credentials)

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

REPO_URL="https://github.com/trilogy-group/coach-bot-external-content-generators.git"
REPO_DIR="coach-bot-repo"

echo "🔗 Setting up coach-bot repository for automatic syncing..."
echo ""

# Check if already exists
if [ -d "$REPO_DIR" ]; then
    echo "⚠️  $REPO_DIR already exists"
    read -p "Remove and re-clone? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$REPO_DIR"
    else
        echo "Keeping existing repository"
        echo "To sync: cd $REPO_DIR && git pull"
        exit 0
    fi
fi

# Try cloning
echo "📥 Cloning repository..."
echo "   This will use your git credentials (SSH keys or cached HTTPS)"
echo ""

if git clone "$REPO_URL" "$REPO_DIR" 2>&1; then
    echo ""
    echo "✅ Repository cloned successfully!"
    echo ""
    echo "📋 Repository location: $REPO_DIR"
    echo "   Functions location: $REPO_DIR/src/content_generators/additional_content/stimulus_image/drawing_functions"
    echo ""
    echo "🔄 To sync updates, run:"
    echo "   cd $REPO_DIR && git pull"
    echo ""
    echo "Or use the sync script:"
    echo "   ./sync_coach_bot.sh"
    echo ""
    echo "The code will now check this local repository for functions!"
else
    echo ""
    echo "❌ Failed to clone repository"
    echo ""
    echo "Possible reasons:"
    echo "1. Repository is private and requires authentication"
    echo "2. No git credentials configured (SSH keys or HTTPS cache)"
    echo ""
    echo "Solutions:"
    echo "1. Set up SSH keys: https://docs.github.com/en/authentication/connecting-to-github-with-ssh"
    echo "2. Or use a GitHub token in .env file"
    echo "3. Or manually clone the repository and point the code to it"
    exit 1
fi
