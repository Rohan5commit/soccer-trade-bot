#!/bin/bash
# Setup OVH AI Training token
# Run this locally to authenticate with OVH and get a token
# Then add the token as GitHub secret OVH_AI_TOKEN

set -euo pipefail

REGION="${1:-GRA}"

echo "=== OVH AI Training Token Setup ==="
echo "Region: $REGION"
echo ""

# Check if ovhai is installed
if ! command -v ovhai &>/dev/null; then
    echo "Installing ovhai CLI..."
    curl -s "https://cli.gra.ai.cloud.ovh.net/install.sh" | bash
    export PATH="$HOME/.local/bin:$PATH"
fi

# Check if already authenticated
if ovhai token list --output json 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); assert len(d)>0" 2>/dev/null; then
    echo "Already authenticated. Current tokens:"
    ovhai token list
    echo ""
    echo "To create a new token:"
    echo "  ovhai token create --name github-training"
    echo ""
else
    echo "Not authenticated. Starting login..."
    echo ""
    echo "This will open a browser window for authentication."
    echo "Use your OVH AI Training credentials (not your regular OVH login)."
    echo ""
    echo "If you don't have AI Training credentials:"
    echo "  1. Go to https://$REGION.training.ai.cloud.ovh.net"
    echo "  2. Click 'Login'"
    echo "  3. Use your OVHcloud account"
    echo "  4. Accept the AI Training terms"
    echo ""
    
    ovhai config set "$REGION"
    ovhai login
    
    echo ""
    echo "Creating a token for GitHub Actions..."
    ovhai token create --name github-training
fi

echo ""
echo "=== Your AI Tokens ==="
ovhai token list
echo ""
echo "Copy the token value and add it as a GitHub secret:"
echo "  gh secret set OVH_AI_TOKEN"
echo ""
echo "Or manually:"
echo "  Go to https://github.com/Rohan5commit/soccer-trade-bot/settings/secrets/actions"
echo "  Add new repository secret: OVH_AI_TOKEN = <token-value>"
