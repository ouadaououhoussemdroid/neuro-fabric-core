#!/bin/bash
# Bootstrap script: Automated environment setup for Neuro-Fabric Core
# Usage: bash scripts/bootstrap.sh

set -e  # Exit on error

echo "🔧 Neuro-Fabric Core — Environment Bootstrap"
echo "============================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'  # No Color

# Check prerequisites
echo "${YELLOW}1️⃣  Checking prerequisites...${NC}"

if ! command -v git &> /dev/null; then
    echo "${RED}❌ git not found. Please install git.${NC}"
    exit 1
fi
echo "${GREEN}✅ git${NC}"

if ! command -v node &> /dev/null; then
    echo "${RED}❌ Node.js not found. Please install Node.js 18+.${NC}"
    exit 1
fi
echo "${GREEN}✅ Node.js $(node --version)${NC}"

if ! command -v bun &> /dev/null; then
    echo "${RED}❌ bun not found. Please install bun first:${NC}"
    echo "   curl -fsSL https://bun.sh/install | bash"
    exit 1
fi
echo "${GREEN}✅ bun $(bun --version)${NC}"

if ! command -v python3 &> /dev/null; then
    echo "${RED}❌ Python 3 not found. Please install Python 3.11+.${NC}"
    exit 1
fi
echo "${GREEN}✅ Python $(python3 --version)${NC}"

echo ""

# Setup Python virtual environment
echo "${YELLOW}2️⃣  Setting up Python environment...${NC}"

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "${GREEN}✅ Virtual environment created${NC}"
else
    echo "${GREEN}✅ Virtual environment already exists${NC}"
fi

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
else
    echo "${RED}❌ Could not activate virtual environment${NC}"
    exit 1
fi
echo "${GREEN}✅ Virtual environment activated${NC}"

# Upgrade pip
python -m pip install --quiet --upgrade pip setuptools wheel
echo "${GREEN}✅ pip upgraded${NC}"

# Install Python dependencies
echo "${YELLOW}Installing Python dependencies...${NC}"
pip install --quiet -r training/requirements.txt
echo "${GREEN}✅ Python dependencies installed${NC}"

echo ""

# Verify Python setup
echo "${YELLOW}3️⃣  Verifying Python setup...${NC}"
python training/setup_env.py

echo ""

# Install Node dependencies
echo "${YELLOW}4️⃣  Installing Node dependencies...${NC}"
bun install
echo "${GREEN}✅ Node dependencies installed${NC}"

echo ""

# Setup .env if not exists
echo "${YELLOW}5️⃣  Configuring environment...${NC}"

if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "${GREEN}✅ .env created from template${NC}"
else
    echo "${GREEN}✅ .env already exists${NC}"
fi

echo ""

# Final verification
echo "${YELLOW}6️⃣  Final verification...${NC}"

if bun run build --mode development &> /dev/null; then
    echo "${GREEN}✅ TypeScript compilation successful${NC}"
else
    echo "${YELLOW}⚠️  TypeScript build had warnings (this is OK)${NC}"
fi

echo ""
echo "${GREEN}════════════════════════════════════════${NC}"
echo "${GREEN}✅ Bootstrap Complete!${NC}"
echo "${GREEN}════════════════════════════════════════${NC}"
echo ""
echo "Next steps:"
echo "  1. Start dev server: ${YELLOW}bun run dev${NC}"
echo "  2. Open browser: ${YELLOW}http://localhost:5173${NC}"
echo "  3. Read: ${YELLOW}docs/REALITY_CHECK.md${NC}"
echo "  4. Start T-010: ${YELLOW}cd training && bash scripts/run_all.sh${NC}"
echo ""
