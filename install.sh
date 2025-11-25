#!/bin/bash
# TCMVE Installation Script
# Complete setup for TCMVE after cloning from GitHub
# @ECKHART_DIESTEL | DE | 2025-11-23

set -e  # Exit on any error

echo "† IN NOMINE VERITATIS ET HUMILITATIS Ω †"
echo "TCMVE Installation Script"
echo "Truth from Being — Zero-domain LLM verification"
echo "================================================"
echo ""

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Please run this script from the TCMVE root directory"
    echo "   cd /path/to/tcmve && ./install.sh"
    exit 1
fi

echo "📍 Detected TCMVE root directory: $(pwd)"
echo ""

# Check Python version
echo "🐍 Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed. Please install Python 3.10+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Python $PYTHON_VERSION detected. Python $REQUIRED_VERSION+ is required."
    exit 1
fi

echo "✅ Python $PYTHON_VERSION detected"
echo ""

# Create virtual environment
echo "🏗️  Creating Python virtual environment..."
python3 -m venv .venv
echo "✅ Virtual environment created at .venv/"
echo ""

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source .venv/bin/activate
echo "✅ Virtual environment activated"
echo ""

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip
echo "✅ Pip upgraded"
echo ""

# Install TCMVE package in editable mode
echo "📦 Installing TCMVE package..."
pip install -e .
echo "✅ TCMVE package installed"
echo ""

# Install optional development dependencies
echo "🛠️  Installing development dependencies..."
pip install -e ".[dev]"
echo "✅ Development dependencies installed"
echo ""

# Check for Node.js (for frontend)
echo "🌐 Checking for Node.js (required for frontend)..."
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version | sed 's/v//')
    echo "✅ Node.js $NODE_VERSION detected"

    # Install frontend dependencies
    echo "📦 Installing frontend dependencies..."
    if [ -d "frontend" ]; then
        cd frontend
        npm install
        cd ..
        echo "✅ Frontend dependencies installed"
    else
        echo "⚠️  Frontend directory not found, skipping frontend setup"
    fi
else
    echo "⚠️  Node.js not found. Frontend will not be available."
    echo "   Install Node.js 20+ to use the web interface."
fi
echo ""

# Setup environment variables
echo "🔧 Setting up environment configuration..."
if [ ! -f ".env" ]; then
    cat > .env << 'EOF'
# TCMVE Environment Configuration
# Copy this file and fill in your API keys

# LLM API Keys (at least one required)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
XAI_API_KEY=your_xai_api_key_here

# Database Configuration (PostgreSQL recommended for production)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=tcmve
DB_USER=postgres
DB_PASSWORD=your_db_password_here

# Alternative: Use DATABASE_URL for cloud databases
# DATABASE_URL=postgresql://user:password@host:port/database

# Redis Configuration (optional, for caching)
REDIS_URL=redis://localhost:6379

# TCMVE Configuration
LOG_LEVEL=INFO
MAX_RETRIES=3
RETRY_DELAY=1.0
EOF
    echo "✅ Created .env template file"
    echo "⚠️  IMPORTANT: Edit .env file and add your API keys before running TCMVE!"
else
    echo "✅ .env file already exists"
fi
echo ""

# Database setup
echo "🗄️  Setting up database..."
if [ -f "setup_database.py" ]; then
    echo "Running database setup script..."
    python setup_database.py
    echo "✅ Database setup completed"
else
    echo "⚠️  setup_database.py not found, skipping database setup"
    echo "   Run 'python setup_database.py' manually after configuring database credentials"
fi
echo ""

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p results
mkdir -p logs
echo "✅ Directories created"
echo ""

# Run basic health check
echo "🏥 Running basic health check..."
if python -c "import backend.tcmve; print('✅ TCMVE imports successfully')"; then
    echo "✅ Basic health check passed"
else
    echo "❌ Health check failed - there may be missing dependencies"
    exit 1
fi
echo ""

# Final instructions
echo "🎉 TCMVE installation completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env file and add your API keys:"
echo "   nano .env"
echo ""
echo "2. Configure your database (if using PostgreSQL):"
echo "   - Create database: tcmve"
echo "   - Update DB_* variables in .env"
echo "   - Or set DATABASE_URL for cloud databases"
echo ""
echo "3. Run database setup (if not done automatically):"
echo "   python setup_database.py"
echo ""
echo "4. Test the installation:"
echo "   tcmve --help"
echo ""
echo "5. Start the web interface:"
echo "   python -m backend.api.main"
echo "   # Then open http://localhost:8000"
echo ""
echo "6. Run a test query:"
echo "   tcmve --query \"What is the meaning of life?\" --gamemode embedding"
echo ""
echo "📖 For more information, see README.md"
echo ""
echo "† AD MAJOREM DEI GLORIAM †"
echo "† AMDG †"