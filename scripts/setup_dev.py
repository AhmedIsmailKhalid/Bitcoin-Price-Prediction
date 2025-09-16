#!/usr/bin/env python3
"""Development environment setup script"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Run shell command with error handling"""
    print(f"🔄 {description}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        sys.exit(1)
        
def check_nodejs():
    """Check Node.js version - optional for this project phase"""
    try:
        print("🔄 Checking Node.js version")
        result = subprocess.run(['node', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ Node.js version: {result.stdout.strip()}")
        else:
            print("⚠️  Node.js not found (optional for current phase)")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("⚠️  Node.js not found (optional for current phase)")

def install_nodejs_deps():
    """Install Node.js dependencies - skip if no package.json"""
    try:
        if not os.path.exists('package.json'):
            print("ℹ️  No package.json found - skipping Node.js dependencies")
            return
        
        print("🔄 Installing Node.js dependencies")
        result = subprocess.run(['npm', 'install'], capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("✅ Installing Node.js dependencies completed")
        else:
            print("⚠️  Node.js dependencies not needed for current phase")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("ℹ️  npm not available - skipping Node.js dependencies")

def main():
    """Setup development environment"""
    print("🚀 Setting up Bitcoin Prediction Engine development environment")
    
    # Check prerequisites
    run_command("python --version", "Checking Python version")
    run_command("node --version", "Checking Node.js version")
    run_command("npm --version", "Checking npm version")
    run_command("docker --version", "Checking Docker version")
    
    # Setup Python environment
    run_command("poetry install", "Installing Python dependencies")
    
    # Setup frontend environment
    run_command("cd frontend && npm install", "Installing Node.js dependencies")
    
    # Copy environment template
    if not Path(".env.dev").exists():
        run_command("cp .env.example .env.dev", "Creating development environment file")
        print("📝 Please edit .env.dev with your API keys")
    
    # Start Docker services
    run_command("docker-compose -f docker-compose.dev.yml up -d", "Starting Docker services")
    
    # Wait for services
    print("⏳ Waiting for services to be ready...")
    import time
    time.sleep(30)
    
    # Download historical data
    run_command("poetry run python scripts/setup_historical_data.py", "Downloading historical data")
    
    # Validate setup
    run_command("poetry run python scripts/validate_setup_data.py", "Validating setup data")
    
    print("🎉 Development environment setup completed!")
    print("\n📋 Next steps:")
    print("1. Edit .env.dev with your API keys")
    print("2. Run 'poetry run python src/api/main.py' to start backend")
    print("3. Run 'cd frontend && npm run dev' to start frontend")
    print("4. Visit http://localhost:3000 for the application")

if __name__ == "__main__":
    main()
