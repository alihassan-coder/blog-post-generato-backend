#!/usr/bin/env python3
"""
Setup script for Blog Generation Agent
This script helps set up the environment and dependencies
"""

import os
import subprocess
import sys

def create_env_file():
    """Create .env file with template values"""
    env_content = """# Blog Generation Agent - Environment Variables
# Get your API keys from:
# Gemini: https://makersuite.google.com/app/apikey
# Tavily: https://tavily.com

GEMINI_API_KEY=your_gemini_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
MONGODB_URI=mongodb://localhost:27017/blog_agent

# JWT Configuration
SECRET_KEY=your_super_secret_key_here_change_in_production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Optional: Set to 'true' to enable debug logging
DEBUG=false
"""
    
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ Created .env file template")
        print("📝 Please edit .env file with your actual API keys")
        return False
    else:
        print("✅ .env file already exists")
        return True

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_mongodb():
    """Check if MongoDB is running"""
    try:
        import pymongo
        client = pymongo.MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
        client.server_info()
        print("✅ MongoDB is running")
        return True
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        print("💡 Please make sure MongoDB is running on localhost:27017")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up Blog Generation Agent...")
    print("=" * 50)
    
    # Create .env file
    env_exists = create_env_file()
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Check MongoDB
    mongodb_ok = check_mongodb()
    
    print("\n" + "=" * 50)
    if env_exists and mongodb_ok:
        print("🎉 Setup complete! You can now run the application.")
        print("\n📋 Next steps:")
        print("1. Edit .env file with your API keys")
        print("2. Run: python main.py")
    else:
        print("⚠️  Setup incomplete. Please fix the issues above.")
    
    return env_exists and mongodb_ok

if __name__ == "__main__":
    main()
