# docs/installation.md

# Installation Guide

This guide will walk you through the process of installing and setting up the Crypto Algotrading Platform.

## System Requirements

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- Internet connection for fetching market data and executing trades

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/crypto-algotrading-platform.git
cd crypto-algotrading-platform

### 2. Create a Virtual Environment

bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

3. Install Dependencies
bash
pip install -r requirements.txt

4. Configure Exchange API Keys
Create an account on a supported cryptocurrency exchange (Binance, Bybit, etc.)
Generate API keys with trading permissions
Add your API keys to the platform through the Settings tab

5. Run the Application
bash
python src/app.py
The application will start and be accessible at http://127.0.0.1:8050/ in your web browser.

