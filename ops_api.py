#!/usr/bin/env python3
"""
Openterface Ops API Server (Refactored)
Provides the same functionality as ops_cli.py but via API endpoints
Uses modular architecture with ops_core modules
"""

import uvicorn
import webbrowser
import time
import threading
from ops_api import create_app

if __name__ == "__main__":
    # Automatically open browser after startup
    def open_browser():
        time.sleep(1)
        webbrowser.open("http://localhost:9000/static/index.html")

    # Open browser in a new thread
    threading.Thread(target=open_browser).start()

    # Run server (use import string to support hot reload)
    uvicorn.run(
        "ops_api:create_app",
        host="0.0.0.0",
        port=9000,
        # reload=True,
        log_level="info"
    )
