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
    # 启动后自动打开浏览器
    def open_browser():
        time.sleep(1)
        webbrowser.open("http://localhost:9000/static/index.html")

    # 在新线程中打开浏览器
    threading.Thread(target=open_browser).start()

    # 运行服务器（使用导入字符串以支持热重载）
    uvicorn.run(
        "ops_api:create_app",
        host="0.0.0.0",
        port=9000,
        # reload=True,
        log_level="info"
    )
