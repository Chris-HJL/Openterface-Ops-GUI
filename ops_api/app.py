"""
FastAPI应用配置
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from .endpoints import router

def create_app() -> FastAPI:
    """
    创建FastAPI应用

    Returns:
        FastAPI应用实例
    """
    # 创建FastAPI应用
    app = FastAPI(title="Openterface Ops API", version="2.0.0")

    # 配置CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 开发环境允许所有来源
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 挂载静态文件
    app.mount("/static", StaticFiles(directory=".", html=True), name="static")

    # 注册路由
    app.include_router(router)

    return app
