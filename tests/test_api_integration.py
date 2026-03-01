"""
集成测试：验证 API 端点对 gettargetscreen 的支持
"""
import sys
import os
import time
import requests
import json

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_api_with_gettargetscreen():
    """测试 API 端点使用 gettargetscreen 配置"""
    print("=" * 70)
    print("集成测试：API 端点 gettargetscreen 支持")
    print("=" * 70)
    
    base_url = "http://localhost:9000"
    
    # 首先测试 /get-image 端点
    print("\n1. 测试 /get-image 端点...")
    try:
        # 创建会话
        session_data = requests.post(
            f"{base_url}/create-session",
            json={
                "session_id": "test_gettargetscreen_session",
                "api_url": "http://localhost:11434/v1/chat/completions",
                "ui_model_api_url": "http://localhost:2345/v1/chat/completions",
                "model": "qwen3-vl:8b-thinking-q4_K_M",
                "ui_model": "fara-7b",
                "max_react_iterations": 5,
                "llm_use_oemr": False,
                "rag_enabled": False,
                "language": "en",
                "multiturn_enabled": False,
                "scene_type": "general"
            }
        )
        
        if session_data.status_code != 200:
            print(f"   ❌ 创建会话失败：{session_data.text}")
            return False
        
        session_id = session_data.json()["session_id"]
        print(f"   ✅ 会话创建成功：{session_id}")
        
        # 测试 /get-image
        print("   调用 /get-image...")
        image_response = requests.post(
            f"{base_url}/get-image",
            json={"session_id": session_id}
        )
        
        if image_response.status_code != 200:
            print(f"   ❌ /get-image 失败：{image_response.text}")
            return False
        
        result = image_response.json()
        print(f"   /get-image 响应：{result['message']}")
        
        if not result["success"]:
            print(f"   ❌ 获取图像失败：{result['message']}")
            return False
        
        if result.get("image"):
            print(f"   ✅ 成功获取图像 (Base64 长度：{len(result['image'])})")
        else:
            print(f"   ⚠️  图像获取成功但没有返回 Base64 数据")
        
        # 清理会话
        requests.post(
            f"{base_url}/delete-session",
            json={"session_id": session_id}
        )
        
        print("\n" + "=" * 70)
        print("✅ API 端点集成测试通过!")
        print("=" * 70)
        return True
        
    except requests.exceptions.ConnectionError:
        print(f"   ❌ 无法连接到服务器 {base_url}")
        print(f"   请确保 ops_api.py 正在运行")
        return False
    except Exception as e:
        print(f"   ❌ 测试出错：{e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_modes():
    """测试不同的配置模式"""
    print("\n" + "=" * 70)
    print("配置模式测试")
    print("=" * 70)
    
    from config import Config, ScreenCaptureMode
    
    print(f"当前配置模式：{Config.SCREEN_CAPTURE_MODE.value}")
    print(f"屏幕捕获超时：{Config.SCREEN_CAPTURE_TIMEOUT} 秒")
    print(f"记录分辨率：{Config.RECORD_SCREEN_RESOLUTION}")
    
    # 验证配置值
    assert isinstance(Config.SCREEN_CAPTURE_MODE, ScreenCaptureMode)
    assert isinstance(Config.SCREEN_CAPTURE_TIMEOUT, int)
    assert isinstance(Config.RECORD_SCREEN_RESOLUTION, bool)
    
    print("✅ 配置项验证通过")


if __name__ == '__main__':
    # 验证配置
    test_config_modes()
    
    # 提示用户
    print("\n请确保以下服务正在运行:")
    print("  1. tcpserver_simulator.py (端口 12345)")
    print("  2. ops_api.py (端口 9000)")
    print("\n等待 3 秒让服务准备...")
    time.sleep(3)
    
    # 运行 API 测试
    success = test_api_with_gettargetscreen()
    
    if not success:
        sys.exit(1)
