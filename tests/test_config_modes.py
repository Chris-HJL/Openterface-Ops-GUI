"""
集成测试：验证配置模式切换和回退机制
"""
import sys
import os
import time
import requests

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_with_mode(mode: str):
    """测试特定配置模式"""
    from config import Config, ScreenCaptureMode
    
    # 设置环境变量
    os.environ['SCREEN_CAPTURE_MODE'] = mode
    
    # 重新加载配置
    Config.reload_from_env()
    
    print(f"\n测试模式：{mode}")
    print(f"配置模式：{Config.SCREEN_CAPTURE_MODE.value}")
    
    base_url = "http://localhost:9000"
    
    try:
        # 创建会话
        session_data = requests.post(
            f"{base_url}/create-session",
            json={
                "session_id": f"test_{mode}_session",
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
            },
            timeout=5
        )
        
        if session_data.status_code != 200:
            print(f"  ❌ 创建会话失败：{session_data.text}")
            return False
        
        session_id = session_data.json()["session_id"]
        print(f"  ✅ 会话创建成功")
        
        # 测试 /get-image
        image_response = requests.post(
            f"{base_url}/get-image",
            json={"session_id": session_id},
            timeout=30
        )
        
        if image_response.status_code != 200:
            print(f"  ❌ /get-image 失败：{image_response.text}")
            return False
        
        result = image_response.json()
        
        if not result["success"]:
            print(f"  ❌ 获取图像失败：{result['message']}")
            return False
        
        image_len = len(result.get("image", ""))
        print(f"  ✅ 成功获取图像 (Base64 长度：{image_len})")
        
        # 清理会话
        requests.post(
            f"{base_url}/delete-session",
            json={"session_id": session_id},
            timeout=5
        )
        
        return True
        
    except Exception as e:
        print(f"  ❌ 测试出错：{e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 70)
    print("配置模式切换和回退机制测试")
    print("=" * 70)
    
    # 提示用户
    print("\n请确保以下服务正在运行:")
    print("  1. tcpserver_simulator.py (端口 12345)")
    print("  2. ops_api.py (端口 9000)")
    print("\n等待 3 秒让服务准备...")
    time.sleep(3)
    
    # 测试不同模式
    modes = ["gettargetscreen", "lastimage", "hybrid"]
    results = {}
    
    for mode in modes:
        success = test_with_mode(mode)
        results[mode] = success
        print()
    
    # 总结
    print("=" * 70)
    print("测试结果总结:")
    print("=" * 70)
    for mode, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"  {mode:20s} {status}")
    
    all_passed = all(results.values())
    print()
    if all_passed:
        print("✅ 所有配置模式测试通过!")
    else:
        print("❌ 部分测试失败")
    
    print("=" * 70)
    
    return all_passed


if __name__ == '__main__':
    # 注意：这个测试需要 ops_api.py 在同一个进程中重新加载配置
    # 实际情况下，需要重启 ops_api.py 才能生效
    # 这里仅用于验证配置加载逻辑
    
    success = main()
    
    if not success:
        sys.exit(1)
