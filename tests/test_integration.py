"""
集成测试：测试与真实模拟器的交互
"""
import sys
import os
import time

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ops_core.image_server.client import ImageServerClient


def test_gettargetscreen_with_simulator():
    """测试 gettargetscreen 命令与模拟器交互"""
    print("=" * 60)
    print("集成测试：gettargetscreen 命令")
    print("=" * 60)
    
    client = ImageServerClient(
        host="localhost",
        port=12345,
        timeout=10
    )
    
    # 测试 gettargetscreen
    print("\n1. 测试 get_target_screen()...")
    result = client.get_target_screen()
    print(f"   结果：{result}")
    assert not result.startswith("Error:"), f"get_target_screen failed: {result}"
    assert os.path.exists(result), f"Image file not created: {result}"
    print(f"   ✅ 成功获取图像：{result}")
    
    # 测试 lastimage
    print("\n2. 测试 get_last_image()...")
    result = client.get_last_image()
    print(f"   结果：{result}")
    assert not result.startswith("Error:"), f"get_last_image failed: {result}"
    assert os.path.exists(result), f"Image file not created: {result}"
    print(f"   ✅ 成功获取图像：{result}")
    
    # 测试智能方法 (优先 gettargetscreen)
    print("\n3. 测试 get_screen_image() - 优先 gettargetscreen...")
    result = client.get_screen_image(
        primary_command="gettargetscreen",
        fallback_command="lastimage"
    )
    print(f"   结果：{result}")
    assert not result.startswith("Error:"), f"get_screen_image failed: {result}"
    assert os.path.exists(result), f"Image file not created: {result}"
    print(f"   ✅ 成功获取图像：{result}")
    
    # 测试智能方法 (优先 lastimage)
    print("\n4. 测试 get_screen_image() - 优先 lastimage...")
    result = client.get_screen_image(
        primary_command="lastimage",
        fallback_command="gettargetscreen"
    )
    print(f"   结果：{result}")
    assert not result.startswith("Error:"), f"get_screen_image failed: {result}"
    assert os.path.exists(result), f"Image file not created: {result}"
    print(f"   ✅ 成功获取图像：{result}")
    
    print("\n" + "=" * 60)
    print("✅ 所有集成测试通过!")
    print("=" * 60)


if __name__ == '__main__':
    print("请确保 tcpserver_simulator.py 正在运行...")
    print("等待 2 秒让服务器准备...")
    time.sleep(2)
    
    try:
        test_gettargetscreen_with_simulator()
    except AssertionError as e:
        print(f"\n❌ 测试失败：{e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试出错：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
