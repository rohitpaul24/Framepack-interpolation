"""
API Test Script

This script tests all endpoints of the Video Loop Generation API.
"""

import requests
import sys
import json
from pathlib import Path

# Server URL
SERVER_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

print("=" * 80)
print("Video Loop Generation API - Test Suite")
print("=" * 80)
print(f"Server: {SERVER_URL}\n")


def test_health():
    """Test /health endpoint"""
    print("1. Testing /health endpoint...")
    try:
        response = requests.get(f"{SERVER_URL}/health")
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ Server is ready")
            print(f"   - Models loaded: {data['models_loaded']}")
            print(f"   - High VRAM: {data['high_vram']}")
            print(f"   - GPU memory: {data['gpu_memory_gb']:.2f} GB")
            print(f"   - Cached prompts: {data['cached_prompts']}")
            return True
        else:
            print(f"   ✗ Server not ready: {response.json()}")
            return False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False


def test_get_config():
    """Test GET /config endpoint"""
    print("\n2. Testing GET /config endpoint...")
    try:
        response = requests.get(f"{SERVER_URL}/config")
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            config = response.json()
            print(f"   ✓ Configuration retrieved")
            print(f"   - Prompt: '{config['prompt']}'")
            print(f"   - Steps: {config['steps']}")
            print(f"   - Total length: {config['total_second_length']}s")
            print(f"   - Latent window: {config['latent_window_size']}")
            return config
        else:
            print(f"   ✗ Failed: {response.json()}")
            return None
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return None


def test_update_config():
    """Test POST /config endpoint"""
    print("\n3. Testing POST /config endpoint...")
    try:
        # Update some config values
        update_data = {
            "steps": 12,
            "total_second_length": 0.6
        }
        
        response = requests.post(
            f"{SERVER_URL}/config",
            json=update_data
        )
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ Configuration updated")
            print(f"   - Updated fields: {result['updated_fields']}")
            
            # Verify the update
            verify = requests.get(f"{SERVER_URL}/config").json()
            print(f"   - Verified steps: {verify['steps']}")
            print(f"   - Verified length: {verify['total_second_length']}s")
            
            # Restore original values
            restore_data = {
                "steps": 10,
                "total_second_length": 0.5
            }
            requests.post(f"{SERVER_URL}/config", json=restore_data)
            print(f"   ✓ Configuration restored")
            return True
        else:
            print(f"   ✗ Failed: {response.json()}")
            return False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False


def test_process_video_basic(video_path):
    """Test POST /process_video with basic parameters"""
    print("\n4. Testing POST /process_video (basic)...")
    
    if not Path(video_path).exists():
        print(f"   ✗ Video file not found: {video_path}")
        return False
    
    try:
        with open(video_path, 'rb') as f:
            files = {'file': f}
            
            print(f"   Uploading: {Path(video_path).name}")
            response = requests.post(
                f"{SERVER_URL}/process_video",
                files=files,
                timeout=600
            )
        
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ Video processed successfully")
            print(f"   - Video name: {result['video_name']}")
            print(f"   - Duration: {result['duration']:.2f}s")
            print(f"   - Resolution: {result['resolution']}")
            print(f"   - FPS: {result['fps']}")
            print(f"   - Loops generated: {result['loops_generated']}")
            print(f"   - Interpolations: {len(result['outputs'].get('interpolations', []))}")
            print(f"   - Loopable videos: {len(result['outputs'].get('loopable_videos', []))}")
            return result
        else:
            error = response.json()
            print(f"   ✗ Failed: {error.get('message', 'Unknown error')}")
            if 'details' in error:
                print(f"   - Details: {error['details']}")
            return None
    except requests.exceptions.Timeout:
        print(f"   ✗ Request timed out")
        return None
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return None


def test_process_video_with_duration(video_path):
    """Test POST /process_video with duration parameter"""
    print("\n5. Testing POST /process_video (with duration=2)...")
    
    if not Path(video_path).exists():
        print(f"   ✗ Video file not found: {video_path}")
        return False
    
    try:
        with open(video_path, 'rb') as f:
            files = {'file': f}
            data = {'duration': 2}
            
            print(f"   Uploading: {Path(video_path).name} (first 2 seconds)")
            response = requests.post(
                f"{SERVER_URL}/process_video",
                files=files,
                data=data,
                timeout=600
            )
        
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ Video processed successfully")
            print(f"   - Loops generated: {result['loops_generated']}")
            print(f"   - Should be 1 loop (for 2 seconds)")
            return result
        else:
            error = response.json()
            print(f"   ✗ Failed: {error.get('message', 'Unknown error')}")
            return None
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return None


def test_process_video_with_options(video_path):
    """Test POST /process_video with all output options"""
    print("\n6. Testing POST /process_video (with all outputs)...")
    
    if not Path(video_path).exists():
        print(f"   ✗ Video file not found: {video_path}")
        return False
    
    try:
        with open(video_path, 'rb') as f:
            files = {'file': f}
            data = {
                'duration': 2,
                'include_loops': True,
                'include_trimmed': True,
                'include_interpolations': True
            }
            
            print(f"   Uploading: {Path(video_path).name} (all outputs)")
            response = requests.post(
                f"{SERVER_URL}/process_video",
                files=files,
                data=data,
                timeout=600
            )
        
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ Video processed successfully")
            outputs = result['outputs']
            print(f"   - Interpolations: {len(outputs.get('interpolations', []))}")
            print(f"   - Loopable videos: {len(outputs.get('loopable_videos', []))}")
            print(f"   - Trimmed videos: {len(outputs.get('trimmed_videos', []))}")
            return result
        else:
            error = response.json()
            print(f"   ✗ Failed: {error.get('message', 'Unknown error')}")
            return None
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return None


def test_download_file(file_url):
    """Test GET /outputs/{video_name}/{folder}/{filename}"""
    print("\n7. Testing file download...")
    try:
        response = requests.get(f"{SERVER_URL}{file_url}", stream=True)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            size = len(response.content)
            print(f"   ✓ File downloaded successfully")
            print(f"   - Size: {size / 1024:.2f} KB")
            return True
        else:
            print(f"   ✗ Failed to download file")
            return False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False


def test_error_handling():
    """Test error handling"""
    print("\n8. Testing error handling...")
    
    # Test invalid duration
    print("   a) Testing invalid duration...")
    try:
        response = requests.post(
            f"{SERVER_URL}/process_video",
            files={'file': ('test.mp4', b'fake data')},
            data={'duration': 100}  # Invalid: > 60
        )
        
        if response.status_code == 400:
            error = response.json()
            print(f"      ✓ Correctly rejected invalid duration")
            print(f"      - Error code: {error['error_code']}")
        else:
            print(f"      ✗ Should have rejected invalid duration")
    except Exception as e:
        print(f"      ✗ Error: {e}")
    
    # Test invalid config update
    print("   b) Testing invalid config update...")
    try:
        response = requests.post(
            f"{SERVER_URL}/config",
            json={'steps': 200}  # Invalid: > 100
        )
        
        if response.status_code == 422:  # Pydantic validation error
            print(f"      ✓ Correctly rejected invalid config")
        else:
            print(f"      ✗ Should have rejected invalid config")
    except Exception as e:
        print(f"      ✗ Error: {e}")


def main():
    """Run all tests"""
    video_path = sys.argv[2] if len(sys.argv) > 2 else "test_video.mp4"
    
    # Run tests
    results = []
    
    # 1. Health check
    results.append(("Health Check", test_health()))
    
    if not results[0][1]:
        print("\n✗ Server not ready. Stopping tests.")
        return
    
    # 2-3. Config tests
    results.append(("Get Config", test_get_config() is not None))
    results.append(("Update Config", test_update_config()))
    
    # 4-6. Video processing tests (only if video file provided)
    if Path(video_path).exists():
        result = test_process_video_basic(video_path)
        results.append(("Process Video (Basic)", result is not None))
        
        # Test download if we got a result
        if result and result['outputs'].get('interpolations'):
            file_url = result['outputs']['interpolations'][0]
            results.append(("Download File", test_download_file(file_url)))
        
        # Note: Skipping duration and options tests to save time
        # Uncomment to test:
        # results.append(("Process Video (Duration)", test_process_video_with_duration(video_path) is not None))
        # results.append(("Process Video (Options)", test_process_video_with_options(video_path) is not None))
    else:
        print(f"\n⚠ Video file not found: {video_path}")
        print("   Skipping video processing tests")
    
    # 8. Error handling
    test_error_handling()
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠ {total - passed} test(s) failed")


if __name__ == "__main__":
    main()
