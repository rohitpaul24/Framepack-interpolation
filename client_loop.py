"""
Simple client script to send videos to the loop generation server.
"""

import requests
import sys
import json
from pathlib import Path


def check_server_health(server_url="http://localhost:8000"):
    """Check if server is ready"""
    try:
        response = requests.get(f"{server_url}/health")
        data = response.json()
        
        if data['status'] == 'ready':
            print(f"✓ Server is ready")
            print(f"  Models loaded: {data['models_loaded']}")
            print(f"  High VRAM mode: {data['high_vram']}")
            print(f"  GPU memory: {data['gpu_memory_gb']:.2f} GB\n")
            return True
        else:
            print(f"✗ Server not ready: {data['message']}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to server at {server_url}")
        print(f"  Make sure the server is running: python server_loop.py")
        return False
    except Exception as e:
        print(f"✗ Error checking server: {e}")
        return False


def process_video(video_path, server_url="http://localhost:8000"):
    """Send video to server for processing"""
    
    if not Path(video_path).exists():
        print(f"✗ Video file not found: {video_path}")
        return None
    
    print(f"Uploading video: {video_path}")
    print("This may take a while depending on video length...\n")
    
    try:
        with open(video_path, 'rb') as f:
            files = {'file': (Path(video_path).name, f, 'video/mp4')}
            response = requests.post(
                f"{server_url}/process_video",
                files=files,
                timeout=600  # 10 minute timeout
            )
        
        if response.status_code == 200:
            result = response.json()
            print("\n" + "="*80)
            print("✓ SUCCESS!")
            print("="*80)
            print(f"\nVideo: {result['video_name']}")
            print(f"Duration: {result['duration']:.2f}s")
            print(f"Resolution: {result['resolution']}")
            print(f"FPS: {result['fps']:.2f}")
            print(f"Loops generated: {result['loops_generated']}\n")
            
            print("Interpolation videos:")
            for path in result['outputs']['interpolations']:
                print(f"  - {path}")
            
            print("\nLoopable videos:")
            for path in result['outputs']['loopable_videos']:
                print(f"  - {path}")
            
            print(f"\nFiles saved in: outputs/")
            
            return result
        else:
            print(f"✗ Server error: {response.status_code}")
            print(response.text)
            return None
            
    except requests.exceptions.Timeout:
        print("✗ Request timed out - video processing took too long")
        return None
    except Exception as e:
        print(f"✗ Error processing video: {e}")
        return None


def download_file(file_url, output_path, server_url="http://localhost:8000"):
    """Download a generated file from the server"""
    try:
        response = requests.get(f"{server_url}{file_url}", stream=True)
        
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✓ Downloaded: {output_path}")
            return True
        else:
            print(f"✗ Download failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error downloading file: {e}")
        return False


if __name__ == "__main__":
    import os
    
    # Get server URL from environment variable or command line
    SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")
    
    # Check if video path is provided
    if len(sys.argv) < 2:
        print("Usage: python client_loop.py <video_path> [server_url]")
        print("\nExamples:")
        print("  # Local server")
        print("  python client_loop.py angry_clip.mp4")
        print()
        print("  # RunPod server")
        print("  python client_loop.py angry_clip.mp4 https://13iowodvjw0osc-8000.proxy.runpod.net")
        print()
        print("  # Or use environment variable")
        print("  export SERVER_URL=https://13iowodvjw0osc-8000.proxy.runpod.net")
        print("  python client_loop.py angry_clip.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    # Override with command line argument if provided
    if len(sys.argv) >= 3:
        SERVER_URL = sys.argv[2]
    
    print("\n" + "="*80)
    print("Video Loop Generation Client")
    print("="*80 + "\n")
    print(f"Server: {SERVER_URL}\n")
    
    # Check server health
    if not check_server_health(SERVER_URL):
        print("\nPlease start the server first:")
        print("  python server_loop.py")
        sys.exit(1)
    
    # Process video
    result = process_video(video_path, SERVER_URL)
    
    if result:
        print("\n" + "="*80)
        print("Processing complete!")
        print("="*80 + "\n")
