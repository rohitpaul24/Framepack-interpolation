"""
Batch Video Processing Script

This script processes all videos in a directory and saves results for verification.
"""

import requests
import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Configuration
SERVER_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
TEST_DIR = sys.argv[2] if len(sys.argv) > 2 else "./test"
RESULTS_DIR = "./resized_batch_results"

# Processing options
DURATION = 1.0  # Interpolation length in seconds
INCLUDE_LOOPS = True
INCLUDE_TRIMMED = True
INCLUDE_INTERPOLATIONS = True
TIMEOUT = 1800

print("=" * 80)
print("Batch Video Processing")
print("=" * 80)
print(f"Server: {SERVER_URL}")
print(f"Test Directory: {TEST_DIR}")
print(f"Results Directory: {RESULTS_DIR}")
print(f"Duration: {DURATION}s")
print(f"Timeout: {TIMEOUT}s ({TIMEOUT//60} minutes)")
print("=" * 80 + "\n")

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

# Find all video files
video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
video_files = []

for ext in video_extensions:
    video_files.extend(Path(TEST_DIR).glob(f"*{ext}"))
    video_files.extend(Path(TEST_DIR).glob(f"*{ext.upper()}"))

if not video_files:
    print(f"❌ No video files found in {TEST_DIR}")
    sys.exit(1)

print(f"Found {len(video_files)} video(s) to process\n")

# Process each video
results = []
successful = 0
failed = 0

for i, video_path in enumerate(video_files, 1):
    video_name = video_path.name
    print(f"[{i}/{len(video_files)}] Processing: {video_name}")
    
    try:
        # Upload and process video
        with open(video_path, 'rb') as f:
            files = {'file': f}
            data = {
                'duration': DURATION,
                'include_loops': INCLUDE_LOOPS,
                'include_trimmed': INCLUDE_TRIMMED,
                'include_interpolations': INCLUDE_INTERPOLATIONS
            }
            
            response = requests.post(
                f"{SERVER_URL}/process_video",
                files=files,
                data=data,
                timeout=TIMEOUT
            )
        
        if response.status_code == 200:
            result = response.json()
            
            # Save result
            result_file = os.path.join(RESULTS_DIR, f"{video_path.stem}_result.json")
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"  ✓ Success")
            print(f"    - Duration: {result['duration']:.2f}s")
            print(f"    - Resolution: {result['resolution']}")
            print(f"    - Loops: {result['loops_generated']}")
            print(f"    - Interpolations: {len(result['outputs'].get('interpolations', []))}")
            print(f"    - Result saved: {result_file}")
            
            results.append({
                'video': video_name,
                'status': 'success',
                'result': result
            })
            successful += 1
            
        else:
            error = response.json()
            print(f"  ✗ Failed: {error.get('message', 'Unknown error')}")
            
            results.append({
                'video': video_name,
                'status': 'failed',
                'error': error
            })
            failed += 1
            
    except requests.exceptions.Timeout:
        print(f"  ✗ Timeout")
        results.append({
            'video': video_name,
            'status': 'timeout',
            'error': 'Request timed out'
        })
        failed += 1
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        results.append({
            'video': video_name,
            'status': 'error',
            'error': str(e)
        })
        failed += 1
    
    print()

# Save summary
summary_file = os.path.join(RESULTS_DIR, f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
summary = {
    'timestamp': datetime.now().isoformat(),
    'server': SERVER_URL,
    'test_directory': TEST_DIR,
    'total_videos': len(video_files),
    'successful': successful,
    'failed': failed,
    'settings': {
        'duration': DURATION,
        'include_loops': INCLUDE_LOOPS,
        'include_trimmed': INCLUDE_TRIMMED,
        'include_interpolations': INCLUDE_INTERPOLATIONS
    },
    'results': results
}

with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

# Print summary
print("=" * 80)
print("Summary")
print("=" * 80)
print(f"Total videos: {len(video_files)}")
print(f"Successful: {successful}")
print(f"Failed: {failed}")
print(f"\nSummary saved: {summary_file}")
print("=" * 80)

# Print failed videos if any
if failed > 0:
    print("\nFailed videos:")
    for result in results:
        if result['status'] != 'success':
            print(f"  - {result['video']}: {result.get('error', 'Unknown error')}")

# Exit code
sys.exit(0 if failed == 0 else 1)
