import json
import os
import subprocess
import logging
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_clip(case, output_dir):
    video_id = case['id']
    url = case['source_url']
    start = case['start_time']
    end = case['end_time']
    
    filename = f"{video_id}.mp4"
    output_path = os.path.join(output_dir, filename)
    
    if os.path.exists(output_path):
        logging.info(f"Skipping {video_id} (Already exists)")
        return
        
    logging.info(f"Downloading {video_id} from {url} ({start}-{end})...")
    
    # yt-dlp command for partial download
    # using 'best[height<=1080]' to save bandwidth but keep quality
    cmd = [
        "yt-dlp",
        "--download-sections", f"*{start}-{end}",
        "-f", "best[height<=1080][ext=mp4]/best[ext=mp4]",
        "--force-keyframes-at-cuts",
        "-o", output_path,
        url
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        logging.info(f"Downloaded {video_id}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to download {video_id}: {e.stderr.decode()}")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    manifest_path = os.path.join(base_dir, 'manifest.json')
    data_dir = os.path.join(base_dir, 'data', 'source')
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    with open(manifest_path, 'r') as f:
        cases = json.load(f)
        
    logging.info(f"Fetching {len(cases)} clips...")
    
    # Download in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        for case in cases:
            executor.submit(fetch_clip, case, data_dir)
            
    logging.info("Fetch complete.")

if __name__ == "__main__":
    main()
