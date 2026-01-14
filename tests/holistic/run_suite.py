import json
import os
import sys
import logging
from concurrent.futures import ProcessPoolExecutor # Use ProcessPool for heavy CPU tasks

# Ensure we can import smart_reframe
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from smart_reframe import SmartReframer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_case(case, source_dir, result_dir):
    video_id = case['id']
    input_path = os.path.join(source_dir, f"{video_id}.mp4")
    output_path = os.path.join(result_dir, f"{video_id}_reframed.mp4")
    
    if not os.path.exists(input_path):
        logging.warning(f"Source missing for {video_id}, skipping.")
        return
        
    logging.info(f"Processing {video_id}...")
    
    # Defaults
    config = {
        "group_aware": True,
        "dynamic_zoom": True,
        "stabilizer_type": "one_euro"
    }
    
    # Overrides
    if 'config_overrides' in case:
        config.update(case['config_overrides'])
        
    try:
        reframer = SmartReframer(
            input_path=input_path,
            output_path=output_path,
            group_aware=config['group_aware'],
            dynamic_zoom=config['dynamic_zoom'],
            stabilizer_type=config['stabilizer_type']
        )
        reframer.run()
        logging.info(f"Finished {video_id}")
    except Exception as e:
        logging.error(f"Error processing {video_id}: {e}")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    manifest_path = os.path.join(base_dir, 'manifest.json')
    source_dir = os.path.join(base_dir, 'data', 'source')
    result_dir = os.path.join(base_dir, 'data', 'results')
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        
    with open(manifest_path, 'r') as f:
        cases = json.load(f)
        
    logging.info(f"Running suite on {len(cases)} cases...")
    
    # Sequential for safety/debug, or Parallel if confident
    # Using Sequential for now to avoid massive CPU contention on local machine
    # unless user has powerful machine. Defaulting to safe sequential.
    for case in cases:
        process_case(case, source_dir, result_dir)
            
    logging.info("Suite run complete.")

if __name__ == "__main__":
    main()
