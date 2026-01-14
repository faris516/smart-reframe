import sys
import os

# Add parent to path to allow import without install
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smart_reframe import SmartReframer

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir))) # root/smart_reframe_lib/examples -> root
    
    # Actually, smart_reframe_lib is inside root?
    # /Desktop/autoreframe/smart_reframe_lib/examples/demo.py
    # dirname -> examples
    # dirname -> smart_reframe_lib
    # dirname -> autoreframe
    
    video_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "temp_batch")
    
    if len(sys.argv) > 1:
        input_video = sys.argv[1]
    else:
        # Default to previous behavior
        input_video = os.path.join(video_dir, "6w0rHWNxneA.mp4")
        if not os.path.exists(input_video):
            import glob
            mps = glob.glob(os.path.join(video_dir, "*.mp4"))
            if mps:
                input_video = mps[0]
            
    output_video = "demo_output.mp4"
    
    if not os.path.exists(input_video):
        print(f"Error: Input file found: {input_video}")
        return

    print(f"Running Smart Reframe on {input_video}...")
    
    try:
        reframer = SmartReframer(input_video, output_video, dynamic_zoom=True)
        # Limit processing for test speed (optional, if API supports it)
        # Process full duration
        reframer.run() 
        print(f"Success! Output saved to {output_video}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
