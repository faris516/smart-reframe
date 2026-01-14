import json
import os
import glob

def generate_html(cases, report_dir):
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Smart Reframe Holistic Test Report</title>
        <style>
            body { font-family: sans-serif; background: #1a1a1a; color: #eee; margin: 0; padding: 20px; }
            .container { max_width: 1200px; margin: 0 auto; }
            .case { background: #2a2a2a; margin-bottom: 30px; padding: 20px; border-radius: 8px; }
            .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
            .tags span { background: #444; padding: 2px 8px; border-radius: 4px; font-size: 0.8em; margin-right: 5px; }
            .videos { display: flex; gap: 20px; justify-content: center; }
            .video-box { text-align: center; }
            video { max-height: 400px; max-width: 100%; border-radius: 4px; background: #000; }
            h2 { margin: 0; font-size: 1.2em; }
            p { color: #aaa; margin: 5px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Holistic Test Report</h1>
            <p>Generated automated comparison of test scenarios.</p>
    """
    
    for case in cases:
        vid_id = case['id']
        src_rel = f"../data/source/{vid_id}.mp4"
        res_rel = f"../data/results/{vid_id}_reframed.mp4"
        
        tags_html = "".join([f"<span>{t}</span>" for t in case.get('tags', [])])
        
        html_content += f"""
        <div class="case">
            <div class="header">
                <div>
                    <h2>{vid_id}</h2>
                    <p>{case.get('description', '')}</p>
                </div>
                <div class="tags">{tags_html}</div>
            </div>
            <div class="videos">
                <div class="video-box">
                    <h3>Source (Landscape)</h3>
                    <video src="{src_rel}" controls muted loop></video>
                </div>
                <div class="video-box">
                    <h3>Result (9:16)</h3>
                    <video src="{res_rel}" controls muted loop autoplay></video>
                </div>
            </div>
        </div>
        """
        
    html_content += """
        </div>
    </body>
    </html>
    """
    
    with open(os.path.join(report_dir, 'index.html'), 'w') as f:
        f.write(html_content)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    manifest_path = os.path.join(base_dir, 'manifest.json')
    report_dir = os.path.join(base_dir, 'report')
    
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
        
    with open(manifest_path, 'r') as f:
        cases = json.load(f)
        
    generate_html(cases, report_dir)
    print(f"Report generated at {os.path.join(report_dir, 'index.html')}")

if __name__ == "__main__":
    main()
