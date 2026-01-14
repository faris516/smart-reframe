# Smart Reframe (Lib)

Turn your landscape videos into vertical viral gold. Automatically. 🚀

This isn't just a simple center-crop tool. **Smart Reframe** watches your video like a real camera operator. It listens to who is speaking, detects where the action is, and smoothly pans to keep the important stuff in frame.

Ideally suited for repurposing 16:9 content (Youtube, Podcasts) for TikTok, Reels, and Shorts without the manual headache.

![Example Output](assets/full_demo.mp4)

---

## Why use this?
Because manually keyframing vertical videos is painful. 
This library solves the "floating head" problem by:

- **Knowing who is talking**: It combines Face Detection with Audio Activity. If someone on the left starts speaking, the camera moves there.
- **Respecting Space**: It leaves "headroom" and uses asymmetric smoothing (fast zoom out, cinematic slow zoom in) so people don't get cut off.
- **Cinematic Stability**: It feels like a tripod pan, not a robot jitter.
- **Group Intelligence**: If two people are interacting, it widens the shot to keep them both in the conversation.

## Features at a Glance
- 🎙️ **Active Speaker Tracking**: Follows the voice.
- 🎥 **Virtual Tripod**: Smooths out camera shake and mimics professional panning.
- 👥 **Group Awareness**: Automatically zooms out for groups.
- 🔍 **Text Awareness**: Spots text/banners and adjusts framing so you don't chop off captions.
- ⚡ **Auto-Zoom**: Dynamic close-up correction (no more extreme nose-closeups).

## Installation

```bash
pip install smart-reframe
```
*(Or if you are hacking on the source)*
```bash
pip install -e .
```

## How to use it

It's designed to be simple. Just point it at a video file.

```python
from smart_reframe import SmartReframer

# Fire it up
reframer = SmartReframer(
    input_path="./my_podcast.mp4",
    output_path="./viral_short.mp4",
    group_aware=True,   # Enable group shots
    dynamic_zoom=True   # Enable smooth zooms
)

reframer.run()
```

## Under the Hood
We use a robust stack to make this happen:
- **MediaPipe** for seeing faces.
- **CV2 / NumPy** for the heavy lifting.
- **FFmpeg** for rock-solid video encoding.
- **OneEuro Filters** for that buttery smooth motion.

## License
MIT. Go make something cool.

---
*Created with the help of **Vibe Coding**.* ✨
