# Part 5: Media Elements in Streamlit

This section covers displaying images, videos, audio, and logos in Streamlit applications.

## Overview

Streamlit provides built-in support for various media types, making it easy to embed rich visual content in your data applications. This is particularly useful for multimedia dashboards, image analysis apps, and content-rich presentations.

## Files in This Section

- **[app.py](Part_5_Media/app.py)** - Main application demonstrating all media elements
- **[p5_media.py](Part_5_Media/p5_media.py)** - Tutorial script with detailed comments explaining each element

## Key Concepts Covered

### 1. st.logo()

Displays a logo in the app's header area.

```python
logo_link = "https://cdn.simpleicons.org/youtube"
st.logo(
    logo_link,
    link="https://streamlit.io/gallery",
)
```

**Parameters:**
- `image` - URL or path to the logo image
- `link` - External URL to navigate when clicked (optional)
- `icon_image` - Alternate icon for sidebar display

**Note:** Streamlit scales images to 24px height with max 240px width. Use 10:1 aspect ratio or less.

---

### 2. st.image()

Displays images in various formats.

```python
image_url = "https://images.unsplash.com/photo-1705900266125-3b999d5a5c62"
st.image(
    image_url,
    caption="Twilight Whisper over Mount Cook of New Zealand",
    use_column_width=True,
)
```

**Parameters:**
- `caption` - Text displayed below the image
- `use_column_width` - If True, expands to fill column width
- `width` - Specific width in pixels (overrides use_column_width)
- `output_format` - 'PNG', 'JPEG', or 'auto'

**Supported formats:** URL, local file path, PIL Image, numpy array, bytes

---

### 3. st.video()

Embeds a video player.

```python
video_url = "https://sample-videos.com/video321/mp4/720/big_buck_bunny_720p_2mb.mp4"
st.video(video_url, start_time=3)
```

**Parameters:**
- `data` - Video URL, file path, or bytes
- `start_time` - Start position in seconds
- `end_time` - End position in seconds (optional)

**Supported formats:** MP4, WebM, and most browser-supported video formats

---

### 4. st.audio()

Embeds an audio player.

```python
audio_url = "https://sample-videos.com/audio/mp3/crowd-cheering.mp3"
st.audio(audio_url, format="audio/mp3", start_time=5)
```

**Parameters:**
- `data` - Audio URL, file path, or bytes
- `format` - MIME type (e.g., 'audio/mp3', 'audio/wav')
- `start_time` - Start position in seconds

## How It Works

1. **Media Loading**: Streamlit fetches media from URLs or reads local files
2. **Rendering**: Uses HTML5 `<img>`, `<video>`, and `<audio>` tags for playback
3. **Auto-refresh**: The example uses `st_autorefresh` to periodically refresh content (useful for live dashboards)

## Loading Local Files

You can also load media from local files:

```python
# Load image from file
from PIL import Image
img = Image.open("path/to/image.png")
st.image(img, caption="Local Image")

# Load video from file
video_file = open("path/to/video.mp4", "rb")
st.video(video_file)

# Load audio from file  
audio_file = open("path/to/audio.mp3", "rb")
st.audio(audio_file, format="audio/mp3")
```

## Running the App

```bash
streamlit run Part_5_Media/app.py
# or
streamlit run Part_5_Media/p5_media.py
```

**Note:** Requires `streamlit-autorefresh` package:
```bash
pip install streamlit-autorefresh
```

## Next Steps

After completing this section, you should understand:
- How to display images from URLs and local files
- Embedding video and audio content
- Adding captions and controlling playback

Proceed to [Part 6: Charts](../Part_6_Charts/) to learn about data visualization.