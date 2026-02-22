# Part 4: Input Elements in Streamlit

This section covers all user input widgets available in Streamlit, from basic buttons to file uploaders and camera input.

## Overview

Streamlit provides a comprehensive set of input widgets that allow users to interact with your application. These widgets capture different types of user input and return values that can be used in your app's logic.

## Files in This Section

- **[app.py](Part_4_Inputs/app.py)** - Main application demonstrating all input elements
- **[p4_inputs.py](Part_4_Inputs/p4_inputs.py)** - Tutorial script with detailed comments explaining each element

## Key Concepts Covered

### 1. st.button()

Creates a clickable button that returns `True` when clicked.

```python
btn1 = st.button("Click Me", key="button", help="Click me to see the magic", type='secondary', disabled=False)
if btn1:
    st.write("Button Clicked")
```

**Parameters:**
- `key` - Unique identifier for the widget
- `help` - Tooltip text displayed on hover
- `type` - Button style: 'primary', 'secondary', or 'tertiary'
- `disabled` - Whether the button is clickable

---

### 2. st.link_button()

Creates a button that navigates to a URL when clicked.

```python
if st.link_button("Click Me", "https://www.streamlit.io/"):
    st.write("Link Button Clicked")
```

**Use case:** Redirect users to external pages while maintaining app-like experience.

---

### 3. st.download_button()

Creates a button that downloads a file when clicked.

```python
if st.download_button("Download Me", "hello world", "hello.txt", mime='text/plain'):
    st.write("Download Button Clicked")
```

**Parameters:**
- `label` - Button text
- `data` - Content to download (string or bytes)
- `file_name` - Name of downloaded file
- `mime` - MIME type of the file

---

### 4. st.checkbox()

A toggle checkbox for boolean input.

```python
checkbox_val = st.checkbox("Check Me", value=False)
if checkbox_val:
    st.write("Checkbox Checked")
```

**Use case:** Enable/disable features, confirm agreements, toggle settings.

---

### 5. st.radio()

Single-selection radio buttons.

```python
radio_val = st.radio("Select Color", ["Red", "Green", "Blue"], index=0)
if radio_val:
    st.write(f"You selected {radio_val}")
```

**Parameters:**
- `options` - List of options to choose from
- `index` - Default selection index (0-based)

---

### 6. st.selectbox()

Dropdown single-selection widget.

```python
select_val = st.selectbox("Select Color", ["Red", "Green", "Blue", "Black"], index=1)
```

**Use case:** When you have many options and want to save space with a dropdown.

---

### 7. st.multiselect()

Multi-selection dropdown that returns a list.

```python
multiselect_val = st.multiselect("Select Colors", ["Red", "Green", "Blue", "Black"], default=["Red"])
```

**Use case:** When users need to select multiple options from a list.

---

### 8. st.select_slider()

Slider for selecting from discrete options.

```python
select_slider_val = st.select_slider("Select Value", options=range(1, 101), value=50)
```

**Parameters:**
- `options` - Range or list of values to select from
- `value` - Default selection

---

### 9. st.text_input()

Single-line text input field.

```python
text_input_val = st.text_input("Enter some text", value="", max_chars=50)
```

**Parameters:**
- `max_chars` - Maximum number of characters allowed
- `type` - 'default' or 'password'

---

### 10. st.text_area()

Multi-line text input field.

```python
text_area_val = st.text_area("Enter some text", value="", height=150, max_chars=200)
```

**Parameters:**
- `height` - Height of the text area in pixels
- `max_chars` - Maximum characters allowed

---

### 11. st.number_input()

Numeric input field with optional min/max bounds.

```python
number_input_val = st.number_input("Enter a number", value=0, min_value=0, max_value=100, step=1)
```

**Parameters:**
- `min_value` - Minimum allowable value
- `max_value` - Maximum allowable value
- `step` - Increment/decrement step size

---

### 12. st.date_input()

Calendar date picker.

```python
date_input_val = st.date_input("Enter a date")
```

**Returns:** Python `datetime.date` object

---

### 13. st.time_input()

Time picker widget.

```python
time_input_val = st.time_input("Enter a time")
```

**Returns:** Python `datetime.time` object

---

### 14. st.file_uploader()

File upload widget for accepting user files.

```python
file_uploader_val = st.file_uploader("Upload a file", type=["png", "jpg", "txt"])
if file_uploader_val:
    st.write(f"You uploaded {file_uploader_val.name}")
```

**Parameters:**
- `type` - List of allowed file extensions
- `accept_multiple_files` - Allow multiple file uploads

---

### 15. st.color_picker()

Color selection widget returning hex color codes.

```python
color_picker_val = st.color_picker("Pick a color", value="#00f900")
```

**Returns:** Hex color string (e.g., "#FF5500")

---

### 16. st.camera_input()

Camera capture widget for taking photos.

```python
camera_input_val = st.camera_input("Take a picture", help="Capture an image using your camera")
if camera_input_val:
    st.write("Picture captured successfully")
```

**Returns:** UploadedFile object containing the captured image

## How It Works

1. Each input widget returns a value that changes when user interacts
2. Streamlit re-runs the script on every interaction (by default)
3. The returned values can be used directly in your app logic
4. Use `key` parameter to uniquely identify widgets and access via `st.session_state[key]`

## Running the App

```bash
streamlit run Part_4_Inputs/app.py
# or
streamlit run Part_4_Inputs/p4_inputs.py
```

## Next Steps

After completing this section, you should understand:
- How to capture different types of user input
- Widget return types and how to use them
- Configuring widget behavior with parameters

Proceed to [Part 5: Media](../Part_5_Media/) to learn about displaying images, videos, and audio.