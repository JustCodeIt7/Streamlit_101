# Part 1: Introduction to Streamlit - st.write() and Magic Commands

This section covers the foundational concepts of Streamlit, focusing on displaying content using `st.write()` and related text display functions.

## Overview

Streamlit is a Python framework for building data apps quickly. This section introduces the most commonly used display functions that allow you to show text, data, and media in your applications.

## Files in This Section

- **[app.py](Part_1_write/app.py)** - Main application file demonstrating basic Streamlit features
- **[p1_write.py](Part_1_write/p1_write.py)** - Tutorial script with detailed comments explaining each concept

## Key Concepts Covered

### 1. st.title()
Sets the main title of the Streamlit app, displayed as a large heading at the top of the page.

```python
st.title("Intro to Streamlit")
```

### 2. st.subheader()
Creates a subheading below the main title for section organization.

```python
st.subheader("This is a subheader")
```

### 3. st.write() - The Versatile Display Function

The `st.write()` function is one of Streamlit's most versatile functions. It can display:
- **Text strings** - Plain text output
- **DataFrames** - Pandas DataFrames are automatically rendered as tables
- **Charts** - Matplotlib, Plotly, and other chart objects
- **Images** - PIL/Pillow images

```python
# Display a simple DataFrame
df = pd.DataFrame({
    "Column 1": [1, 2, 3, 4],
    "Column 2": [10, 20, 30, 40]
})
st.write(df)
```

### 4. st.markdown()

For more control over text formatting, `st.markdown()` allows you to render Markdown-formatted text including:
- Headers (##, ###)
- Bold and italic text
- Lists
- Links
- Code blocks

```python
markdown_txt = ("### This is a Markdown Header\n"
                "#### This is a Markdown Subheader\n"
                "This is a Markdown paragraph.\n")
st.markdown(markdown_txt)
```

### 5. st.write_stream()

The `st.write_stream()` function allows you to stream data in real-time, displaying content incrementally as it becomes available. This is useful for:
- Live data feeds
- Chat responses
- Long-running computations

```python
def stream_data(txt="Hello, World!"):
    for word in txt.split(" "):
        yield word + " "
        time.sleep(0.1)

if stream_btn:
    st.write_stream(stream_data(TEXT))
```

## How It Works

1. **Title and Subheader** - These provide structure to your app
2. **st.write()** - Automatically detects the type of data passed and renders it appropriately
3. **Markdown** - Gives you full control over text formatting using standard Markdown syntax
4. **Streaming** - Uses Python generators (`yield`) to progressively display content

## Running the App

To run this Streamlit app:

```bash
streamlit run Part_1_write/app.py
# or
streamlit run Part_1_write/p1_write.py
```

## Next Steps

After completing this section, you should understand:
- How to set up basic text display in Streamlit
- The difference between `st.write()` and `st.markdown()`
- How to stream data dynamically using generators

Proceed to [Part 2: Text Elements](../Part_2_Text/) to learn about more text formatting options.