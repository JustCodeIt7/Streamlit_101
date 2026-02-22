# Part 16: Custom Components in Streamlit

This section covers extending Streamlit with custom components.

## Overview

Streamlit's Component API allows you to create custom interactive elements using HTML, CSS, and JavaScript. This enables:
- Integration of third-party JavaScript libraries
- Creation of custom UI elements
- Enhanced interactivity beyond built-in widgets

## Files in This Section

- **[app.py](Part_16_Custom_Components/app.py)** - Placeholder for custom components demo
- **[p16_custom.py](Part_16_Custom_Components/p16_custom.py)** - Template file (placeholder)

## Key Concepts Covered

### 1. Creating Custom Components

Custom components are created using:
- **Frontend**: HTML/CSS/JavaScript bundle
- **Backend**: Python wrapper using `streamlit.components.v1`

```python
import streamlit as st
from streamlit.components.v1 import html

# Embed custom component
def my_component():
    return html("""
        <div class="custom-widget">
            <button onclick="alert('Hello!')">Click Me</button>
        </div>
    """)
```

---

### 2. Component Structure

A typical custom component structure:

```
my_component/
├── frontend/
│   ├── index.html
│   ├── index.js
│   └── styles.css
└── my_component.py
```

---

### 3. Using Existing Components

Many pre-built components are available:
- **streamlit-echarts**: Interactive charts
- **streamlit-webrtc**: WebRTC support
- **streamlit-js-eval**: JavaScript evaluation

```python
from streamlit.components.v1 import declare_component

# Load custom component
my_comp = declare_component("my_custom_widget", path="path/to/component")
```

---

### 4. Component Communication

Components can communicate with Streamlit via:

```javascript
// In frontend (JavaScript)
streamlit.setComponentValue("value_to_send_python");

// In Python
result = my_component()
```

## Running the App

Since this section is a placeholder, there's no runnable demo yet.

## Next Steps

To learn more about creating custom components:
1. Visit [Streamlit Component API docs](https://docs.streamlit.io/develop/custom-components)
2. Use `streamlit component template` to generate starter code
3. Explore existing community components on GitHub