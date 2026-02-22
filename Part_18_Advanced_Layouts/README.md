# 📦 Advanced Layouts: Mastering `st.container()` in Streamlit

## Tutorial Overview

This section explores the advanced features of [`st.container()`](https://docs.streamlit.io/develop/api-reference/layout/st.container), one of Streamit's most powerful layout components. The tutorial covers container basics, border and height control, width modes, horizontal layouts, and alignment options—all through interactive examples with live demos.

## 🎯 Learning Objectives

- Understand how to use `st.container()` for grouping related elements
- Master container styling with borders, heights, and widths
- Create horizontal layouts using containers
- Control element alignment within containers
- Build complex, well-organized Streamlit interfaces

---

## 📚 Topics Covered

### 1️⃣ Basic Usage

The fundamental way to group multiple elements together in a logical section.

**Key Concepts:**
- Using `with st.container()` context manager
- Creating containers without context manager (object reference approach)
- Adding borders for visual separation

```python
# Context manager approach
with st.container(border=True):
    st.write("Inside the container")
    st.bar_chart(np.random.randn(20, 2))

# Object reference approach
c = st.container(border=True)
c.write("First...")
st.write("Between")  # Outside container
c.write("Last!")
```

**Best For:**
- Grouping related content sections
- Creating visual boundaries around content blocks
- Organizing complex layouts

---

### 2️⃣ Border & Height Control

Customize the visual appearance of containers with borders and fixed heights.

**Key Concepts:**
- `border=True` for visible container boundaries
- `height` parameter for fixed-height containers (in pixels)
- Combining both for styled, sized sections

```python
# Fixed height container
with st.container(height=200, border=True):
    for i in range(10):
        st.write(f"Line {i+1}")
```

**Best For:**
- Scrollable content areas
- Consistent section sizing
- Visual hierarchy in dashboards

---

### 3️⃣ Width Control

Control how containers occupy available space with three width modes.

**Key Concepts:**
- `width="stretch"` - fills available horizontal space (default)
- `width="content"` - sizes to fit content only
- `width=<number>` - fixed pixel width

```python
# Stretch to fill available space
with st.container(width="stretch", border=True):
    st.write("Full width container")

# Size to content
with st.container(width="content", border=True):
    st.write("Fits content")

# Fixed 300px width
with st.container(width=300, border=True):
    st.write("Fixed width")
```

**Best For:**
- Responsive layouts
- Sidebar-style panels
- Consistent column-like structures

---

### 4️⃣ Horizontal Layout

Create horizontal arrangements of elements using containers.

**Key Concepts:**
- `horizontal=True` enables horizontal layout mode
- Elements flow left-to-right instead of top-to-bottom
- Works with buttons, inputs, and other widgets

```python
# Create horizontal button row
flex = st.container(horizontal=True, border=True)
for i in range(4):
    flex.button(f"Button {i+1}")
```

**Best For:**
- Button toolbars
- Horizontal form inputs
- Inline widget arrangements

---

### 5️⃣ Alignment & Gap Control

Fine-tune element positioning within containers.

**Key Concepts:**
- `vertical_alignment` - "top", "center", or "bottom"
- `gap` - spacing between elements: None, "small", "medium", "large"
- Combining alignment with height for vertical centering

```python
# Center-aligned with medium gap
with st.container(
    height=180,
    horizontal=True,
    vertical_alignment="center",
    gap="medium",
    border=True
):
    st.button("Short")
    st.text_area("Tall", height=100)
    st.button("Short 2")
```

**Best For:**
- Aligning elements of different heights
- Creating polished, professional layouts
- Consistent spacing across sections

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install streamlit numpy pandas
```

### Running the Tutorial App
```bash
cd Part_18_Advanced_Layouts
streamlit run 18_app.py
```

The app includes interactive demos for each topic—modify parameters using the sliders and controls to see how containers respond in real-time.

---

## 📁 Project Structure

```
Part_18_Advanced_Layouts/
├── README.md          # This documentation
└── 18_app.py          # Interactive tutorial application
```

---

## 🔍 Comparison: Container vs. Other Layout Elements

| Feature | `st.container()` | `st.columns()` | `st.expander()` |
|---------|------------------|----------------|-----------------|
| **Grouping** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Border** | ✅ Yes | ❌ No | ✅ Yes (implicit) |
| **Height Control** | ✅ Yes | ❌ No | ❌ No |
| **Width Control** | ✅ Yes | ✅ Yes | ❌ No |
| **Horizontal Layout** | ✅ Yes | ✅ Yes | ❌ No |
| **Collapsible** | ❌ No | ❌ No | ✅ Yes |

---

## 💡 Pro Tips

### 1. Nesting Containers
```python
with st.container(border=True):
    st.write("Outer container")
    with st.container(border=True):
        st.write("Nested container")
```

### 2. Combining with Session State
```python
if 'container_state' not in st.session_state:
    st.session_state.container_state = {}

# Update state from within container
with st.container():
    if st.button("Update"):
        st.session_state.container_state['updated'] = True
```

### 3. Dynamic Content Placement
```python
# Place content strategically using containers
header_container = st.container()
content_container = st.container()

with header_container:
    st.title("My App")

with content_container:
    # Main content here
    pass
```

---

## 📚 Official Documentation

- [st.container API Reference](https://docs.streamlit.io/develop/api-reference/layout/st.container)
- [Layout Elements Overview](https://docs.streamlit.io/develop/concepts/layout)
- [Advanced Layout Techniques](https://docs.streamlit.io/library/api-reference/layout)

---

## 🔧 Troubleshooting

### Container Not Rendering
```python
# Ensure proper context manager usage
with st.container():
    # ✅ Correct - content inside context
    pass
```

### Elements Overflowing Height
```python
# Use scroll or increase height
with st.container(height=400):
    # Content here
```

### Horizontal Layout Not Working
```python
# Must set horizontal=True explicitly
flex = st.container(horizontal=True)  # ✅ Correct
flex = st.container()                 # ❌ Default is vertical
```

---

## 🎉 What's Next?

After mastering `st.container()`, explore these related topics:

- **Part_18_Multipage_Stock_Dashboard** - Real-world application using containers in a multi-page dashboard
- **Custom Components** - Building reusable UI components with Streamlit

---

*Built with ❤️ for the Streamlit community*