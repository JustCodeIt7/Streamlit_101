# Part 2: Text Elements in Streamlit

This section covers all the text display elements available in Streamlit, from headers to code blocks and mathematical expressions.

## Overview

Streamlit provides a variety of text formatting options that allow you to structure your app's content effectively. These elements help create well-organized, readable applications with proper hierarchy and visual separation.

## Files in This Section

- **[app.py](Part_2_Text/app.py)** - Main application demonstrating all text elements
- **[p2_txt.py](Part_2_Text/p2_txt.py)** - Tutorial script with detailed comments explaining each element

## Key Concepts Covered

### 1. st.title()

The main title of your app, displayed as the largest heading.

```python
st.title("Streamlit Tutorials")
```

**Use case:** Main application title at the top of the page.

---

### 2. st.header()

A section header that is smaller than the title but still prominent.

```python
st.header("This is a Header")
```

**Use case:** Major sections within your app content.

---

### 3. st.subheader()

A subheading for subsections, smaller than headers.

```python
st.subheader("This is a Subheader")
```

**Use case:** Subsections under main headers for better organization.

---

### 4. st.caption()

Small text typically used for additional context or notes.

```python
st.caption("This is a caption")
```

**Use case:** Adding supplementary information in smaller font, such as data source citations or timestamps.

---

### 5. st.code()

Displays code with syntax highlighting and optional line numbers.

```python
code_txt = """
import pandas as pd
import streamlit as st

st.title("Streamlit Tutorials") 
for i in range(10):
    st.write(i)
"""
st.code(code_txt, language='python', wrap_lines=True, line_numbers=True)
```

**Parameters:**
- `language` - Programming language for syntax highlighting ('python', 'javascript', etc.)
- `wrap_lines` - Whether to wrap long lines
- `line_numbers` - Show line numbers

---

### 6. st.text()

Plain text without any styling or formatting.

```python
st.text("This is a text")
```

**Use case:** Simple, unformatted text output.

---

### 7. st.latex()

Renders mathematical expressions using LaTeX notation.

```python
# Einstein's equation
st.latex(r"e = mc^2")

# Integral example
st.latex(r"\int_a^b x^2 dx")
```

**Use case:** Displaying mathematical formulas, equations, and scientific notation. Uses LaTeX syntax (note the `r` prefix for raw strings).

---

### 8. st.divider()

A horizontal line that visually separates content sections.

```python
st.write("This is some text below the divider.")
st.divider()
st.write("This is some other text below the divider.")
```

**Use case:** Creating visual separation between different content areas.

## How It Works

Each text element renders HTML elements in Streamlit's frontend:
- `st.title()` → `<h1>` tag
- `st.header()` → `<h2>` tag  
- `st.subheader()` → `<h3>` tag
- `st.caption()` → Small font, typically `<small>` or caption styling
- `st.code()` → Preformatted code block with syntax highlighting
- `st.text()` → Plain paragraph text
- `st.latex()` → Rendered MathML or LaTeX output
- `st.divider()` → Horizontal rule `<hr>`

## Running the App

```bash
streamlit run Part_2_Text/app.py
# or
streamlit run Part_2_Text/p2_txt.py
```

## Next Steps

After completing this section, you should understand:
- How to structure content with hierarchical headings
- When to use each text element type
- How to display code with syntax highlighting
- Rendering mathematical expressions

Proceed to [Part 3: Data Elements](../Part_3_Data_Elements/) to learn about displaying data in tables and DataFrames.