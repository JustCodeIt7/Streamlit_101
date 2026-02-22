# Part 12: State Management in Streamlit

This section covers session state management, including initialization, callbacks, and widget association.

## Overview

Streamlit's session state allows you to persist data across script reruns within a user's session. This is essential for:
- Maintaining user inputs between interactions
- Tracking application state
- Building interactive applications with memory

## Files in This Section

- **[app.py](Part_12_State_Management/app.py)** - Main application demonstrating state management basics
- **[p12_app.py](Part_12_State_Management/p12_app.py)** - Tutorial script with callbacks and forms
- **Various versions**: `app-v2.py`, `app-3.py`, `app copy.py` - Alternative implementations

## Key Concepts Covered

### 1. Initialize Values in Session State

Session state can be initialized using either attribute or dictionary syntax:

```python
# Attribute syntax (recommended)
if "counter" not in st.session_state:
    st.session_state.counter = 1

# Dictionary syntax
if "input_text" not in st.session_state:
    st.session_state["input_text"] = ""
```

**Key Points:**
- Always check if key exists before initializing
- Use attribute or dictionary syntax interchangeably
- State persists across reruns within a session

---

### 2. Update Session State with Callbacks

Callbacks allow you to execute code when widget values change:

```python
def name_callback():
    st.session_state.name = st.session_state.name_input.title()

st.text_input("Enter your name", key="name_input", on_change=name_callback)
```

**Key Points:**
- `on_change` callback fires when value changes
- Callback runs before the main script
- Useful for data validation and transformation

---

### 3. Delete Items from Session State

Remove state items to reset or clear data:

```python
# Remove specific key
if "temp_data" in st.session_state:
    del st.session_state["temp_data"]

# Clear all session state (on button click)
if st.button("Clear All"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
```

---

### 4. Session State and Widget Association

Widgets can be associated with session state using the `key` parameter:

```python
# Key automatically syncs with session state
st.text_input("Name", key="name")
# Access via: st.session_state.name

st.slider("Value", 0, 100, 50, key="slider_value")
# Access via: st.session_state.slider_value
```

**Key Points:**
- Using `key` creates automatic state binding
- Widget changes update session state automatically
- Useful for form inputs that need to persist

---

### 5. Forms and Callbacks

Forms can use callbacks for batch processing:

```python
def form_callback():
    st.write(f"Slider value: {st.session_state.my_slider}")
    st.write(f"Checkbox value: {st.session_state.my_checkbox}")

with st.form(key="my_form"):
    st.slider("My slider", 0, 10, 5, key="my_slider")
    st.checkbox("Yes or No", key="my_checkbox")
    st.form_submit_button(label="Submit", on_click=form_callback)
```

**Key Points:**
- Form submit triggers callback
- All form values available in session state
- Single rerun for entire form submission

---

### 6. Display Current State

For debugging, display all session state:

```python
with st.sidebar:
    st.header("Current State")
    st.write(st.session_state)
```

## Running the App

```bash
streamlit run Part_12_State_Management/app.py
# or
streamlit run Part_12_State_Management/p12_app.py
```

## Next Steps

After completing this section, you should understand:
- How to initialize and persist session state
- Using callbacks for reactive updates
- Associating widgets with state keys
- Managing form submissions with callbacks

Proceed to [Part 13: Connections & Secrets](../Part_13_Conn_Secrets/) to learn about secure configuration.