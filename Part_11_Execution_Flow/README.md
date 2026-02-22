# Part 11: Execution Flow Elements in Streamlit

This section covers advanced execution control mechanisms including dialogs, fragments, rerun controls, and forms.

## Overview

Streamlit provides several elements to control application flow:
- **Dialogs**: Modal windows for focused interactions
- **Fragments**: Independent rerunnable components
- **Rerun Controls**: Manual script execution triggers
- **Forms**: Batch input submission

## Files in This Section

- **[app.py](Part_11_Execution_Flow/app.py)** - Main application demonstrating all execution flow elements
- **[p11_flow.py](Part_11_Execution_Flow/p11_flow.py)** - Tutorial script with detailed comments

## Key Concepts Covered

### 1. st.dialog()

Creates a modal dialog window for focused user interactions.

```python
@st.dialog("Example Dialog")
def show_dialog():
    st.write("This is a modal dialog.")
    name = st.text_input("Enter your name")
    if st.button("Submit"):
        st.session_state.dialog_name = name
        st.rerun()

# Button to open the dialog
if st.button("Open Dialog"):
    show_dialog()
```

**Key Features:**
- Modal overlay that blocks interaction with main app
- Must call `st.rerun()` to close after submission
- Can contain any Streamlit elements

---

### 2. st.fragment()

Creates a component that can rerun independently from the rest of the app.

```python
@st.fragment
def update_counter():
    if "counter" not in st.session_state:
        st.session_state.counter = 0
    
    st.write(f"Counter: {st.session_state.counter}")
    
    if st.button("Increment (Fragment Rerun)"):
        st.session_state.counter += 1
        st.rerun(scope="fragment")

update_counter()
```

**Key Features:**
- Only the fragment reruns when triggered
- Improves performance for interactive components
- Useful for widgets that need frequent updates

---

### 3. st.rerun()

Triggers a script rerun, either full app or fragment-specific.

```python
# Full app rerun (default)
st.rerun()

# Fragment-only rerun
st.rerun(scope="fragment")
```

**Use Cases:**
- Refresh UI after state changes
- Close dialogs after submission
- Update fragments without reloading entire page

---

### 4. st.stop()

Halts script execution at the point where it's called.

```python
if st.checkbox("Stop execution"):
    st.warning("Execution will stop here.")
    st.stop()  # Script stops here if checked

# This only executes if not stopped
st.success("This will only show if execution wasn't stopped.")
```

**Use Cases:**
- Conditional content display
- Early exit based on user authentication or settings

---

### 5. st.form and st.form_submit_button()

Groups multiple inputs together for batch submission.

```python
with st.form("example_form"):
    st.write("Inside the form:")
    name = st.text_input("Name")
    age = st.number_input("Age", min_value=0, max_value=120)
    submitted = st.form_submit_button("Submit Form")
    
    if submitted:
        st.session_state.form_data = {"name": name, "age": age}
        st.success("Form submitted successfully!")
```

**Key Features:**
- Single submission for all form inputs
- Prevents individual widget reruns on each input change
- Returns `True` only when submit button is clicked

---

### 6. Combining Elements

Dialogs can contain forms for complex interactions:

```python
@st.dialog("Complex Dialog")
def complex_dialog():
    with st.form("dialog_form"):
        name = st.text_input("Name")
        age = st.number_input("Age", min_value=0, max_value=120)
        submitted = st.form_submit_button("Submit")
    
    if submitted:
        st.session_state.dialog_result = {"name": name, "age": age}
        st.rerun()

if st.button("Open Complex Dialog"):
    complex_dialog()
```

## Running the App

```bash
streamlit run Part_11_Execution_Flow/app.py
# or
streamlit run Part_11_Execution_Flow/p11_flow.py
```

## Next Steps

After completing this section, you should understand:
- How to create modal dialogs for focused interactions
- Using fragments for independent component updates
- Controlling script execution flow with rerun and stop
- Building forms for batch input handling

Proceed to [Part 12: State Management](../Part_12_State_Management/) to learn about session state persistence.