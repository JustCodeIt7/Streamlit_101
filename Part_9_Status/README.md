# Part 9: Status Elements in Streamlit

This section covers status indicators, progress bars, and feedback components used to communicate operation states to users.

## Overview

Streamlit provides various status elements to give users visual feedback during operations. These are essential for:
- Showing loading progress
- Displaying success/error/warning messages
- Indicating ongoing processes with spinners
- Celebrating achievements with fun effects

## Files in This Section

- **[app.py](Part_9_Status/app.py)** - Main application demonstrating all status elements
- **[p9_status.py](Part_9_Status/p9_status.py)** - Tutorial script with detailed comments explaining each element

## Key Concepts Covered

### 1. st.progress()

Displays a progress bar for tracking operation completion.

```python
progress_text = "Operation in progress. Please wait."
my_bar = st.progress(value=0, text=progress_text)

# Update the progress bar
for percent_complete in range(100):
    time.sleep(0.001)
    my_bar.progress(percent_complete + 1, text=progress_text)

# Clear the progress bar
my_bar.empty()
```

**Methods:**
- `progress(value)` - Updates to a specific percentage (0-100)
- `empty()` - Removes the progress bar from display

---

### 2. st.success()

Displays a success message with green styling.

```python
st.success("Operation completed successfully!", icon="✅")
```

---

### 3. st.error()

Displays an error message with red styling.

```python
st.error("An error occurred while processing your request!")
```

---

### 4. st.warning()

Displays a warning message with yellow/orange styling.

```python
st.warning("This action cannot be undone!")
```

---

### 5. st.info()

Displays an informational message with blue styling.

```python
st.info("Your session will expire in 5 minutes.")
```

---

### 6. st.exception()

Displays exception information including stack trace.

```python
try:
    raise Exception("This is an exception!")
except Exception as e:
    st.exception(e)
```

**Use case:** Display detailed error information for debugging or user understanding.

---

### 7. st.spinner()

Displays a loading spinner while executing code within the context manager.

```python
with st.spinner("Processing your request..."):
    time.sleep(1.5)  # Simulate long-running operation
st.success("Done!")
```

**Use case:** Indicate that an operation is in progress without blocking the UI.

---

### 8. st.balloons()

Displays a fun balloon celebration effect.

```python
bbtn = st.button("Click me to display balloons")
if bbtn:
    st.balloons()
```

**Use case:** Celebrate achievements, successful completions, or special occasions.

---

### 9. st.snow()

Displays a snowfall animation effect.

```python
snow_btn = st.button("Click me to display snow")
if snow_btn:
    st.snow()
```

**Use case:** Festive effects for winter holidays or special events.

## How It Works

1. **Progress Bar**: Creates a visual indicator that updates as operations progress
2. **Status Messages**: Display colored banners with appropriate icons
3. **Spinners**: Show animated loading indicators during async operations
4. **Effects**: Trigger celebratory animations on user actions

## Running the App

```bash
streamlit run Part_9_Status/app.py
# or
streamlit run Part_9_Status/p9_status.py
```

## Next Steps

After completing this section, you should understand:
- How to show progress for long-running operations
- Display different types of status messages
- Use spinners for loading states

Proceed to [Part 10: Navigation & Pages](../Part_10_Nav_Pages/) to learn about multi-page navigation.