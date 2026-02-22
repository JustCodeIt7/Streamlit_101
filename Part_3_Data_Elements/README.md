# Part 3: Data Elements in Streamlit

This section covers data display elements in Streamlit, including interactive tables, static tables, JSON display, editable grids, and metric cards.

## Overview

Streamlit provides powerful components for displaying and interacting with data. These elements allow you to present data in various formats while maintaining interactivity where needed.

## Files in This Section

- **[app.py](Part_3_Data_Elements/app.py)** - Main application demonstrating all data elements
- **[p3_data.py](Part_3_Data_Elements/p3_data.py)** - Tutorial script with detailed comments explaining each element

## Key Concepts Covered

### 1. st.dataframe()

Displays an interactive DataFrame that users can sort, filter, and scroll through.

```python
df = pd.DataFrame({
    "Column 1": [1, 2, 3, 4],
    "Column 2": [10, 20, 30, 40]
})

st.dataframe(df, width=500, height=200, hide_index=False)
```

**Features:**
- Column sorting (click headers)
- Horizontal and vertical scrolling
- Configurable dimensions with `width` and `height`
- Option to show/hide index with `hide_index`

---

### 2. st.table()

Displays a static table where all content is rendered at once.

```python
st.table(df)
```

**Differences from st.dataframe:**
- Static rendering - entire table loaded upfront
- No sorting or filtering capabilities
- Better for small datasets that don't need interactivity

---

### 3. st.json()

Displays JSON data in a formatted, collapsible structure.

```python
data = {
    "Column 1": [1, 2, 3, 4],
    "Column 2": [10, 20, 30, 40]
}
st.json(data, expanded=True)
```

**Features:**
- Pretty-printed formatting
- Collapsible/expandable nodes with `expanded` parameter
- Syntax highlighting

---

### 4. st.data_editor()

Creates an interactive data editor where users can modify data directly.

```python
df = pd.DataFrame({
    "Column 1": [1, 2, 3, 4],
    "Column 2": [10, 20, 30, 40]
})

st.data_editor(df)
```

**Features:**
- In-place cell editing
- Add/delete rows with `num_rows` parameter
- Column configuration for customization
- Returns edited data as a DataFrame

```python
# With column configuration
st.data_editor(
    df,
    column_config={
        "Column 1": st.column_config.Column(
            help="Custom help text",
            width="medium",
            required=True,
        )
    },
    num_rows="dynamic",  # Allow adding/deleting rows
)
```

---

### 5. st.metric()

Displays a metric value with optional delta (change) indicator.

```python
st.metric("Metric 1", 100, 5)
st.metric("Metric 2", 200, -3)
```

**Parameters:**
- `label` - The metric name/description
- `value` - The numeric value to display
- `delta` - Optional change/delta value (positive or negative)

**Use cases:**
- KPIs and dashboard metrics
- Showing changes from previous periods
- Status indicators

---

### 6. st.column_config

A configuration object used within `st.data_editor` to customize column behavior.

```python
column_config={
    "widgets": st.column_config.Column(
        "Streamlit Widgets",
        help="Streamlit **widget** commands 🎈",
        width="medium",
        required=True,
    )
}
```

**Available options:**
- `width` - Column width ('small', 'medium', 'large')
- `required` - Whether the column is required
- `help` - Help text displayed on hover
- `disabled` - Make column read-only

## How It Works

1. **DataFrame/Table**: Renders pandas DataFrames using AgGrid under the hood for interactivity
2. **JSON**: Parses and renders JSON with collapsible tree structure
3. **Data Editor**: Creates an editable grid that captures user input and returns modified data
4. **Metric**: Renders a card showing value with optional delta indicator

## Running the App

```bash
streamlit run Part_3_Data_Elements/app.py
# or
streamlit run Part_3_Data_Elements/p3_data.py
```

## Next Steps

After completing this section, you should understand:
- When to use interactive vs static tables
- How to allow users to edit data in-place
- Displaying metrics with change indicators
- Configuring columns for better UX

Proceed to [Part 4: Inputs](../Part_4_Inputs/) to learn about user input widgets.