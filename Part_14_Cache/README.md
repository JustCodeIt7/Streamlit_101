# Part 14: Cache in Streamlit

This section covers caching mechanisms to improve application performance.

## Overview

Streamlit provides two main caching decorators:
- **st.cache_data**: For caching data and function results
- **st.cache_resource**: For caching global resources like ML models and database connections

## Files in This Section

- **[app.py](Part_14_Cache/app.py)** - Main application demonstrating cache usage
- **[p14_cache.py](Part_14_Cache/p14_cache.py)** - Tutorial script with detailed explanations
- **performance.py**, **performance2.py** - Performance testing examples

## Key Concepts Covered

### 1. st.cache_data

Use for caching function results that return data:

```python
@st.cache_data
def load_data(nrows):
    time.sleep(2)  # Simulate expensive computation
    df = pd.DataFrame(np.random.randn(nrows, 3), columns=["A", "B", "C"])
    return df

# First call: ~2 seconds
# Subsequent calls: nearly instant (cached)
data = load_data(1000)
```

**Key Points:**
- Caches function return values
- Ideal for data loading functions
- Parameters affect cache key (same params = cached result)

---

### 2. st.cache_resource

Use for caching global resources shared across sessions:

```python
@st.cache_resource(ttl=60)
def load_model():
    time.sleep(3)  # Simulate model loading
    return "Pretend this is a large ML model"

# First call: ~3 seconds  
# Subsequent calls: nearly instant (cached, shared across users)
model = load_model()
```

**Key Points:**
- Caches the resource object itself
- Ideal for ML models, database connections
- Shared across all user sessions
- Use `ttl` parameter for time-to-live

---

### 3. Cache Parameters

| Parameter | Description |
|-----------|-------------|
| `ttl` | Time-to-live in seconds before cache expires |
| `max_entries` | Maximum number of entries in the cache |

```python
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_api_data():
    return fetch_data()

@st.cache_data(max_entries=1000)  # Max 1000 cached entries  
def get_large_array(seed):
    return np.random.rand(100000)
```

---

### 4. Clearing Cache

Clear cache to refresh data or troubleshoot issues:

```python
# Clear specific function's cache
load_data.clear()

# Clear all data caches
st.cache_data.clear()

# Clear all resource caches
st.cache_resource.clear()

# Clear everything
st.cache_data.clear()
st.cache_resource.clear()
```

---

### 5. Cache Invalidation

Cache is invalidated when:
- Function code changes
- Input parameters differ
- TTL expires (if set)
- Cache manually cleared

## Running the App

```bash
streamlit run Part_14_Cache/app.py
# or
streamlit run Part_14_Cache/p14_cache.py
```

## Best Practices

1. **Use st.cache_data for**: DataFrames, API responses, computed values
2. **Use st.cache_resource for**: ML models, database connections, file handles
3. **Set appropriate TTL**: Balance freshness vs performance
4. **Clear cache when needed**: When underlying data changes significantly

## Next Steps

After completing this section, you should understand:
- When to use cache_data vs cache_resource
- Setting cache parameters (ttl, max_entries)
- Clearing and managing cached values

Proceed to [Part 16: Custom Components](../Part_16_Custom_Components/) to learn about extending Streamlit.