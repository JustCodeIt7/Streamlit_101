"""
About Page for Page Links Navigation Demo
=========================================
"""

import streamlit as st

st.set_page_config(page_title="About Us", page_icon="ℹ️")

st.title("ℹ️ About Us")

st.markdown("""
## Welcome to the Streamlit Navigation Tutorial Series

This tutorial series demonstrates three advanced methods for implementing 
multi-page navigation in Streamlit applications:

### 🔗 Method 1: Page Links (`st.page_link`)
- Static navigation using clickable links
- Simple and straightforward implementation
- Great for traditional website-style navigation
- Supports both internal pages and external URLs

### 🔄 Method 2: Switch Page (`st.switch_page`)
- Programmatic navigation with conditional logic
- Perfect for form flows and user journeys
- Enables complex routing based on application state
- Ideal for validation-driven navigation

### 🎛️ Method 3: Custom Selector Navigation
- Dynamic content switching within a single page
- Multiple navigation patterns (dropdown, radio, buttons)
- Advanced state management capabilities
- Single-page application (SPA) experience

## Tutorial Features

✅ **Current Streamlit Syntax** - Uses the latest Streamlit API  
✅ **Comprehensive Examples** - Real-world use cases and patterns  
✅ **Best Practices** - Professional implementation guidelines  
✅ **Interactive Demos** - Hands-on learning experience  
✅ **Well Documented** - Clear explanations and code comments  

## Navigation Options

Use the sidebar or the links below to explore different navigation methods:
""")

# Navigation links
col1, col2, col3 = st.columns(3)

with col1:
    st.page_link("../app1_page_links.py", label="🔗 Page Links Demo", icon="🔗")

with col2:
    st.page_link("../app2_switch_page.py", label="🔄 Switch Page Demo", icon="🔄")

with col3:
    st.page_link("../app3_custom_selector.py", label="🎛️ Custom Selector Demo", icon="🎛️")

st.markdown("---")

# Contact information
st.subheader("📞 Contact & Support")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Tutorial Resources:**
    - 📚 [Streamlit Documentation](https://docs.streamlit.io)
    - 🎥 [Video Tutorials](https://streamlit.io/gallery)
    - 💬 [Community Forum](https://discuss.streamlit.io)
    """)

with col2:
    st.markdown("""
    **Quick Links:**
    - 🏠 [Home Page](../app1_page_links.py)
    - 📊 [Dashboard](dashboard.py)
    - 📞 [Contact](contact.py)
    """)

# Back navigation
st.markdown("---")
st.page_link("../app1_page_links.py", label="← Back to Home", icon="🏠")