"""
Streamlit Tutorial: Advanced Multi-Page Navigation - Method 1: st.page_link
===========================================================================

This app demonstrates navigation using st.page_link, which creates clickable links
to other pages in your multi-page Streamlit application.

Key Features:
- Static navigation links in sidebar
- Direct page navigation via clickable links
- Clean, organized link presentation
- Icons and labels for better UX

Usage:
Run this app with: streamlit run app1_page_links.py
"""

import streamlit as st

# Configure the page
st.set_page_config(
    page_title="Page Links Demo",
    page_icon="🔗",
    layout="wide"
)

# Initialize session state for demo data
if 'page_visits' not in st.session_state:
    st.session_state.page_visits = {
        'home': 0,
        'about': 0,
        'contact': 0,
        'dashboard': 0
    }

# Track current page visit
current_page = 'home'
st.session_state.page_visits[current_page] += 1

# Main content
st.title("🔗 Navigation Method 1: Page Links")
st.write("This demo shows how to use `st.page_link` for navigation between pages.")

# Create main content area with two columns
col1, col2 = st.columns([2, 1])

with col1:
    st.header("About Page Links Navigation")
    st.markdown("""
    ### What is st.page_link?
    
    `st.page_link` creates clickable links that navigate users to different pages 
    in your multi-page Streamlit application. This method is ideal for:
    
    - **Static navigation menus** - Links that don't change based on user state
    - **Clean link presentation** - Professional-looking navigation elements
    - **External links** - Can also link to external URLs
    - **Accessible navigation** - Screen reader friendly
    
    ### Syntax:
    ```python
    st.page_link("page_file.py", label="Page Name", icon="🏠")
    st.page_link("https://external-url.com", label="External Link")
    ```
    
    ### Best Practices:
    - Use descriptive labels
    - Include relevant icons
    - Group related links together
    - Place navigation in consistent locations (sidebar, header)
    """)
    
    # Demo section
    st.header("Interactive Demo")
    st.info("🎯 **Try the navigation links in the sidebar** to see how page links work!")
    
    # Show current page stats
    st.subheader("Page Visit Statistics")
    visit_data = st.session_state.page_visits
    
    for page, visits in visit_data.items():
        st.write(f"**{page.title()}**: {visits} visits")

with col2:
    st.header("Navigation Tips")
    st.markdown("""
    ### Link Types:
    - **Internal pages**: Use relative file paths
    - **External URLs**: Use full URLs with http/https
    - **Same-app pages**: Reference other .py files
    
    ### Icon Options:
    - Emoji: 🏠, 📊, 📞, ℹ️
    - Material Icons: `:material/home:`
    - Font Awesome: `:fa-solid fa-home:`
    """)

# Sidebar Navigation using st.page_link
st.sidebar.title("🧭 Navigation Menu")
st.sidebar.markdown("---")

# Primary Navigation Links
st.sidebar.subheader("📍 Main Pages")
st.page_link("app1_page_links.py", label="🏠 Home", icon="🏠")
st.page_link("pages/about.py", label="ℹ️ About Us", icon="ℹ️") 
st.page_link("pages/contact.py", label="📞 Contact", icon="📞")
st.page_link("pages/dashboard.py", label="📊 Dashboard", icon="📊")

st.sidebar.markdown("---")

# Secondary Navigation Links
st.sidebar.subheader("🔧 Other Apps")
st.page_link("app2_switch_page.py", label="🔄 Switch Page Demo", icon="🔄")
st.page_link("app3_custom_selector.py", label="🎛️ Custom Selector Demo", icon="🎛️")

st.sidebar.markdown("---")

# External Links
st.sidebar.subheader("🌐 External Resources")
st.page_link("https://docs.streamlit.io/develop/api-reference/navigation/page_link", 
            label="📚 st.page_link Docs")
st.page_link("https://streamlit.io", label="🚀 Streamlit Website")

st.sidebar.markdown("---")

# Implementation Code Example
st.sidebar.subheader("💻 Code Example")
st.sidebar.code("""
# Basic page link
st.page_link("page.py", 
            label="Page Name", 
            icon="🏠")

# External link
st.page_link("https://example.com", 
            label="External Site")
""", language="python")

# Footer with tutorial information
st.markdown("---")
st.markdown("""
### 📝 Tutorial Notes:

**Advantages of st.page_link:**
- ✅ Simple and straightforward
- ✅ Works well for static navigation
- ✅ Supports both internal and external links
- ✅ Good for traditional website-style navigation

**Considerations:**
- ⚠️ Links are static (can't be dynamically hidden/shown easily)
- ⚠️ No built-in state management for active page highlighting
- ⚠️ Requires separate pages files to be created

**Next:** Check out `app2_switch_page.py` for programmatic navigation!
""")