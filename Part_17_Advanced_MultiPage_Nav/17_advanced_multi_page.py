"""
Streamlit Tutorial: Advanced Multi-Page Navigation - Main Entry Point
======================================================================

This is the main entry point for the Advanced Multi-Page Navigation tutorial series.
From here, you can explore three different navigation methods and understand when
to use each approach in your Streamlit applications.

Author: Streamlit Tutorial Series
Created: 2024
Updated: Using latest Streamlit API syntax
"""

import streamlit as st

# Configure the main page
st.set_page_config(
    page_title="Advanced Multi-Page Navigation Tutorial",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main header
st.title("🧭 Advanced Multi-Page Navigation in Streamlit")
st.markdown("**Master three powerful navigation methods for professional Streamlit applications**")

# Introduction section
st.markdown("""
Welcome to the comprehensive tutorial on advanced multi-page navigation in Streamlit! 
This tutorial series demonstrates three distinct navigation approaches, each optimized 
for different use cases and application architectures.
""")

# Navigation methods overview
st.header("🎯 Navigation Methods Overview")

# Create three columns for the main navigation methods
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🔗 Method 1: Page Links")
    st.markdown("""
    **Static Navigation with `st.page_link`**
    
    Perfect for traditional website-style navigation:
    - ✅ Simple and straightforward
    - ✅ Clean link presentation  
    - ✅ External URL support
    - ✅ Accessible navigation
    
    **Best For:**
    - Documentation sites
    - Static content apps
    - External resource linking
    """)
    
    if st.button("🚀 Launch Page Links Demo", key="launch_page_links", use_container_width=True):
        st.switch_page("app1_page_links.py")

with col2:
    st.subheader("🔄 Method 2: Switch Page")
    st.markdown("""
    **Programmatic Navigation with `st.switch_page`**
    
    Ideal for conditional logic and form flows:
    - ✅ Conditional navigation
    - ✅ Form validation flows
    - ✅ Role-based routing
    - ✅ Error handling
    
    **Best For:**
    - Multi-step forms
    - User authentication
    - Data validation flows
    """)
    
    if st.button("🚀 Launch Switch Page Demo", key="launch_switch_page", use_container_width=True):
        st.switch_page("app2_switch_page.py")

with col3:
    st.subheader("🎛️ Method 3: Custom Selector")
    st.markdown("""
    **Dynamic Navigation with Widgets**
    
    Advanced single-page app experience:
    - ✅ Instant content switching
    - ✅ Complex state management
    - ✅ Multiple UI patterns
    - ✅ SPA-like experience
    
    **Best For:**
    - Admin dashboards
    - Data exploration tools
    - Interactive applications
    """)
    
    if st.button("🚀 Launch Custom Selector Demo", key="launch_custom_selector", use_container_width=True):
        st.switch_page("app3_custom_selector.py")

st.markdown("---")

# Quick comparison table
st.header("📊 Method Comparison")

comparison_data = {
    "Feature": [
        "Implementation Complexity",
        "Page Reload Required", 
        "State Management",
        "Conditional Logic Support",
        "Form Integration",
        "Performance",
        "User Experience"
    ],
    "🔗 Page Links": [
        "Simple",
        "Yes",
        "Basic", 
        "Limited",
        "Basic",
        "Good",
        "Traditional"
    ],
    "🔄 Switch Page": [
        "Moderate",
        "Yes", 
        "Advanced",
        "Excellent",
        "Excellent", 
        "Good",
        "Guided"
    ],
    "🎛️ Custom Selector": [
        "Complex",
        "No",
        "Advanced", 
        "Excellent",
        "Good",
        "Excellent",
        "Modern SPA"
    ]
}

st.table(comparison_data)

st.markdown("---")

# Tutorial features section
st.header("✨ Tutorial Features")

feature_col1, feature_col2 = st.columns(2)

with feature_col1:
    st.markdown("""
    ### 🎓 What You'll Learn
    
    - **Latest Streamlit API** - Using current syntax and best practices
    - **Navigation Patterns** - When and how to use each method
    - **State Management** - Proper session state handling
    - **User Experience** - Creating intuitive navigation flows
    - **Real-World Examples** - Production-ready code patterns
    - **Performance Tips** - Optimizing your navigation
    """)

with feature_col2:
    st.markdown("""
    ### 🛠️ Interactive Demos
    
    - **Live Examples** - See each method in action
    - **Code Samples** - Copy-paste ready implementations
    - **Best Practices** - Professional development guidelines
    - **Error Handling** - Robust navigation patterns
    - **Accessibility** - Screen reader friendly navigation
    - **Mobile Ready** - Responsive design considerations
    """)

st.markdown("---")

# Quick start section
st.header("🚀 Quick Start Guide")

tab1, tab2, tab3 = st.tabs(["📋 Prerequisites", "⚡ Running the Demos", "📁 Project Structure"])

with tab1:
    st.markdown("""
    ### System Requirements
    
    ```bash
    # Install required packages
    pip install streamlit pandas numpy plotly
    ```
    
    ### Recommended Setup
    - Python 3.8 or higher
    - Latest version of Streamlit
    - Modern web browser
    - VS Code (optional, for development)
    """)

with tab2:
    st.markdown("""
    ### Running Individual Demos
    
    ```bash
    # Method 1: Page Links
    streamlit run app1_page_links.py
    
    # Method 2: Switch Page  
    streamlit run app2_switch_page.py
    
    # Method 3: Custom Selector
    streamlit run app3_custom_selector.py
    ```
    
    ### Running This Overview
    
    ```bash
    # Main tutorial entry point
    streamlit run 17_advanced_multi_page.py
    ```
    """)

with tab3:
    st.code("""
Part_17_Advanced_MultiPage_Nav/
├── README.md                    # Comprehensive documentation
├── 17_advanced_multi_page.py    # This overview file
├── app1_page_links.py          # Method 1: Page Links Demo
├── app2_switch_page.py         # Method 2: Switch Page Demo  
├── app3_custom_selector.py     # Method 3: Custom Selector Demo
└── pages/
    ├── about.py                # About page (shared)
    ├── contact.py              # Contact page (shared)  
    └── dashboard.py            # Dashboard page (shared)
    """, language="text")

st.markdown("---")

# Sidebar navigation
st.sidebar.title("🧭 Tutorial Navigation")
st.sidebar.markdown("---")

st.sidebar.subheader("📚 Main Demos")
if st.sidebar.button("🔗 Page Links Demo", key="sidebar_page_links"):
    st.switch_page("app1_page_links.py")

if st.sidebar.button("🔄 Switch Page Demo", key="sidebar_switch_page"):
    st.switch_page("app2_switch_page.py")
    
if st.sidebar.button("🎛️ Custom Selector Demo", key="sidebar_custom_selector"):
    st.switch_page("app3_custom_selector.py")

st.sidebar.markdown("---")

st.sidebar.subheader("📄 Additional Pages")
if st.sidebar.button("ℹ️ About", key="sidebar_about"):
    st.switch_page("pages/about.py")
    
if st.sidebar.button("📊 Dashboard", key="sidebar_dashboard"):
    st.switch_page("pages/dashboard.py")
    
if st.sidebar.button("📞 Contact", key="sidebar_contact"):
    st.switch_page("pages/contact.py")

st.sidebar.markdown("---")

# Sidebar tips
st.sidebar.subheader("💡 Pro Tips")
st.sidebar.info("""
**Choosing the Right Method:**

🔗 **Page Links** for simple, static navigation

🔄 **Switch Page** for conditional logic and forms

🎛️ **Custom Selector** for complex, interactive apps
""")

st.sidebar.success("""
**Tutorial Goal:** Learn when and how to use each navigation method effectively in your Streamlit applications.
""")

# Footer section
st.markdown("---")

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("""
    ### 📚 Resources
    - [Streamlit Documentation](https://docs.streamlit.io)
    - [API Reference](https://docs.streamlit.io/develop/api-reference)
    - [Community Gallery](https://streamlit.io/gallery)
    """)

with footer_col2:
    st.markdown("""
    ### 🔗 Quick Links  
    - [st.page_link Docs](https://docs.streamlit.io/develop/api-reference/navigation/page_link)
    - [st.switch_page Docs](https://docs.streamlit.io/develop/api-reference/navigation/switch_page)
    - [Multi-page Apps Guide](https://docs.streamlit.io/develop/concepts/multipage-apps)
    """)

with footer_col3:
    st.markdown("""
    ### 🤝 Community
    - [Discussion Forum](https://discuss.streamlit.io)
    - [GitHub Issues](https://github.com/streamlit/streamlit/issues)
    - [Twitter @streamlit](https://twitter.com/streamlit)
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>Advanced Multi-Page Navigation Tutorial</strong></p>
    <p>Created with ❤️ for the Streamlit community | Using latest Streamlit API syntax</p>
</div>
""", unsafe_allow_html=True)