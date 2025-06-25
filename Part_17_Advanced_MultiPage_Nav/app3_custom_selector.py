"""
Streamlit Tutorial: Advanced Multi-Page Navigation - Method 3: Custom Selector
===============================================================================

This app demonstrates navigation using custom selectors like st.selectbox, st.radio,
and other widgets to create dynamic, interactive navigation experiences.

Key Features:
- Custom navigation widgets (selectbox, radio, sidebar options)
- Dynamic page content without actual page switching
- State-based navigation simulation
- Advanced UI patterns for navigation
- Integrated navigation with content management

Usage:
Run this app with: streamlit run app3_custom_selector.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

# Configure the page
st.set_page_config(
    page_title="Custom Selector Demo",
    page_icon="🎛️",
    layout="wide"
)

# Initialize session state
if 'current_section' not in st.session_state:
    st.session_state.current_section = "Dashboard"
    
if 'user_preferences' not in st.session_state:
    st.session_state.user_preferences = {
        "theme": "Light",
        "language": "English",
        "notifications": True
    }
    
if 'navigation_method' not in st.session_state:
    st.session_state.navigation_method = "Dropdown Menu"

# Generate sample data
@st.cache_data
def generate_sample_data():
    """Generate sample data for demonstrations"""
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    data = {
        'date': dates,
        'sales': np.random.randint(100, 1000, len(dates)),
        'users': np.random.randint(50, 500, len(dates)),
        'revenue': np.random.randint(1000, 10000, len(dates))
    }
    return pd.DataFrame(data)

# Page content functions
def show_dashboard():
    """Dashboard section content"""
    st.header("📊 Dashboard")
    
    # Generate and display sample data
    df = generate_sample_data()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Sales", f"{df['sales'].sum():,}", 
                 delta=f"{df['sales'].tail(30).sum() - df['sales'].head(30).sum():,}")
    
    with col2:
        st.metric("Active Users", f"{df['users'].sum():,}", 
                 delta=f"{df['users'].tail(30).sum() - df['users'].head(30).sum():,}")
    
    with col3:
        st.metric("Revenue", f"${df['revenue'].sum():,}", 
                 delta=f"${df['revenue'].tail(30).sum() - df['revenue'].head(30).sum():,}")
    
    # Chart
    chart_type = st.selectbox("Chart Type", ["Line", "Bar", "Area"], key="chart_selector")
    
    if chart_type == "Line":
        fig = px.line(df.tail(90), x='date', y='sales', title='Sales Trend (Last 90 Days)')
    elif chart_type == "Bar":
        monthly_data = df.groupby(df['date'].dt.month)['sales'].sum().reset_index()
        fig = px.bar(monthly_data, x='date', y='sales', title='Monthly Sales')
    else:
        fig = px.area(df.tail(90), x='date', y='sales', title='Sales Area Chart')
    
    st.plotly_chart(fig, use_container_width=True)

def show_analytics():
    """Analytics section content"""
    st.header("📈 Analytics")
    
    tab1, tab2, tab3 = st.tabs(["User Behavior", "Performance", "Reports"])
    
    with tab1:
        st.subheader("User Behavior Analysis")
        
        # Sample user behavior data
        behavior_data = {
            'Action': ['Page Views', 'Clicks', 'Downloads', 'Signups', 'Purchases'],
            'Count': [15420, 8930, 2340, 890, 234],
            'Change': ['+12%', '+8%', '-3%', '+15%', '+22%']
        }
        
        df_behavior = pd.DataFrame(behavior_data)
        st.dataframe(df_behavior, use_container_width=True)
        
        fig = px.bar(df_behavior, x='Action', y='Count', title='User Actions')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Performance Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Page Load Time", "1.2s", delta="-0.3s")
            st.metric("Bounce Rate", "32%", delta="-5%")
        
        with col2:
            st.metric("Session Duration", "4.5min", delta="+1.2min")
            st.metric("Conversion Rate", "3.2%", delta="+0.8%")
    
    with tab3:
        st.subheader("Generated Reports")
        
        report_type = st.radio("Select Report Type", 
                              ["Daily Summary", "Weekly Analysis", "Monthly Overview"])
        
        if st.button("Generate Report"):
            st.success(f"✅ {report_type} generated successfully!")
            st.download_button("📥 Download Report", 
                             data="Sample report content...", 
                             file_name=f"{report_type.lower().replace(' ', '_')}.txt")

def show_settings():
    """Settings section content"""
    st.header("⚙️ Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Appearance")
        
        theme = st.selectbox("Theme", ["Light", "Dark", "Auto"], 
                           index=["Light", "Dark", "Auto"].index(st.session_state.user_preferences["theme"]))
        
        language = st.selectbox("Language", ["English", "Spanish", "French", "German"],
                              index=["English", "Spanish", "French", "German"].index(st.session_state.user_preferences["language"]))
        
        st.subheader("Notifications")
        notifications = st.checkbox("Enable Notifications", 
                                   value=st.session_state.user_preferences["notifications"])
        
        email_alerts = st.checkbox("Email Alerts", value=True)
        push_notifications = st.checkbox("Push Notifications", value=False)
        
        if st.button("💾 Save Settings"):
            st.session_state.user_preferences.update({
                "theme": theme,
                "language": language,
                "notifications": notifications
            })
            st.success("Settings saved successfully!")
    
    with col2:
        st.subheader("Account Information")
        
        st.text_input("Username", value="demo_user", disabled=True)
        st.text_input("Email", value="demo@example.com")
        
        st.subheader("Security")
        
        if st.button("🔑 Change Password"):
            st.info("Password change functionality would be implemented here")
        
        if st.button("🔒 Enable 2FA"):
            st.info("Two-factor authentication setup would be implemented here")

def show_help():
    """Help section content"""
    st.header("❓ Help & Documentation")
    
    # FAQ Section
    st.subheader("🔍 Frequently Asked Questions")
    
    with st.expander("How do I navigate between sections?"):
        st.write("""
        You can navigate using several methods in this demo:
        - **Dropdown Menu**: Use the main navigation dropdown
        - **Sidebar Radio**: Use the radio buttons in the sidebar
        - **Tab Navigation**: Use tabs within sections
        - **Quick Actions**: Use the quick action buttons
        """)
    
    with st.expander("What navigation methods are demonstrated?"):
        st.write("""
        This app demonstrates custom navigation using:
        - `st.selectbox` for dropdown navigation
        - `st.radio` for radio button navigation
        - `st.tabs` for tabbed content
        - Dynamic content rendering based on selection
        - State management for navigation history
        """)
    
    with st.expander("How is this different from st.page_link and st.switch_page?"):
        st.write("""
        **Custom Selector Navigation:**
        - Single page with dynamic content
        - Instant switching (no page reload)
        - Complex state management
        - Integrated UI components
        
        **st.page_link:**
        - Multiple separate page files
        - Traditional link-based navigation
        - Simple and straightforward
        
        **st.switch_page:**
        - Programmatic page switching
        - Great for conditional logic
        - Form flow navigation
        """)
    
    # Contact form
    st.subheader("📞 Contact Support")
    
    with st.form("contact_form"):
        contact_name = st.text_input("Name")
        contact_email = st.text_input("Email")
        contact_message = st.text_area("Message")
        
        if st.form_submit_button("Send Message"):
            if contact_name and contact_email and contact_message:
                st.success("✅ Message sent successfully! We'll get back to you soon.")
            else:
                st.error("❌ Please fill in all fields.")

# Main app layout
st.title("🎛️ Navigation Method 3: Custom Selector")
st.markdown("**Dynamic navigation using custom widgets and state management**")

# Navigation method selector
st.subheader("🎯 Choose Your Navigation Style")

nav_col1, nav_col2 = st.columns([2, 1])

with nav_col1:
    navigation_method = st.selectbox(
        "Select Navigation Method:",
        ["Dropdown Menu", "Sidebar Radio", "Horizontal Buttons", "Quick Actions"],
        key="nav_method_selector"
    )
    st.session_state.navigation_method = navigation_method

with nav_col2:
    st.info(f"**Current Method:** {navigation_method}")

st.markdown("---")

# Different navigation implementations based on selection
if navigation_method == "Dropdown Menu":
    # Main dropdown navigation
    selected_section = st.selectbox(
        "🧭 Navigate to Section:",
        ["Dashboard", "Analytics", "Settings", "Help"],
        index=["Dashboard", "Analytics", "Settings", "Help"].index(st.session_state.current_section),
        key="main_dropdown"
    )
    st.session_state.current_section = selected_section

elif navigation_method == "Sidebar Radio":
    # Sidebar radio navigation
    st.sidebar.title("🎛️ Navigation")
    selected_section = st.sidebar.radio(
        "Choose Section:",
        ["Dashboard", "Analytics", "Settings", "Help"],
        index=["Dashboard", "Analytics", "Settings", "Help"].index(st.session_state.current_section),
        key="sidebar_radio"
    )
    st.session_state.current_section = selected_section

elif navigation_method == "Horizontal Buttons":
    # Horizontal button navigation
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 Dashboard", key="btn_dashboard"):
            st.session_state.current_section = "Dashboard"
    
    with col2:
        if st.button("📈 Analytics", key="btn_analytics"):
            st.session_state.current_section = "Analytics"
    
    with col3:
        if st.button("⚙️ Settings", key="btn_settings"):
            st.session_state.current_section = "Settings"
    
    with col4:
        if st.button("❓ Help", key="btn_help"):
            st.session_state.current_section = "Help"

else:  # Quick Actions
    # Quick action navigation with icons
    st.markdown("**Quick Actions:**")
    
    action_col1, action_col2, action_col3, action_col4, action_col5 = st.columns([1, 1, 1, 1, 1])
    
    with action_col1:
        if st.button("📊", help="Dashboard", key="quick_dashboard"):
            st.session_state.current_section = "Dashboard"
    
    with action_col2:
        if st.button("📈", help="Analytics", key="quick_analytics"):
            st.session_state.current_section = "Analytics"
    
    with action_col3:
        if st.button("⚙️", help="Settings", key="quick_settings"):
            st.session_state.current_section = "Settings"
    
    with action_col4:
        if st.button("❓", help="Help", key="quick_help"):
            st.session_state.current_section = "Help"
    
    with action_col5:
        if st.button("🔄", help="Refresh", key="quick_refresh"):
            st.rerun()

# Display current section indicator
st.markdown(f"### 🎯 Current Section: **{st.session_state.current_section}**")
st.markdown("---")

# Render content based on current section
if st.session_state.current_section == "Dashboard":
    show_dashboard()
elif st.session_state.current_section == "Analytics":
    show_analytics()
elif st.session_state.current_section == "Settings":
    show_settings()
elif st.session_state.current_section == "Help":
    show_help()

# Sidebar information (always visible)
if navigation_method != "Sidebar Radio":
    st.sidebar.title("🎛️ App Info")
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("📊 Current Status")
    st.sidebar.write(f"**Section:** {st.session_state.current_section}")
    st.sidebar.write(f"**Navigation:** {navigation_method}")
    st.sidebar.write(f"**Theme:** {st.session_state.user_preferences['theme']}")
    
    st.sidebar.markdown("---")
    
    # Quick navigation to other apps
    st.sidebar.subheader("🔗 Other Navigation Demos")
    if st.sidebar.button("🔗 Page Links Demo"):
        st.switch_page("app1_page_links.py")
    
    if st.sidebar.button("🔄 Switch Page Demo"):
        st.switch_page("app2_switch_page.py")

# Footer with tutorial information
st.markdown("---")
st.markdown("""
### 📝 Tutorial Notes:

**Advantages of Custom Selector Navigation:**
- ✅ Single page application (SPA) experience
- ✅ Instant content switching (no page reloads)
- ✅ Complex state management capabilities
- ✅ Highly customizable UI/UX
- ✅ Integrated navigation with content
- ✅ Multiple navigation patterns in one app

**Key Components Used:**
- `st.selectbox()` for dropdown navigation
- `st.radio()` for radio button navigation  
- `st.button()` for action-based navigation
- `st.session_state` for state management
- `st.tabs()` for sub-navigation
- Dynamic content rendering functions

**Best Practices:**
- 🎯 Store navigation state in session_state
- 🔄 Use functions to organize section content
- 🎨 Provide multiple navigation methods
- 💾 Persist user preferences
- 🚀 Optimize for performance with @st.cache_data

**Use Cases:**
- Admin dashboards
- Single-page applications
- Complex data exploration tools
- Multi-section forms
- Settings and configuration panels

**Comparison Summary:**
- **Page Links**: Best for simple, static navigation
- **Switch Page**: Best for conditional logic and form flows  
- **Custom Selector**: Best for complex, interactive applications
""")