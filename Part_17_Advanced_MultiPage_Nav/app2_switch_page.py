"""
Streamlit Tutorial: Advanced Multi-Page Navigation - Method 2: st.switch_page
==============================================================================

This app demonstrates navigation using st.switch_page, which programmatically
switches to different pages based on user interactions and application logic.

Key Features:
- Programmatic page switching
- Conditional navigation logic
- Button-triggered navigation
- Dynamic page routing
- State-based navigation decisions

Usage:
Run this app with: streamlit run app2_switch_page.py
"""

import streamlit as st
import time

# Configure the page
st.set_page_config(
    page_title="Switch Page Demo",
    page_icon="🔄",
    layout="wide"
)

# Initialize session state
if 'user_role' not in st.session_state:
    st.session_state.user_role = None
    
if 'navigation_history' not in st.session_state:
    st.session_state.navigation_history = []
    
if 'form_completed' not in st.session_state:
    st.session_state.form_completed = False

# Main content
st.title("🔄 Navigation Method 2: Switch Page")
st.write("This demo shows how to use `st.switch_page` for programmatic navigation.")

# Create layout with tabs for different demo scenarios
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Basic Demo", 
    "🔐 Conditional Logic", 
    "📝 Form Flow", 
    "💻 Code Examples"
])

with tab1:
    st.header("Basic st.switch_page Usage")
    st.markdown("""
    The simplest use of `st.switch_page` is to navigate directly to another page
    when a user clicks a button or triggers an action.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Direct Navigation")
        st.write("Click any button to immediately switch to that page:")
        
        if st.button("🏠 Go to Home Page", key="home_btn"):
            st.switch_page("app1_page_links.py")
            
        if st.button("🎛️ Go to Custom Selector", key="selector_btn"):
            st.switch_page("app3_custom_selector.py")
            
        if st.button("📊 Go to Dashboard", key="dashboard_btn"):
            st.switch_page("pages/dashboard.py")
    
    with col2:
        st.subheader("Navigation with Delay")
        st.write("Demonstrate navigation with processing time:")
        
        if st.button("🚀 Process & Navigate", key="process_btn"):
            with st.spinner("Processing your request..."):
                time.sleep(2)  # Simulate processing
                st.success("Processing complete! Redirecting...")
                time.sleep(1)
                st.switch_page("pages/results.py")

with tab2:
    st.header("Conditional Navigation Logic")
    st.markdown("""
    `st.switch_page` shines when you need to make navigation decisions based on
    user state, form validation, or business logic.
    """)
    
    # Role-based navigation demo
    st.subheader("🔐 Role-Based Navigation")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("**Step 1:** Select your role")
        role = st.selectbox(
            "Choose your role:",
            ["None", "Admin", "User", "Guest"],
            key="role_selector"
        )
        
        if st.button("🎯 Navigate Based on Role", key="role_nav_btn"):
            if role == "Admin":
                st.session_state.user_role = "admin"
                st.success("Admin access granted! Redirecting to admin panel...")
                time.sleep(1)
                st.switch_page("pages/admin.py")
            elif role == "User":
                st.session_state.user_role = "user"
                st.info("User access granted! Redirecting to dashboard...")
                time.sleep(1)
                st.switch_page("pages/dashboard.py")
            elif role == "Guest":
                st.session_state.user_role = "guest"
                st.warning("Guest access - redirecting to public page...")
                time.sleep(1)
                st.switch_page("pages/public.py")
            else:
                st.error("Please select a valid role!")
    
    with col2:
        st.write("**Navigation Logic:**")
        st.code("""
def navigate_by_role(role):
    if role == "Admin":
        st.switch_page("pages/admin.py")
    elif role == "User":
        st.switch_page("pages/dashboard.py")
    elif role == "Guest":
        st.switch_page("pages/public.py")
    else:
        st.error("Invalid role!")
        """, language="python")

with tab3:
    st.header("Multi-Step Form Flow")
    st.markdown("""
    Use `st.switch_page` to create complex form flows where users progress
    through multiple pages based on their inputs and validation results.
    """)
    
    # Form validation demo
    st.subheader("📝 Form Validation Flow")
    
    with st.form("demo_form"):
        name = st.text_input("Full Name", placeholder="Enter your name")
        email = st.text_input("Email", placeholder="Enter your email")
        age = st.number_input("Age", min_value=1, max_value=120, value=25)
        terms = st.checkbox("I agree to the terms and conditions")
        
        submitted = st.form_submit_button("Submit & Navigate")
        
        if submitted:
            # Validation logic
            errors = []
            if not name.strip():
                errors.append("Name is required")
            if not email.strip() or "@" not in email:
                errors.append("Valid email is required")
            if not terms:
                errors.append("You must agree to terms and conditions")
            
            if errors:
                for error in errors:
                    st.error(f"❌ {error}")
            else:
                # Store form data in session state
                st.session_state.form_data = {
                    "name": name,
                    "email": email,
                    "age": age
                }
                st.session_state.form_completed = True
                
                st.success("✅ Form validated successfully!")
                st.info("Redirecting to confirmation page...")
                time.sleep(2)
                st.switch_page("pages/confirmation.py")

with tab4:
    st.header("💻 Code Examples & Best Practices")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Basic Usage")
        st.code("""
# Simple page switch
if st.button("Go to Page"):
    st.switch_page("target_page.py")

# With user feedback
if st.button("Submit"):
    st.success("Submitted!")
    st.switch_page("success.py")
        """, language="python")
        
        st.subheader("Conditional Navigation")
        st.code("""
# Role-based navigation
if user_role == "admin":
    st.switch_page("admin.py")
elif user_role == "user":
    st.switch_page("dashboard.py")
else:
    st.switch_page("login.py")
        """, language="python")
    
    with col2:
        st.subheader("Form Flow Navigation")
        st.code("""
# Multi-step form
if form_step == 1:
    # Collect basic info
    if st.button("Next"):
        st.switch_page("step2.py")
elif form_step == 2:
    # Collect additional info
    if st.button("Submit"):
        st.switch_page("confirmation.py")
        """, language="python")
        
        st.subheader("Error Handling")
        st.code("""
try:
    # Validate data
    if validate_input(data):
        st.switch_page("success.py")
    else:
        st.error("Validation failed")
except Exception as e:
    st.error(f"Error: {e}")
    st.switch_page("error.py")
        """, language="python")

# Sidebar with navigation options
st.sidebar.title("🔄 Switch Page Demo")
st.sidebar.markdown("---")

st.sidebar.subheader("🎯 Quick Navigation")
if st.sidebar.button("🏠 Home (Page Links)", key="sidebar_home"):
    st.switch_page("app1_page_links.py")
    
if st.sidebar.button("🎛️ Custom Selector", key="sidebar_custom"):
    st.switch_page("app3_custom_selector.py")

st.sidebar.markdown("---")

# Show navigation history
st.sidebar.subheader("📋 Session Info")
if st.session_state.user_role:
    st.sidebar.write(f"**Current Role:** {st.session_state.user_role}")
if st.session_state.form_completed:
    st.sidebar.write("**Form Status:** ✅ Completed")
else:
    st.sidebar.write("**Form Status:** ⏳ Pending")

# Footer with tutorial information
st.markdown("---")
st.markdown("""
### 📝 Tutorial Notes:

**Advantages of st.switch_page:**
- ✅ Programmatic control over navigation
- ✅ Perfect for conditional logic and form flows
- ✅ Can be triggered by any user interaction
- ✅ Integrates well with validation and business logic
- ✅ Supports complex multi-step processes

**Best Practices:**
- 🎯 Provide user feedback before switching
- 🔄 Validate inputs before navigation
- 💾 Store necessary data in session state
- ⚠️ Handle errors gracefully
- 🎨 Show loading states for better UX

**Use Cases:**
- Form submission flows
- Role-based access control
- Wizard-style interfaces
- Error handling and redirects
- Dynamic routing based on data

**Next:** Check out `app3_custom_selector.py` for advanced custom navigation!
""")