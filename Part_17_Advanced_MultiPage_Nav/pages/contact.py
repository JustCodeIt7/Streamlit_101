"""
Contact Page for Navigation Demos
=================================
"""

import streamlit as st

st.set_page_config(page_title="Contact", page_icon="📞")

st.title("📞 Contact Us")

st.markdown("""
We'd love to hear from you! Whether you have questions about the navigation tutorial,
suggestions for improvements, or need help implementing these patterns in your own apps.
""")

# Contact form
st.subheader("💬 Send us a Message")

with st.form("contact_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("Full Name *", placeholder="Enter your name")
        email = st.text_input("Email Address *", placeholder="your.email@example.com")
    
    with col2:
        company = st.text_input("Company/Organization", placeholder="Optional")
        subject = st.selectbox("Subject", [
            "General Question",
            "Tutorial Feedback", 
            "Bug Report",
            "Feature Request",
            "Implementation Help",
            "Other"
        ])
    
    message = st.text_area("Message *", 
                          placeholder="Tell us how we can help you...",
                          height=150)
    
    # Newsletter subscription
    newsletter = st.checkbox("📧 Subscribe to updates about new tutorials")
    
    submitted = st.form_submit_button("📤 Send Message", use_container_width=True)
    
    if submitted:
        # Validation
        errors = []
        if not name.strip():
            errors.append("Name is required")
        if not email.strip() or "@" not in email:
            errors.append("Valid email address is required")
        if not message.strip():
            errors.append("Message is required")
        
        if errors:
            for error in errors:
                st.error(f"❌ {error}")
        else:
            st.success("✅ Thank you! Your message has been sent successfully.")
            st.balloons()
            
            # Show submitted information
            with st.expander("📄 Message Summary"):
                st.write(f"**Name:** {name}")
                st.write(f"**Email:** {email}")
                if company:
                    st.write(f"**Company:** {company}")
                st.write(f"**Subject:** {subject}")
                st.write(f"**Newsletter:** {'Yes' if newsletter else 'No'}")
                st.write(f"**Message:** {message}")

st.markdown("---")

# Contact information
col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Tutorial Information")
    st.markdown("""
    **Navigation Methods Covered:**
    - 🔗 Page Links (`st.page_link`)
    - 🔄 Switch Page (`st.switch_page`) 
    - 🎛️ Custom Selectors (widgets)
    
    **Tutorial Features:**
    - ✅ Latest Streamlit syntax
    - ✅ Interactive demos
    - ✅ Best practices
    - ✅ Real-world examples
    """)

with col2:
    st.subheader("🔗 Quick Links")
    st.markdown("""
    **Resources:**
    - 📚 [Streamlit Docs](https://docs.streamlit.io)
    - 💬 [Community Forum](https://discuss.streamlit.io)
    - 🎥 [Video Gallery](https://streamlit.io/gallery)
    - 📖 [API Reference](https://docs.streamlit.io/develop/api-reference)
    """)

st.markdown("---")

# FAQ Section
st.subheader("❓ Frequently Asked Questions")

with st.expander("How do I choose the right navigation method?"):
    st.markdown("""
    **Choose based on your app's needs:**
    
    - **Page Links** (`st.page_link`): Best for simple, static navigation like traditional websites
    - **Switch Page** (`st.switch_page`): Best for conditional logic, form flows, and validation-driven navigation
    - **Custom Selectors**: Best for complex single-page applications with dynamic content
    """)

with st.expander("Can I combine multiple navigation methods?"):
    st.markdown("""
    Yes! You can mix and match navigation methods in the same app:
    - Use page links for main navigation
    - Use switch_page for form submissions
    - Use custom selectors for sub-sections
    """)

with st.expander("Are these examples production-ready?"):
    st.markdown("""
    These examples demonstrate the core concepts and can be adapted for production use:
    - Add proper error handling
    - Implement security measures
    - Add data persistence
    - Optimize performance for larger datasets
    """)

# Navigation back to main apps
st.markdown("---")
st.subheader("🧭 Return to Navigation Demos")

col1, col2, col3 = st.columns(3)

with col1:
    st.page_link("../app1_page_links.py", label="🔗 Page Links Demo", icon="🔗")

with col2:
    st.page_link("../app2_switch_page.py", label="🔄 Switch Page Demo", icon="🔄")

with col3:
    st.page_link("../app3_custom_selector.py", label="🎛️ Custom Selector Demo", icon="🎛️")

# Additional navigation
st.markdown("---")
st.page_link("../app1_page_links.py", label="← Back to Home", icon="🏠")