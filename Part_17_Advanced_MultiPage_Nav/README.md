# 🧭 Advanced Multi-Page Navigation in Streamlit

## Tutorial Series Overview

This comprehensive tutorial demonstrates **three advanced methods** for implementing multi-page navigation in Streamlit applications, using the latest API features and best practices. Each method serves different use cases and provides unique advantages for building professional Streamlit applications.

## 📚 Navigation Methods Covered

### 🔗 Method 1: Page Links (`st.page_link`)
**File:** `app1_page_links.py`

Static navigation using clickable links - perfect for traditional website-style navigation.

**Key Features:**
- Clean, organized link presentation
- Support for both internal pages and external URLs
- Icons and labels for better UX
- Screen reader friendly (accessible)

**Best For:**
- Simple, static navigation menus
- Traditional website-style apps
- Documentation sites
- External resource linking

### 🔄 Method 2: Switch Page (`st.switch_page`)
**File:** `app2_switch_page.py`

Programmatic navigation with conditional logic - ideal for form flows and complex user journeys.

**Key Features:**
- Conditional navigation logic
- Button-triggered navigation
- Form validation and submission flows
- State-based routing decisions

**Best For:**
- Multi-step forms and wizards
- Conditional logic and validation
- Role-based access control
- Error handling and redirects
- Dynamic routing based on user data

### 🎛️ Method 3: Custom Selector Navigation
**File:** `app3_custom_selector.py`

Dynamic content switching using custom widgets - perfect for single-page applications.

**Key Features:**
- Multiple navigation patterns (dropdown, radio, buttons)
- Dynamic content without page reloads
- Advanced state management
- Single-page application (SPA) experience

**Best For:**
- Admin dashboards
- Data exploration tools
- Complex interactive applications
- Settings and configuration panels
- Apps requiring instant content switching

## 🚀 Quick Start

### Prerequisites
```bash
pip install streamlit pandas numpy plotly
```

### Running the Apps

1. **Page Links Demo:**
   ```bash
   streamlit run app1_page_links.py
   ```

2. **Switch Page Demo:**
   ```bash
   streamlit run app2_switch_page.py
   ```

3. **Custom Selector Demo:**
   ```bash
   streamlit run app3_custom_selector.py
   ```

## 📁 Project Structure

```
Part_17_Advanced_MultiPage_Nav/
├── README.md                    # This documentation
├── app1_page_links.py          # Method 1: Page Links Demo
├── app2_switch_page.py         # Method 2: Switch Page Demo  
├── app3_custom_selector.py     # Method 3: Custom Selector Demo
├── pages/
│   ├── about.py                # About page (linked from demos)
│   ├── contact.py              # Contact page (linked from demos)
│   └── dashboard.py            # Dashboard page (linked from demos)
└── 17_advanced_multi_page.py   # [Optional] Main entry point
```

## 🎯 Key Learning Objectives

### Understanding Navigation Patterns
- **Static vs Dynamic**: When to use fixed links vs programmatic switching
- **State Management**: How to maintain app state across navigation
- **User Experience**: Creating intuitive navigation flows

### Technical Implementation
- **Latest Streamlit API**: Using current syntax and best practices
- **Session State**: Proper state management techniques
- **Error Handling**: Graceful navigation error management
- **Performance**: Optimizing navigation for better user experience

### Real-World Applications
- **Form Flows**: Multi-step data collection
- **Dashboards**: Complex data visualization apps
- **Admin Panels**: Role-based navigation systems
- **Documentation**: Help and support systems

## 💻 Code Examples

### Basic Page Link
```python
# Simple internal page link
st.page_link("pages/about.py", label="About Us", icon="ℹ️")

# External link
st.page_link("https://streamlit.io", label="Streamlit Website")
```

### Conditional Page Switching
```python
# Role-based navigation
if user_role == "admin":
    st.switch_page("pages/admin.py")
elif user_role == "user":
    st.switch_page("pages/dashboard.py")
else:
    st.switch_page("pages/login.py")
```

### Custom Selector Navigation
```python
# Dynamic content switching
selected_section = st.selectbox("Navigate to:", ["Dashboard", "Analytics", "Settings"])

if selected_section == "Dashboard":
    show_dashboard()
elif selected_section == "Analytics":
    show_analytics()
else:
    show_settings()
```

## 🔧 Advanced Features Demonstrated

### 1. State Management
- Session state persistence across navigation
- Form data retention during multi-step flows
- User preference storage

### 2. Validation and Error Handling
- Form validation before navigation
- Graceful error handling
- User feedback and loading states

### 3. UI/UX Best Practices
- Loading indicators during transitions
- Clear navigation feedback
- Accessible navigation patterns
- Responsive design considerations

### 4. Performance Optimization
- Efficient data caching with `@st.cache_data`
- Optimized re-rendering
- Minimal state updates

## 📋 Comparison Matrix

| Feature | Page Links | Switch Page | Custom Selector |
|---------|------------|-------------|-----------------|
| **Implementation** | Simple | Moderate | Complex |
| **Page Reloads** | Yes | Yes | No |
| **State Management** | Basic | Advanced | Advanced |
| **Conditional Logic** | Limited | Excellent | Excellent |
| **Form Integration** | Basic | Excellent | Good |
| **Performance** | Good | Good | Excellent |
| **Use Case** | Static Sites | Form Flows | SPAs |

## 🎨 Customization Options

### Styling Navigation Elements
```python
# Custom button styling
if st.button("🎯 Custom Button", help="Tooltip text"):
    st.switch_page("target.py")

# Styled selectbox
option = st.selectbox(
    "Choose Section:",
    options=["Home", "About", "Contact"],
    format_func=lambda x: f"📍 {x}"
)
```

### Dynamic Navigation Menus
```python
# Role-based menu construction
menu_items = ["Home"]
if user_role == "admin":
    menu_items.extend(["Admin Panel", "User Management"])
if user_role in ["admin", "user"]:
    menu_items.append("Dashboard")

selected = st.selectbox("Navigate:", menu_items)
```

## 🔍 Troubleshooting

### Common Issues

1. **Page Not Found Errors**
   ```python
   # Ensure file paths are correct
   st.page_link("pages/about.py", label="About")  # ✅ Correct
   st.page_link("about.py", label="About")        # ❌ Wrong if in pages/
   ```

2. **State Loss During Navigation**
   ```python
   # Store important data in session state
   if 'user_data' not in st.session_state:
       st.session_state.user_data = {}
   ```

3. **Navigation Not Working**
   ```python
   # Check for typos in file names
   st.switch_page("pages/dashboard.py")  # ✅ File exists
   st.switch_page("pages/dashbord.py")   # ❌ Typo
   ```

## 🚀 Production Deployment

### Best Practices
- **Error Handling**: Implement comprehensive error handling
- **Security**: Validate user inputs and permissions
- **Performance**: Use caching for data-heavy operations
- **Monitoring**: Add logging for navigation events

### Deployment Checklist
- [ ] All page files exist and are accessible
- [ ] Navigation paths are correct
- [ ] Session state is properly managed
- [ ] Error handling is implemented
- [ ] User feedback is provided
- [ ] Performance is optimized

## 📚 Additional Resources

### Streamlit Documentation
- [st.page_link API Reference](https://docs.streamlit.io/develop/api-reference/navigation/page_link)
- [st.switch_page API Reference](https://docs.streamlit.io/develop/api-reference/navigation/switch_page)
- [Multi-page Apps Guide](https://docs.streamlit.io/develop/concepts/multipage-apps)

### Community Resources
- [Streamlit Gallery](https://streamlit.io/gallery)
- [Community Forum](https://discuss.streamlit.io)
- [GitHub Examples](https://github.com/streamlit/streamlit/tree/develop/lib/streamlit)

## 🤝 Contributing

Feel free to:
- Report issues or bugs
- Suggest improvements
- Submit pull requests
- Share your own navigation patterns

## 📄 License

This tutorial is provided as educational content. Feel free to use and modify the code for your own projects.

---

**Happy Streamlit Development! 🎈**

*Created with ❤️ for the Streamlit community*