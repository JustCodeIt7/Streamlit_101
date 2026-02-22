import streamlit as st
import numpy as np

st.set_page_config(page_title="st.container Tutorial", layout="wide")
st.title("📦 `st.container` — Interactive Tutorial")
st.caption("Explore all major features of st.container through live examples.")

tabs = st.tabs(
    [
        "1️⃣ Basic Usage",
        "2️⃣ Border & Height",
        "3️⃣ Width Control",
        "4️⃣ Horizontal Layout",
        "5️⃣ Alignment & Gap",
    ]
)

# ── Tab 1: Basic Usage ──────────────────────────────────────────────────────
with tabs[0]:
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("🔤 Code")
        st.code("""with st.container(border=True):
    st.write("Inside the container")
    st.bar_chart(np.random.randn(20, 2))

c = st.container(border=True)
c.write("First...")
st.write("Between")
c.write("Last!")""")
    with col2:
        st.subheader("▶ Live Demo")
        with st.container(border=True):
            st.write("✅ Inside the container")
            st.bar_chart(np.random.randn(20, 2))

        c = st.container(border=True)
        c.write("🟢 First...")
        st.info("ℹ️ Between")
        c.write("🟢 Last!")

# ── Tab 2: Border & Height ──────────────────────────────────────────────────
with tabs[1]:
    h = st.slider("Set container height (px)", 100, 400, 200)
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("🔤 Code")
        st.code("""with st.container(border=True):
    st.write("Always bordered")

with st.container(height=h, border=True):
    for i in range(10):
        st.write(f"Line {i+1}")""")
    with col2:
        st.subheader("▶ Live Demo")
        with st.container(border=True):
            st.write("👆 Always bordered")
        with st.container(height=h, border=True):
            for i in range(10):
                st.write(f"📄 Line {i + 1}")

# ── Tab 3: Width Control ────────────────────────────────────────────────────
with tabs[2]:
    w_choice = st.radio(
        "Choose width mode", ["stretch", "content", "fixed (300px)"], horizontal=True
    )
    w = (
        "stretch"
        if w_choice == "stretch"
        else ("content" if w_choice == "content" else 300)
    )
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("🔤 Code")
        st.code("""with st.container(width=w, border=True):
    st.write(f"Container width={w!r}")""")
    with col2:
        st.subheader("▶ Live Demo")
        with st.container(width=w, border=True):
            st.write(f"📐 Container with `width={w!r}`")
            st.write("Some content inside here.")

# ── Tab 4: Horizontal Layout ────────────────────────────────────────────────
with tabs[3]:
    align = st.select_slider(
        "horizontal_alignment", ["left", "center", "right", "distribute"]
    )
    n = st.slider("Number of buttons", 1, 8, 4)
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("🔤 Code")
        st.code("""flex = st.container(horizontal=True, horizontal_alignment=align, border=True)
for i in range(n):
    flex.button(f"Btn {i+1}")""")
    with col2:
        st.subheader("▶ Live Demo")
        flex = st.container(horizontal=True, horizontal_alignment=align, border=True)
        for i in range(n):
            flex.button(f"Btn {i + 1}", key=f"btn_{i}")

# ── Tab 5: Alignment & Gap ──────────────────────────────────────────────────
with tabs[4]:
    v_align = st.select_slider("vertical_alignment", ["top", "center", "bottom"])
    gap = st.select_slider("gap", ["None", "small", "medium", "large"], value="small")
    gap_val = None if gap == "None" else gap
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("🔤 Code")
        st.code("""with st.container(height=180, horizontal=True, vertical_alignment=v_align, gap=gap_val, border=True):
    st.button("Short")
    st.text_area("Tall", height=100)
    st.button("Short 2")""")
    with col2:
        st.subheader("▶ Live Demo")
        with st.container(
            height=180,
            horizontal=True,
            vertical_alignment=v_align,
            gap=gap_val,
            border=True,
        ):
            st.button("Short")
            st.text_area(
                "Tall", height=100, label_visibility="collapsed", key="ta_tall"
            )
            st.button("Short 2")

st.divider()
st.caption(
    "📚 Docs: https://docs.streamlit.io/develop/api-reference/layout/st.container"
)
