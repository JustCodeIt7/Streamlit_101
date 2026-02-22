import streamlit as st
import numpy as np

st.set_page_config(page_title="st.container Tutorial", layout="wide")
st.title("📦 `st.container` — Interactive Tutorial")
st.caption("Explore all major features of st.container through live examples.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "1️⃣ Basic Usage",
        "2️⃣ Border & Height",
        "3️⃣ Width Control",
        "4️⃣ Horizontal Layout",
        "5️⃣ Alignment & Gap",
    ]
)

# ── Tab 1: Basic Usage ──────────────────────────────────────────────────────
with tab1:
    st.header("Basic Usage")
    st.markdown(
        "Containers let you group elements and even **insert content out of order**."
    )

    col_code, col_demo = st.columns(2, gap="large")

    with col_code:
        st.subheader("🔤 `with` notation")
        st.code(
            """
with st.container():
    st.write("Inside the container")
    st.bar_chart(np.random.randn(20, 2))

st.write("Outside the container")
""",
            language="python",
        )

        st.subheader("🔤 Out-of-order insertion")
        st.code(
            """
c = st.container(border=True)
c.write("Written first...")
st.write("This appears BETWEEN")
c.write("...but added last!")
""",
            language="python",
        )

    with col_demo:
        st.subheader("▶ Live Demo")
        with st.container(border=True):
            st.write("✅ Inside the container")
            st.bar_chart(np.random.randn(20, 2))
        st.write("⬆ Outside the container")

        st.divider()
        c = st.container(border=True)
        c.write("🟢 Written first (via container object)...")
        st.info(
            "ℹ️ This `st.write` call appears **between** in the source, but renders in the middle."
        )
        c.write("🟢 ...added last, but grouped with the first!")

# ── Tab 2: Border & Height ──────────────────────────────────────────────────
with tab2:
    st.header("Border & Scrollable Height")

    col_code, col_demo = st.columns(2, gap="large")

    with col_code:
        st.subheader("🔤 Code")
        st.code(
            """
# border=True always shows border
with st.container(border=True):
    st.write("Bordered container")

# Fixed height → scrollable
with st.container(height=200, border=True):
    for i in range(20):
        st.write(f"Line {i+1}")
""",
            language="python",
        )
        st.info(
            "💡 `border=None` (default) auto-shows border only when height is fixed."
        )

    with col_demo:
        st.subheader("▶ Live Demo")
        with st.container(border=True):
            st.write("👆 Always bordered")

        st.write("")
        height_val = st.slider(
            "Set container height (px)", 100, 400, 200, key="h_slider"
        )
        with st.container(height=height_val, border=True):
            for i in range(25):
                st.write(f"📄 Line {i + 1}")

# ── Tab 3: Width Control ────────────────────────────────────────────────────
with tab3:
    st.header("Width Control")

    col_code, col_demo = st.columns(2, gap="large")

    with col_code:
        st.subheader("🔤 Code")
        st.code(
            """
# Stretch (default) — full parent width
with st.container(width="stretch", border=True):
    st.write("Stretch width")

# Content — shrinks to content
with st.container(width="content", border=True):
    st.write("Content width")

# Fixed pixels
with st.container(width=300, border=True):
    st.write("300px wide")
""",
            language="python",
        )

    with col_demo:
        st.subheader("▶ Live Demo")
        width_choice = st.radio(
            "Choose width mode",
            ["stretch", "content", "fixed (300px)"],
            horizontal=True,
        )
        w = (
            "stretch"
            if width_choice == "stretch"
            else ("content" if width_choice == "content" else 300)
        )
        with st.container(width=w, border=True):
            st.write(f"📐 Container with `width={w!r}`")
            st.write("Some content inside here.")

# ── Tab 4: Horizontal Layout ────────────────────────────────────────────────
with tab4:
    st.header("Horizontal Layout")

    col_code, col_demo = st.columns(2, gap="large")

    with col_code:
        st.subheader("🔤 Code")
        st.code(
            """
flex = st.container(
    horizontal=True,
    horizontal_alignment="distribute",
    border=True,
)
for i in range(4):
    flex.button(f"Button {i+1}")
""",
            language="python",
        )
        st.info("💡 Elements overflow to the next line if they don't fit.")

    with col_demo:
        st.subheader("▶ Live Demo")
        align = st.select_slider(
            "horizontal_alignment",
            options=["left", "center", "right", "distribute"],
            key="h_align",
        )
        n_btns = st.slider("Number of buttons", 1, 8, 4, key="n_btns")
        flex = st.container(horizontal=True, horizontal_alignment=align, border=True)
        for i in range(n_btns):
            flex.button(f"Btn {i + 1}", key=f"btn_{i}")

# ── Tab 5: Alignment & Gap ──────────────────────────────────────────────────
with tab5:
    st.header("Vertical Alignment & Gap")

    col_code, col_demo = st.columns(2, gap="large")

    with col_code:
        st.subheader("🔤 Code")
        st.code(
            """
with st.container(
    height=200,
    horizontal=True,
    vertical_alignment="center",
    gap="large",
    border=True,
):
    st.button("Short")
    st.text_area("Tall widget", height=100)
    st.button("Short again")
""",
            language="python",
        )

    with col_demo:
        st.subheader("▶ Live Demo")
        v_align = st.select_slider(
            "vertical_alignment", options=["top", "center", "bottom"], key="v_align"
        )
        gap_size = st.select_slider(
            "gap",
            options=[
                "None",
                "xxsmall",
                "xsmall",
                "small",
                "medium",
                "large",
                "xlarge",
                "xxlarge",
            ],
            value="small",
            key="gap_size",
        )
        gap_val = None if gap_size == "None" else gap_size
        with st.container(
            height=180,
            horizontal=True,
            vertical_alignment=v_align,
            gap=gap_val,
            border=True,
        ):
            st.button("Short")
            st.text_area("Tall", height=100, key="ta", label_visibility="collapsed")
            st.button("Short 2")

st.divider()
st.caption(
    "📚 Docs: https://docs.streamlit.io/develop/api-reference/layout/st.container"
)
