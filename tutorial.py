# streamlit run tutorial.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(
    page_title="Streamlit Capability Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared data loader ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading Fashion-MNIST …")
def load_data():
    from tensorflow.keras.datasets import fashion_mnist
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    return X_train, y_train, X_test, y_test

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"
]
CLASS_ICONS = ["👕","👖","🧥","👗","🧥","👡","👔","👟","👜","👢"]

X_train, y_train, X_test, y_test = load_data()

st.title("Visualization Tool Exploration: Streamlit")
st.subheader("Using Fashion-MNIST as a playground for Streamlit features")
url = "https://www.streamlit.io"
st.markdown("""
    This tutorial is focused on **Streamlit** and its capabilities rather than the dataset. Fashion-MNIST is simply a
    convenient, interesting dataset to make the widgets feel meaningful. I am sure that Streamlit has other capabilities not encapsulated in this tutorial! When in doubt, I recommend referencing the [Streamlit Documentation](%s)."""%url)

st.info("I wanted to create a tutorial for Streamlit as its interactivity allows the audience to explore the visuals on their own terms. Adding in widgets like expanders can help make the product more author-driven or audience-driven depending on what the viewer prefers. I ultimatley think that Streamlit is a simple tool to create narrative digital stories. This tutorial also helps to merge both basic and genuine implementations of Streamlit's capabilities.")


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Streamlit Explorer")
    st.caption("A tutorial on Streamlit's interactive capabilities using Fashion-MNIST as our dataset.")
    st.divider()

    st.markdown("Like the sidebar feature?")
    with st.expander("See the code!"):
        st.code("""with st.sidebar:
    st.title("Streamlit Explorer")
    st.caption("A tutorial on Streamlit's interactive capabilities using Fashion-MNIST as our dataset.")
    st.divider()""")
        
    st.divider()

    st.markdown("""
    **Tabs in this app**
    1. Dataset Overview
    2. Widgets & Formatting
    3. Charts & Plotly
    4. Forms & Validation
    5. MNIST Exploration
    """)
    st.divider()
    st.info("""
**The Grammar of Graphics**  
Choices matter! Every visualization is built from components:
- Data  
- Marks (points, lines, bars)  
- Encodings (position, color, size)  
- Scales and coordinate systems  

Changing any one of these changes how the viewer interprets the data.

Good visualization design is not just aesthetic. The way that humans *perceive and reason* should be kept in mind.
""")

def fig_style(fig, axes=None):
    bg = "#ffffff"
    fg =  "#111111"
    fig.patch.set_facecolor(bg)
    if axes is not None:
        for ax in (axes if hasattr(axes, '__iter__') else [axes]):
            ax.set_facecolor(bg)
            ax.tick_params(colors=fg)
            ax.spines[:].set_color( "#ccc")
            ax.xaxis.label.set_color(fg)
            ax.yaxis.label.set_color(fg)
            ax.title.set_color(fg)
    return fig

# ══════════════════════════ TABS ══════════════════════════ #

tab1, tab2, tab3, tab4, tab5= st.tabs([
    "Overview & Dataset",
    "Widgets & Formatting",
    "Charts & Plotly",
    "Forms & Validation",
    "MNIST Exploration"
])

# ══════════════════════════ TAB 1 — OVERVIEW & DATASET ══════════════════════════ #

with tab1:
    st.header("Fashion-MNIST Overview")
    st.markdown("""
    Fashion-MNIST contains **70,000 grayscale 28×28 images** across 10 clothing classes.
    It's perfectly balanced (6,000 train / 1,000 test per class), which makes it ideal
    for demonstrating visualizations without worrying about class imbalance.
    """)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training images", "60,000")
    c2.metric("Test images", "10,000")
    c3.metric("Classes", "10")
    c4.metric("Image size", "28 × 28 px")

    st.markdown("I personally gained access to the dataset through TensorFlow/Keras:")
    st.code("""from tensorflow.keras.datasets import fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
""")

    st.divider()

    st.subheader("Sample Images")
    cols = st.columns(10)
    for i in range(10):
        idxs = np.where(y_train == i)[0]
        img = X_train[np.random.choice(idxs)]
        fig, ax = plt.subplots(figsize=(1.2, 1.2))
        fig_style(fig, ax)
        ax.imshow(img, cmap="gray")
        ax.axis("off")
        ax.set_title(f"{CLASS_ICONS[i]}\n{CLASS_NAMES[i]}", fontsize=6,
                     color= "black")
        cols[i].pyplot(fig, use_container_width=True)
        plt.close()

    st.caption("One random sample per class. Re-run the app to see different images.")

    st.info("A major focus of the course is balancing the complexity of the data with the simplicity of design. This tutorial exemplifies that by working with a truly complex dataset (many images, many classes, many pixels) yet keeping the design simple (widgets only appear with a purpose).")

# ══════════════════════════ TAB 2 — WIDGETS & FORMATTING ══════════════════════════ #

with tab2:
    st.title("Widget Showcase")
    st.markdown("""
    Streamlit's widget system turns Python variables into interactive controls.
    Every widget returns a value and triggers a **full script rerun** when changed.
    Here we demonstrate some of the most popular widget types, applied to image exploration.
    """)

    st.divider()

    # ── Slider ────────────────────────────────────────────────────────────────
    st.header("`st.slider`")

    n_images = st.slider("Number of sample images to show", 1, 20, 6)
    brightness = st.slider("Brightness adjustment", -100, 100, 0,
                               help="Adds a constant to all pixel values")
    st.code("""
n_images = st.slider("Number of sample images to show", 1, 20, 6)
brightness = st.slider("Brightness adjustment", -100, 100, 0,
                        help="Adds a constant to all pixel values")""")

    cls_pick = st.selectbox("Class to preview", CLASS_NAMES, key="slider_cls")
    cls_i = CLASS_NAMES.index(cls_pick)
    candidates = X_train[y_train == cls_i]
    means = candidates.reshape(len(candidates), -1).mean(axis=1)

    show_n = min(n_images, len(candidates))
    chosen = candidates[np.random.choice(len(candidates), show_n, replace=False)]
    cols_w = st.columns(show_n)
    for col, img in zip(cols_w, chosen):
        adj = np.clip(img.astype(int) + brightness, 0, 255).astype(np.uint8)
        fig, ax = plt.subplots(figsize=(1.5, 1.5))
        fig_style(fig, ax)
        ax.imshow(adj, cmap="gray", vmin=0, vmax=255)
        ax.axis("off")
        col.pyplot(fig, use_container_width=True)
        plt.close()

    st.divider()

    # ── Select widgets ────────────────────────────────────────────────────────
    st.header("`st.selectbox` & `st.multiselect`")
    col_c, col_d = st.columns(2)

    with col_c:
        cmap_sel = st.selectbox("Colormap (selectbox)", ["gray", "viridis", "plasma", "hot", "coolwarm"])
        st.code("cmap = st.selectbox('Colormap (selectbox)', ['gray', 'viridis', 'plasma', 'hot', 'coolwarm'])")

    with col_d:
        classes_multi = st.multiselect("Classes (multiselect)", CLASS_NAMES, default=CLASS_NAMES[:3])
        st.code("cls = st.multiselect('Classes (multiselect)', CLASS_NAMES, default=CLASS_NAMES[:3])")


    if classes_multi:
        fig_s, axes_s = plt.subplots(1, len(classes_multi),
                                     figsize=(2.5 * len(classes_multi), 2.5))
        fig_style(fig_s)
        if len(classes_multi) == 1:
            axes_s = [axes_s]
        for ax, cn in zip(axes_s, classes_multi):
            ci = CLASS_NAMES.index(cn)
            imgs = X_train[y_train == ci]
            display = imgs[np.random.randint(len(imgs))]
            ax.imshow(display, cmap=cmap_sel)
            ax.set_title(cn, color= "black", fontsize=8)
            ax.axis("off")
        plt.tight_layout()
        st.pyplot(fig_s)
        plt.close()
    else:
        st.info("Select at least one class above.")

    st.divider()

    # ── Text & Number input ────────────────────────────────────────────────────
    st.header("`st.text_input` & `st.number_input` & `st.color_picker`")

    label_text = st.text_input("Custom chart title", "My Fashion-MNIST Chart")
    n_bins = st.number_input("Histogram bins", min_value=5, max_value=100, value=30, step=5)
    st.code("""
title = st.text_input("Custom chart title", "My Fashion-MNIST Chart")
bins = st.number_input("Histogram bins", min_value=5, max_value=100, value=30, step=5)""")
    col_color, col_text = st.columns(2)
    with col_color:
        c1, c2, c3 = st.columns([1,2,1])
        with c2:
            accent_color = st.color_picker("Accent color", "#4c8bf5")
    with col_text:
        st.markdown("Select the color of the bars!")
        st.code("""accent_color = st.color_picker("Accent color", "#4c8bf5")""")

    means_all = X_train.reshape(len(X_train), -1).mean(axis=1)
    fig_h, ax_h = plt.subplots(figsize=(7, 3))
    fig_style(fig_h, ax_h)
    ax_h.hist(means_all, bins=int(n_bins), color=accent_color, alpha=0.85, edgecolor="white")
    ax_h.set_xlabel("Mean Pixel Intensity")
    ax_h.set_ylabel("Count")
    ax_h.set_title(label_text)
    plt.tight_layout()
    st.pyplot(fig_h)
    plt.close()

    st.info("""Color was an important aspect of data visualizations that we discussed this semester. Accessibility broadens the audience for visuals. Above, users can select which color they prefer for the plot.
 
Histograms seem objective, but the number of bins directly affects what patterns appear.

- Too few bins → oversimplified patterns  
- Too many bins → noisy, hard to interpret""")

    st.divider()

    # ── Checkbox ─────────────────────────────────────────────────
    st.header("`st.toggle`")

    invert_img = st.toggle("Invert image colors")
    st.code("""
invert_img = st.toggle("Invert image colors")
if invert_img:
    demo_img = 255 - demo_img
        """)

    demo_img = X_train[42]
    if invert_img:
        demo_img = 255 - demo_img
    fig_d, ax_d = plt.subplots(figsize=(2.5, 2.5))
    fig_style(fig_d, ax_d)
    ax_d.imshow(demo_img, cmap="gray")
    ax_d.axis("off")
    st.pyplot(fig_d)
    plt.close()

    st.info("I really appreciate how these interactive widgets give control to the viewer, almost merging the author-driven and reader-driven storytelling introduced in the Digital Scholarship Lab pre-class assignment.")

    st.divider()


    st.title("Layouts & Columns")
    st.markdown("""
    Streamlit provides several layout primitives for controlling how content is arranged.
    This tab demonstrates each one with concrete examples.
    """)

    st.divider()

    # ── Columns ───────────────────────────────────────────────────────────────
    st.header("`st.columns`")
    cols_demo = st.columns([1,1,1])

    demo_classes = CLASS_NAMES[:len([1,1,1])]
    for col, cn in zip(cols_demo, demo_classes):
        ci = CLASS_NAMES.index(cn)
        img = X_train[y_train == ci][0]
        col.subheader(cn)
        fig_c, ax_c = plt.subplots(figsize=(2, 2))
        fig_style(fig_c, ax_c)
        ax_c.imshow(img, cmap="gray"); ax_c.axis("off")
        col.pyplot(fig_c, use_container_width=True)
        plt.close()

    with st.expander("Columns code"):
        st.code("""
# Simple Columns
col1, col2, col3 = st.columns(3)
with col1:
    ...
                
# Example Above
cols_demo = st.columns([1,1,1])
demo_classes = CLASS_NAMES[:len([1,1,1])]
for col, cn in zip(cols_demo, demo_classes):
    ci = CLASS_NAMES.index(cn)
    img = X_train[y_train == ci][0]
    col.subheader(cn)
    fig_c, ax_c = plt.subplots(figsize=(2, 2))
    fig_style(fig_c, ax_c)
    ax_c.imshow(img, cmap="gray"); ax_c.axis("off")
    col.pyplot(fig_c, use_container_width=True)
    plt.close()
        """, language="python")

    st.divider()

    # ── Expander ──────────────────────────────────────────────────────────────
    st.header("`st.expander` — Collapsible Sections")
    st.markdown("`st.expander` hides content until the user clicks. Great for code snippets, details, or secondary information that would clutter the main view. You will see lots of expanders to show the code for each tool!")

    for cn in CLASS_NAMES[:4]:
        with st.expander(f"{CLASS_ICONS[CLASS_NAMES.index(cn)]} {cn}"):
            ci = CLASS_NAMES.index(cn)
            imgs = X_train[y_train == ci][:8]
            fig_e, axes_e = plt.subplots(1, 8, figsize=(12, 1.5))
            fig_style(fig_e, axes_e)
            for ax, im in zip(axes_e, imgs):
                ax.imshow(im, cmap="gray"); ax.axis("off")
            plt.tight_layout()
            st.pyplot(fig_e)
            plt.close()
            st.caption(f"8 random {cn} images. Mean pixel: {imgs.mean():.1f}")


    with st.expander("Expander Code"):
        st.code("""
# Simple Expander
with st.expander("Show more"):
    st.code("hidden code")
                
# Example Above
for cn in CLASS_NAMES[:4]:
     with st.expander(f"{CLASS_ICONS[CLASS_NAMES.index(cn)]} {cn} — click to expand"):
        ci = CLASS_NAMES.index(cn)
        imgs = X_train[y_train == ci][:8]
        fig_e, axes_e = plt.subplots(1, 8, figsize=(12, 1.5))
        fig_style(fig_e, axes_e)
        for ax, im in zip(axes_e, imgs):
            ax.imshow(im, cmap="gray"); ax.axis("off")
        plt.tight_layout()
        st.pyplot(fig_e)
        plt.close()
        st.caption(f"8 random {cn} images. Mean pixel: {imgs.mean():.1f}")
                """)
    st.divider()

    # ── Tabs ────────────────────────────────────────────────────
    st.header("`st.tabs`")
    st.markdown("Tabs can be seen at the top of this application as well as a nested version below! Here we show per-class stats in nested tabs.")

    inner_tabs = st.tabs(CLASS_NAMES[:5])
    for i, tab in enumerate(inner_tabs):
        with tab:
            imgs = X_train[y_train == i]
            c1, c2, c3 = st.columns(3)
            c1.metric("Count (train)", f"{len(imgs):,}")
            c2.metric("Mean pixel", f"{imgs.mean():.1f}")
            c3.metric("Std pixel", f"{imgs.std():.1f}")

    with st.expander("Tabs code"):
        st.code("""
# Top-of-Screen Tabs Example
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview & Dataset",
    "Widgets & Formatting",
    "Session State",
    "Charts & Plotly",
    "Forms & Validation",
    "MNIST Exploration"
])
                
with tab1:
     ...
                
# Example Above
inner_tabs = st.tabs(CLASS_NAMES[:5])
for i, tab in enumerate(inner_tabs):
    with tab:
        imgs = X_train[y_train == i]
        c1, c2, c3 = st.columns(3)
        c1.metric("Count (train)", f"{len(imgs):,}")
        c2.metric("Mean pixel", f"{imgs.mean():.1f}")
        c3.metric("Std pixel", f"{imgs.std():.1f}")
        """, language="python")


# ══════════════════════════ TAB 3 — CHARTS & PLOTLY ══════════════════════════ #

with tab3:
    st.title("Charts & Visualizations")
    st.markdown("""
    Streamlit supports multiple charting backends. This tab compares them
    and shows the most useful patterns for each.
    """)

    st.divider()

    # ── Native charts ─────────────────────────────────────────────────────────
    st.header("Native Streamlit Charts")
    st.markdown("""
    `st.bar_chart`, `st.line_chart`, `st.area_chart` are the quickest way to render
    data. They accept DataFrames directly and auto-label from column names.
    """)

    counts = [(y_train == i).sum() for i in range(10)]
    df_counts = pd.DataFrame({"Count": counts}, index=CLASS_NAMES)

    chart_type = st.radio("Chart type", ["bar_chart", "area_chart", "line_chart"], horizontal=True)
    if chart_type == "bar_chart":
        st.bar_chart(df_counts)
    elif chart_type == "area_chart":
        st.area_chart(df_counts)
    else:
        st.line_chart(df_counts)

    with st.expander("Basic chart code"):
        st.code("""
# Simple Examples
import pandas as pd
df = pd.DataFrame({"Count": counts}, index=CLASS_NAMES)
st.bar_chart(df)
st.line_chart(df)
st.area_chart(df)
                
# Examples Above
counts = [(y_train == i).sum() for i in range(10)]
df_counts = pd.DataFrame({"Count": counts}, index=CLASS_NAMES)

chart_type = st.radio("Chart type", ["bar_chart", "area_chart", "line_chart"], horizontal=True)
if chart_type == "bar_chart":
    st.bar_chart(df_counts)
elif chart_type == "area_chart":
    st.area_chart(df_counts)
else:
    st.line_chart(df_counts)""", language="python")

    st.info(""" 
Switching between bar, line, and area charts doesn’t change the data—but it changes interpretation. Choosing the right method for visualizing your data is important!

- Bar charts emphasize comparison  
- Line charts emphasize trends  
- Area charts emphasize accumulation  
""")

    st.divider()

    # ── Plotly interactive ────────────────────────────────────────────────────
    st.header("Plotly")
    st.markdown("""
    `st.plotly_chart` renders fully interactive Plotly figures, including capabilities like hover tooltips,
    zoom, pan, download as PNG. This is how I made my entire semester project!.
    """)

    n_pca = st.slider("PCA sample size", 200, 3000, 1000, step=200, key="pca_plotly")
    rng = np.random.RandomState(7)
    idx_p = rng.choice(len(X_train), n_pca, replace=False)
    X_p = X_train[idx_p].reshape(n_pca, -1).astype(np.float32) / 255.0
    y_p = y_train[idx_p]
    pca2 = PCA(n_components=2, random_state=42)
    coords = pca2.fit_transform(X_p)

    df_pca = pd.DataFrame({
        "PC1": coords[:, 0], "PC2": coords[:, 1],
        "Class": [CLASS_NAMES[i] for i in y_p]
    })

    fig_px = px.scatter(
        df_pca, x="PC1", y="PC2", color="Class",
        title="PCA 2D Projection (interactive — hover, zoom, pan!)",
        template= "plotly_white",
        opacity=0.65, height=500,
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig_px.update_traces(marker_size=5)
    st.plotly_chart(fig_px, use_container_width=True)

    with st.expander("Plotly code"):
        st.code("""
import plotly.express as px

fig = px.scatter(df, x="PC1", y="PC2", color="Class",
                 template="plotly_dark", opacity=0.65)
st.plotly_chart(fig, use_container_width=True)
        """, language="python")

    st.info("""Streamlit and Plotly work together so well to create interactive applications! As Cairo says, "knowledge-building insight is much more common in interactive visualizations". """)

    st.divider()

    # ── Matplotlib ────────────────────────────────────────────────────────────
    st.header("Matplotlib")
    st.markdown("""
    `st.pyplot(fig)` renders any matplotlib figure. Gives you maximum control
    over every visual detail. Use `use_container_width=True` to make it responsive.
    """)

    sel_cls_heat = st.selectbox("Class for heatmap", CLASS_NAMES, key="heat_cls")
    ci_h = CLASS_NAMES.index(sel_cls_heat)
    mean_img = X_train[y_train == ci_h].mean(axis=0)

    fig_m, axes_m = plt.subplots(1, 2, figsize=(9, 4))
    fig_style(fig_m, axes_m)
    axes_m[0].imshow(mean_img, cmap="gray")
    axes_m[0].set_title(f"Mean — {sel_cls_heat}")
    axes_m[0].axis("off")
    im = axes_m[1].imshow(mean_img, cmap="viridis")
    axes_m[1].set_title(f"Mean — {sel_cls_heat} — viridis colormap")
    axes_m[1].axis("off")
    plt.colorbar(im, ax=axes_m[1], shrink=0.8)
    plt.tight_layout()
    st.pyplot(fig_m, use_container_width=True)
    plt.close()

    with st.expander("Matplotlib code"):
        st.code("""
# Above Example
ci_h = CLASS_NAMES.index(sel_cls_heat)
mean_img = X_train[y_train == ci_h].mean(axis=0)

fig_m, axes_m = plt.subplots(1, 2, figsize=(9, 4))
fig_style(fig_m, axes_m)
axes_m[0].imshow(mean_img, cmap="gray")
axes_m[0].set_title(f"Mean — {sel_cls_heat}")
axes_m[0].axis("off")
im = axes_m[1].imshow(mean_img, cmap="viridis")
axes_m[1].set_title(f"Mean — {sel_cls_heat} — viridis colormap")
axes_m[1].axis("off")
plt.colorbar(im, ax=axes_m[1], shrink=0.8)
plt.tight_layout()
st.pyplot(fig_m, use_container_width=True)""", language="python")

    st.divider()

    # ── st.dataframe ──────────────────────────────────────────────────────────
    st.header("`st.dataframe` vs `st.table`")
    st.markdown("""
    `st.dataframe` renders an **interactive sortable table** with column headers you can click.
    `st.table` renders a **static** table, which is better for small reference tables.
    """)

    stats_data = {
        "Class": CLASS_NAMES,
        "Train count": [(y_train == i).sum() for i in range(10)],
        "Mean pixel": [X_train[y_train == i].mean().round(2) for i in range(10)],
        "Std pixel": [X_train[y_train == i].std().round(2) for i in range(10)],
    }
    df_stats = pd.DataFrame(stats_data)

    show_interactive = st.toggle("Interactive dataframe (vs static table)", value=True)
    if show_interactive:
        st.dataframe(df_stats, use_container_width=True, hide_index=True)
        st.caption("`st.dataframe` — click column headers to sort")
    else:
        st.table(df_stats)
        st.caption("`st.table` — static, not sortable")

    with st.expander("Dataframe & table code"):
        st.code("""
# Above Example
show_interactive = st.toggle("Interactive dataframe (vs static table)", value=True)
if show_interactive:
    st.dataframe(df_stats, use_container_width=True, hide_index=True)
    st.caption("`st.dataframe` — click column headers to sort")
else:
    st.table(df_stats)
    st.caption("`st.table` — static, not sortable")""", language="python")
        
    st.info("Again, interactivity features can help to make the story more audience- and reader-driven. Cairo recommends adding interactivity so people can organize the data at will, similar to what you can do with the interactive dataframe above.")


# ══════════════════════════ TAB 4 — FORMS & VALIDATION ══════════════════════════ #

with tab4:
    st.title("Forms & Input Validation")

    st.markdown("""
    `st.form` groups widgets so that **no rerun happens until the user clicks Submit**.
    This is essential for multi-field input where intermediate states are invalid, for example, requiring all fields to be filled before processing.

    Without a form, every widget change triggers an immediate rerun.
    Inside a form, all changes are batched until the submit button is pressed.
    """)

    st.divider()

    # ── Basic form ────────────────────────────────────────────────────────────
    st.header("Basic Form Pattern")

    with st.form("experiment_form"):
        st.subheader("Configure Classifier Experiment")
        st.caption("All fields must be set before submitting. No reruns happen until you click Run.")

        f_col1, f_col2 = st.columns(2)
        with f_col1:
            f_classes = st.multiselect("Classes to include", CLASS_NAMES, default=["T-shirt/top", "Trouser", "Sneaker"])
            f_n = st.select_slider("Training examples per class", [100, 200, 500, 1000], value=200)
        with f_col2:
            f_hidden = st.selectbox("MLP hidden units", [32, 64, 128], index=1)
            f_pca = st.selectbox("PCA dims", [20, 50, 100], index=1)
            f_name = st.text_input("Experiment name", "My Experiment")

        submitted = st.form_submit_button("Run Experiment", type="primary")

    if submitted:
        if len(f_classes) < 2:
            st.error("Please select at least 2 classes.")
        elif not f_name.strip():
            st.error("Please enter an experiment name.")
        else:
            with st.spinner(f"Running '{f_name}' …"):
                class_indices = [CLASS_NAMES.index(c) for c in f_classes]
                mask_train = np.isin(y_train, class_indices)
                X_f_train = X_train[mask_train].reshape(-1, 784).astype("float32") / 255.0
                y_f_train = y_train[mask_train]

                mask_test = np.isin(y_test, class_indices)
                X_f_test = X_test[mask_test].reshape(-1, 784).astype("float32") / 255.0
                y_f_test = y_test[mask_test]

                rng_f = np.random.RandomState(42)
                per_class = f_n
                idx_keep = []
                for ci in class_indices:
                    ci_idx = np.where(y_f_train == ci)[0]
                    chosen = rng_f.choice(ci_idx, min(per_class, len(ci_idx)), replace=False)
                    idx_keep.extend(chosen)
                X_sub_f = X_f_train[idx_keep]
                y_sub_f = y_f_train[idx_keep]

                pca_f = PCA(n_components=f_pca, random_state=42)
                X_sub_pca = pca_f.fit_transform(X_sub_f)
                X_test_pca = pca_f.transform(X_f_test)

                clf_f = MLPClassifier(hidden_layer_sizes=(f_hidden,), max_iter=20, random_state=42)
                clf_f.fit(X_sub_pca, y_sub_f)
                acc_f = (clf_f.predict(X_test_pca) == y_f_test).mean()

            st.success(f"**{f_name}** complete! Test accuracy: **{acc_f:.1%}**")

            cm_f = confusion_matrix(y_f_test, clf_f.predict(X_test_pca))
            fig_f, ax_f = plt.subplots(figsize=(6, 5))
            fig_style(fig_f, ax_f)
            sns.heatmap(cm_f.astype(float) / cm_f.sum(axis=1, keepdims=True),
                        ax=ax_f, cmap="Blues", annot=True, fmt=".2f",
                        xticklabels=f_classes, yticklabels=f_classes,
                        linewidths=0.3)
            ax_f.set_xlabel("Predicted"); ax_f.set_ylabel("True")
            ax_f.set_title(f"{f_name} — Confusion Matrix")
            ax_f.tick_params(colors= "black", labelsize=8)
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            st.pyplot(fig_f)
            plt.close()

    with st.expander("Form code"):
        st.code("""
with st.form("my_form"):
    name = st.text_input("Name")
    n    = st.slider("N", 1, 100, 10)
    submitted = st.form_submit_button("Submit")

# Nothing above reruns until Submit is clicked
if submitted:
    if not name:
        st.error("Name required")
    else:
        run_experiment(name, n)
        """, language="python")

    st.info(""" As Cairo says in his chapter on uncertainty, if possible, people should 'work to find a way to display it [uncertainty] on the visualization itself in a manner that doesn't clutter it'. """
    "Confusion matrices are great in displaying the uncertainty and failures in model performance, ultimatley enhacing transparency and truthfulness.")

    st.divider()

    # ── Multi-step wizard ─────────────────────────────────────────────────────
    st.header("Multi-Step Wizard with Session State")
    st.markdown("""
    Forms don't have to be one-shot. Combining `st.form` with `st.session_state`
    lets you build multi-step wizards where each page is a separate form.
    """)

    if "wizard_step" not in st.session_state:
        st.session_state.wizard_step = 1
    if "wizard_data" not in st.session_state:
        st.session_state.wizard_data = {}

    step = st.session_state.wizard_step
    st.progress(step / 3, text=f"Step {step} of 3")

    if step == 1:
        with st.form("wizard_step1"):
            st.subheader("Step 1: Choose your class")
            wiz_cls = st.selectbox("Class", CLASS_NAMES)
            if st.form_submit_button("Next →"):
                st.session_state.wizard_data["class"] = wiz_cls
                st.session_state.wizard_step = 2
                st.rerun()

    elif step == 2:
        with st.form("wizard_step2"):
            st.subheader("Step 2: Display options")
            wiz_cmap = st.selectbox("Colormap", ["gray", "viridis", "hot"])
            wiz_n = st.slider("How many images", 3, 10, 5)
            c_b, c_n = st.columns(2)
            with c_b:
                if st.form_submit_button("← Back"):
                    st.session_state.wizard_step = 1
                    st.rerun()
            with c_n:
                if st.form_submit_button("Next →"):
                    st.session_state.wizard_data["cmap"] = wiz_cmap
                    st.session_state.wizard_data["n"] = wiz_n
                    st.session_state.wizard_step = 3
                    st.rerun()

    elif step == 3:
        d = st.session_state.wizard_data
        st.subheader("Step 3: Results")
        st.markdown(f"Showing **{d['n']}** images of **{d['class']}** with colormap `{d['cmap']}`")
        ci_w = CLASS_NAMES.index(d["class"])
        sample_w = X_train[y_train == ci_w][:d["n"]]
        wiz_cols = st.columns(d["n"])
        for wc, wim in zip(wiz_cols, sample_w):
            fig_w, ax_w = plt.subplots(figsize=(1.5, 1.5))
            fig_style(fig_w, ax_w)
            ax_w.imshow(wim, cmap=d["cmap"]); ax_w.axis("off")
            wc.pyplot(fig_w, use_container_width=True)
            plt.close()
        if st.button("Start over"):
            st.session_state.wizard_step = 1
            st.session_state.wizard_data = {}
            st.rerun()

# ══════════════════════════ TAB 5 — MNIST Exploration ══════════════════════════ #

with tab5:

    @st.cache_data(show_spinner="Loading data …")
    def load_data():
        from tensorflow.keras.datasets import fashion_mnist
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
        return X_train, y_train, X_test, y_test

    @st.cache_data(show_spinner="Training classifier (may take ~60s) …")
    def train_model(n_train, hidden_size, pca_dim):
        X_train, y_train, X_test, y_test = load_data()

        X_tr = X_train[:n_train].reshape(n_train, -1).astype("float32") / 255.0
        y_tr = y_train[:n_train]
        X_te = X_test.reshape(len(X_test), -1).astype("float32") / 255.0

        pca = PCA(n_components=pca_dim, random_state=42)
        X_tr = pca.fit_transform(X_tr)
        X_te = pca.transform(X_te)

        clf = MLPClassifier(
            hidden_layer_sizes=(hidden_size,),
            max_iter=30, random_state=42,
            early_stopping=True, validation_fraction=0.1,
            verbose=False
        )
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        acc = (y_pred == y_test).mean()
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred,
                                    target_names=CLASS_NAMES, output_dict=True)
        return acc, cm, report, clf, pca

    X_train, y_train, X_test, y_test = load_data()

    st.markdown("Below we can see some examples generated by Claude of the capabilites of Streamlit and Fashion-MNIST!")
    # ── Training controls ─────────────────────────────────────────────────────────
    st.header("Training Controls")

    c1, c2, c3 = st.columns(3)
    with c1:
        n_train = st.select_slider("Training examples", [1000, 5000, 10000, 30000, 60000], value=10000)
    with c2:
        hidden_size = st.select_slider("Hidden units", [64, 128, 256, 512], value=128)
    with c3:
        pca_dim = st.select_slider("PCA dimensions", [50, 100, 200, 784], value=100)

    if st.button("Train Classifier", type="primary"):
        st.cache_data.clear()  # force retrain on new params

    with st.spinner("Training …"):
        acc, cm, report, clf, pca = train_model(n_train, hidden_size, pca_dim)

    st.success(f"**Test accuracy: {acc:.1%}**")

    st.divider()

    @st.cache_data(show_spinner="Loading data …")
    def load_data():
        from tensorflow.keras.datasets import fashion_mnist
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
        return X_train, y_train, X_test, y_test

    @st.cache_data(show_spinner="Training model for uncertainty analysis …")
    def get_model_and_probs():
        X_train, y_train, X_test, y_test = load_data()
        X_tr = X_train[:20000].reshape(20000, -1).astype("float32") / 255.0
        X_te = X_test.reshape(len(X_test), -1).astype("float32") / 255.0

        pca = PCA(n_components=100, random_state=42)
        X_tr_pca = pca.fit_transform(X_tr)
        X_te_pca = pca.transform(X_te)

        clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=40, random_state=42,
                            early_stopping=True, validation_fraction=0.1)
        clf.fit(X_tr_pca, y_train[:20000])

        probs = clf.predict_proba(X_te_pca)
        preds = probs.argmax(axis=1)
        confs = probs.max(axis=1)
        return probs, preds, confs, pca

    X_train, y_train, X_test, y_test = load_data()

    with st.spinner("Preparing model …"):
        probs, preds, confs, pca = get_model_and_probs()

    correct_mask = preds == y_test

    # ── Individual image viewer ────────────────────────────────────────────────────
    st.header("Individual Prediction Inspector")
    st.markdown("""
    Select a filter below to browse specific types of predictions. The bar chart shows
    the full softmax distribution — not just the top prediction.
    """)

    filter_option = st.radio(
        "Show me:",
        ["Random images", "High-confidence errors", "Low-confidence correct"],
        horizontal=True
    )

    n_show = st.slider("Images to display", 3, 9, 6)

    if filter_option == "Random images":
        sample_idx = np.random.choice(len(X_test), n_show, replace=False)
    elif filter_option == "High-confidence errors":
        error_idx = np.where(~correct_mask)[0]
        sorted_err = error_idx[np.argsort(confs[error_idx])[::-1]]
        sample_idx = sorted_err[:n_show]
    else:
        correct_idx = np.where(correct_mask)[0]
        sorted_low = correct_idx[np.argsort(confs[correct_idx])]
        sample_idx = sorted_low[:n_show]

    cols = st.columns(n_show)
    for col, idx in zip(cols, sample_idx):
        img = X_test[idx]
        true_label = CLASS_NAMES[y_test[idx]]
        pred_label = CLASS_NAMES[preds[idx]]
        confidence = confs[idx]
        is_correct = correct_mask[idx]

        fig_img, ax_img = plt.subplots(figsize=(2, 2))
        fig_img.patch.set_facecolor("#0e1117")
        ax_img.imshow(img, cmap="gray")
        ax_img.axis("off")
        border_color = "#4caf50" if is_correct else "#f44336"
        for spine in ax_img.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)
        col.pyplot(fig_img)
        plt.close()

        fig_bar, ax_bar = plt.subplots(figsize=(2.2, 3))
        fig_bar.patch.set_facecolor("#0e1117")
        ax_bar.set_facecolor("#0e1117")
        bar_colors = ["#4caf50" if i == y_test[idx] else
                    ("#f44336" if i == preds[idx] else "#555")
                    for i in range(10)]
        ax_bar.barh(CLASS_NAMES, probs[idx], color=bar_colors)
        ax_bar.set_xlim(0, 1)
        ax_bar.tick_params(colors="white", labelsize=6)
        ax_bar.spines[:].set_color("#333")
        ax_bar.set_title(
            f"True: {true_label}\nPred: {pred_label}\n{confidence:.0%}",
            color="white", fontsize=7
        )
        col.pyplot(fig_bar)
        plt.close()

    st.info("""Working with uncertainty is a big concept in visualizations and statistics. We directly explored this in the Day 12 In-Class Assignment when visualizing the confidence intervals of the climate model predictions. The above visualization emphasizes how being transparent with uncertainty is valuable. 
            
“Data Feminism” encourages us to challenge the idea of a single “correct” answer and instead surface ambiguity and context.

By showing full probability distributions, this visualization avoids false certainty, encourages critical thinking, and makes the model’s reasoning more interpretable.""")

    st.divider()

    # ── Noise Perturbation Playground ─────────────────────────────────────────────
    st.header("Noise Perturbation Playground")
    st.markdown("""
    Slide the intensity up and watch the model's softmax distribution destabilize.
    Each image shows the **clean** input alongside the **noisy** version — green bars
    are the true class, red bars show where the model lands if it flips.
    """)

    NOISE_TYPES = ["Gaussian", "Salt & Pepper", "Blur", "Brightness Shift", "Occlusion"]

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        noise_type = st.selectbox("Noise type", NOISE_TYPES)
    with col_b:
        noise_intensity = st.slider("Intensity", 0.0, 1.0, 0.15, step=0.01)
    with col_c:
        n_display = st.select_slider("Images to show", [4, 6, 8], value=4)

    if st.button("Resample images"):
        st.session_state["noise_indices"] = np.random.choice(len(X_test), n_display, replace=False)

    if "noise_indices" not in st.session_state:
        st.session_state["noise_indices"] = np.random.choice(len(X_test), n_display, replace=False)

    sample_idx = st.session_state["noise_indices"][:n_display]

    def apply_noise(img, intensity, noise_type):
        img = img.astype("float32") / 255.0
        if noise_type == "Gaussian":
            img += np.random.normal(0, intensity * 0.5, img.shape)
        elif noise_type == "Salt & Pepper":
            mask = np.random.rand(*img.shape)
            img[mask < intensity * 0.1] = 0.0
            img[mask > 1 - intensity * 0.1] = 1.0
        elif noise_type == "Blur":
            from scipy.ndimage import uniform_filter
            sigma = intensity * 5
            img = uniform_filter(img, size=max(1, int(sigma * 2 + 1)))
        elif noise_type == "Brightness Shift":
            img += (intensity * 2 - 1) * 0.7
        elif noise_type == "Occlusion":
            bw = int(intensity * 20)
            bh = int(intensity * 20)
            if bw > 0 and bh > 0:
                bx = np.random.randint(0, max(1, 28 - bw))
                by = np.random.randint(0, max(1, 28 - bh))
                img[by:by+bh, bx:bx+bw] = 0.0
        return np.clip(img, 0, 1)

    # Use the trained clf + pca from the training block above
    cols = st.columns(n_display)
    flip_count = 0

    for col, idx in zip(cols, sample_idx):
        clean_img = X_test[idx]
        true_label = y_test[idx]
        noisy_img = apply_noise(clean_img.copy(), noise_intensity, noise_type)

        # Get predictions
        clean_flat = (clean_img.reshape(1, -1).astype("float32") / 255.0)
        noisy_flat = noisy_img.reshape(1, -1).astype("float32")

        clean_pca = pca.transform(clean_flat)
        noisy_pca = pca.transform(noisy_flat)

        clean_probs = clf.predict_proba(clean_pca)[0]
        noisy_probs = clf.predict_proba(noisy_pca)[0]

        clean_pred = clean_probs.argmax()
        noisy_pred = noisy_probs.argmax()
        is_flipped = noisy_pred != true_label

        if is_flipped:
            flip_count += 1

        # Side-by-side clean vs noisy
        fig, axes = plt.subplots(1, 2, figsize=(2.5, 1.3))
        fig.patch.set_facecolor("#0e1117")
        for ax, im, title in zip(axes, [clean_img, noisy_img * 255], ["clean", "noisy"]):
            ax.imshow(im.astype("uint8") if title=="clean" else (im*255).clip(0,255).astype("uint8"),
                    cmap="gray")
            ax.axis("off")
            ax.set_title(title, color="white", fontsize=7)
        col.pyplot(fig)
        plt.close()

        # Softmax bar chart
        fig_bar, ax_bar = plt.subplots(figsize=(2.2, 3.2))
        fig_bar.patch.set_facecolor("#0e1117")
        ax_bar.set_facecolor("#0e1117")
        bar_colors = [
            "#4caf50" if i == true_label else
            ("#f44336" if i == noisy_pred and is_flipped else "#555")
            for i in range(10)
        ]
        ax_bar.barh(CLASS_NAMES, noisy_probs, color=bar_colors)
        ax_bar.set_xlim(0, 1)
        ax_bar.tick_params(colors="white", labelsize=6)
        ax_bar.spines[:].set_color("#333")
        status = "FLIPPED" if is_flipped else "stable"
        ax_bar.set_title(
            f"True: {CLASS_NAMES[true_label]}\n"
            f"Pred: {CLASS_NAMES[noisy_pred]} ({status})\n"
            f"Conf: {noisy_probs.max():.0%}",
            color="#f44336" if is_flipped else "white", fontsize=7
        )
        col.pyplot(fig_bar)
        plt.close()
