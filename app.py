import streamlit as st
import cv2
import numpy as np
from count_tool import ColorAnalyzer
import time

st.set_page_config(layout="wide", page_title="Color Region Analyzer")

st.title("🎨 Color Region Analyzer")

# --- Sidebar Controls ---
st.sidebar.header("Settings")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png", "tiff", "tif"], max_upload_size=500)

mode = st.sidebar.radio("Palette Mode", ["Auto Palette (K-Means)", "Manual Palette"])

if mode == "Auto Palette (K-Means)":
    num_colors = st.sidebar.slider("Number of Colors", 2, 20, 10)
else:
    hex_input = st.sidebar.text_area("Manual HEX Colors (comma separated)", "#ffffff, #3994c0, #fe3b37")
    manual_palette = [h.strip() for h in hex_input.split(",")]

st.sidebar.subheader("Filtering Parameters")
min_area = st.sidebar.number_input("Min Area (pixels)", 0, 10000, 50)
max_aspect = st.sidebar.slider("Max Aspect Ratio", 1.0, 10.0, 4.0)
area_ratio = st.sidebar.slider("Area Ratio (Solidity)", 0.0, 1.0, 0.4)

st.sidebar.subheader("Morphology")
morph_size = st.sidebar.slider("Kernel Size", 0, 10, 3)
iterations = st.sidebar.slider("Iterations", 1, 5, 1)

# --- Logic Processing ---
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if st.sidebar.button("Run Analysis"):
        # We use st.status to show a collapsible log of events
        with st.status("Processing Image...", expanded=True) as status:

            st.write("Preparing image and color space...")
            start_time = time.time()

            # Step 1: Clustering
            if mode == "Auto Palette (K-Means)":
                st.write(f"Running K-Means for {num_colors} colors (this may take a moment)...")
                clustered_bgr, palette_bgr, label_map = ColorAnalyzer.get_labels_and_palette(img, num_colors)
            else:
                st.write("Mapping image to manual palette...")
                clustered_bgr, palette_bgr, label_map = ColorAnalyzer.apply_manual_palette(img, manual_palette)

            # Step 2: Per-color processing with a progress bar
            results = []
            total_count = 0
            num_palettes = len(palette_bgr)

            st.write(f"Analyzing {num_palettes} color layers...")
            progress_bar = st.progress(0)

            for i, color in enumerate(palette_bgr):
                hex_val = ColorAnalyzer.bgr_to_hex(color)
                st.write(f"Processing color {i + 1}/{num_palettes}: {hex_val}")

                # Create mask
                mask = (label_map == i).astype(np.uint8) * 255

                # Morphology
                mask = ColorAnalyzer.process_mask(mask, morph_size, iterations)

                # Filter and Count
                count, filtered, removed = ColorAnalyzer.filter_and_count(mask, min_area, max_aspect, area_ratio)

                results.append({
                    "id": i,
                    "color_bgr": color,
                    "hex": hex_val,
                    "count": count,
                    "mask": mask,
                    "filtered": filtered,
                    "removed": removed
                })
                total_count += count

                # Update progress
                progress_bar.progress((i + 1) / num_palettes)

            end_time = time.time()
            status.update(label=f"Analysis Complete in {end_time - start_time:.2f}s!", state="complete", expanded=False)

            # Store in session state
            st.session_state['results'] = results
            st.session_state['clustered_bgr'] = clustered_bgr
            st.session_state['total_count'] = total_count

    # --- Display Results ---
    if 'results' in st.session_state:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
        with col2:
            st.subheader("Clustered Image")
            st.image(cv2.cvtColor(st.session_state['clustered_bgr'], cv2.COLOR_BGR2RGB), use_container_width=True)

        st.divider()
        # Summary Statistics
        st.subheader("📊 Summary Statistics")
        st.metric("Total Regions Found", st.session_state['total_count'])

        # Displaying the color swatches in the table
        summary_data = [{"Color": r['hex'], "Count": r['count']} for r in st.session_state['results']]
        st.table(summary_data)

        # Exploration Mode
        st.subheader("🔍 Mask Inspection")
        color_options = [f"Color {r['id']} ({r['hex']})" for r in st.session_state['results']]
        selected_option = st.selectbox("Select a color to inspect:", color_options)
        selected_idx = color_options.index(selected_option)
        res = st.session_state['results'][selected_idx]

        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.image(res['mask'], caption="Raw Color Mask")
        m_col2.image(res['filtered'], caption="Filtered (Kept) Regions")
        m_col3.image(res['removed'], caption="Removed Pixels (Noise)")

else:
    st.info("Please upload an image in the sidebar to begin.")