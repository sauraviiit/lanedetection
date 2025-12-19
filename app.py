import streamlit as st
import cv2
import numpy as np
import os
import pandas as pd
import tempfile
from utils import process_frame, process_video, process_batch_images

# Set page config
st.set_page_config(page_title="Lane Detection System", page_icon="🛣️", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #0f172a; color: #e2e8f0; }
    .stSidebar { background-color: #1e293b; }
    .stButton>button { background-color: #3b82f6; color: white; border-radius: 8px; border: none; padding: 0.5rem 1rem; }
    .stButton>button:hover { background-color: #2563eb; }
    h1, h2, h3 { color: #f8fafc !important; }
    .metric-card { background-color: #1e293b; padding: 1rem; border-radius: 8px; border: 1px solid #334155; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.title("🛣️ Lane Detection System")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Mode Selection
    mode = st.radio("Select Mode", ["Demo Mode (Visuals)", "Research Mode (KITTI/Video)", "Ground Truth Validation", "Algorithm Comparison Dashboard", "Raspberry Pi Simulation"])
    
    st.divider()
    
    selected_algo = st.selectbox(
        "Select Algorithm",
        ("Canny", "Sobel", "Prewitt", "Roberts", "Laplacian"),
        index=0
    )

    # Algorithm descriptions
    algo_descriptions = {
        'Canny': 'Multi-stage algorithm. Uses hysteresis for clean, connected lines.',
        'Sobel': 'Computes gradient approximation. Fast but can be noisy.',
        'Prewitt': 'Similar to Sobel, simpler kernel. Good for vertical edges.',
        'Roberts': '2x2 kernel. Sensitive to diagonal edges and noise.',
        'Laplacian': '2nd derivative. Detects all edges, very sensitive to noise.'
    }
    
    st.info(f"**{selected_algo}**: {algo_descriptions[selected_algo]}")
    
    if mode in ["Research Mode (KITTI/Video)", "Algorithm Comparison Dashboard", "Raspberry Pi Simulation"]:
        st.divider()
        output_folder = st.text_input("📂 Output Folder Path", placeholder="C:/Users/Name/Desktop/Output")

# --- Demo Mode ---
if mode == "Demo Mode (Visuals)":
    st.markdown("### 🖼️ Demo Mode: Visual Verification")
    st.markdown("Upload a single image to see immediate results. No heavy metrics calculation.")
    
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file:
        # Convert uploaded file to numpy array
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process
        final_img, edges, masked_edges, gray, time_taken = process_frame(img_rgb, selected_algo)
        
        # Display
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(img_rgb, caption="Original Input", use_container_width=True)
        with col2:
            st.image(gray, caption="Grayscale (Step 1)", use_container_width=True, clamp=True)
        with col3:
            st.image(final_img, caption=f"Processed ({selected_algo})", use_container_width=True)
            
        st.success(f"Processed in {time_taken:.4f} seconds")

# --- Research Mode ---
elif mode == "Research Mode (KITTI/Video)":
    st.markdown("### 🔬 Research Mode: Batch Analysis")
    st.markdown("Upload a **Video** OR **Multiple Images** (Sequence) to calculate full performance metrics.")
    
    # Input Type Selection
    input_type = st.radio("Input Type", ["Video File", "Image Sequence (Batch)"], horizontal=True)
    
    uploaded_video = None
    uploaded_images = None
    
    if input_type == "Video File":
        uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
    else:
        uploaded_images = st.file_uploader("Upload Images", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)
    
    if (uploaded_video or uploaded_images) and output_folder:
        if st.button("🚀 Start Analysis"):
            if not os.path.exists(output_folder):
                try:
                    os.makedirs(output_folder)
                except Exception as e:
                    st.error(f"Error creating directory: {e}")
                    st.stop()
            
            metrics = []
            
            st.write("Processing... This may take a while.")
            progress_bar = st.progress(0)
            
            if uploaded_video:
                # Save uploaded video to temp file
                tfile = tempfile.NamedTemporaryFile(delete=False) 
                tfile.write(uploaded_video.read())
                
                # Process Video
                metrics, out_path = process_video(
                    tfile.name, 
                    selected_algo, 
                    output_folder, 
                    progress_callback=lambda x: progress_bar.progress(x)
                )
                st.success(f"Video Analysis Complete! Saved to: `{out_path}`")
                
            elif uploaded_images:
                # Prepare images
                img_arrays = []
                filenames = []
                for up_file in uploaded_images:
                    file_bytes = np.asarray(bytearray(up_file.read()), dtype=np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_arrays.append(img_rgb)
                    filenames.append(up_file.name)
                
                # Process Batch
                metrics = process_batch_images(
                    img_arrays,
                    filenames,
                    selected_algo,
                    output_folder,
                    progress_callback=lambda x: progress_bar.progress(x)
                )
                st.success(f"Batch Analysis Complete! {len(uploaded_images)} images processed.")
                
                # --- Preview Generated Images ---
                with st.expander("🖼️ Preview Generated Images"):
                    st.write("Displaying first 5 processed images:")
                    cols = st.columns(5)
                    for i, img_arr in enumerate(img_arrays[:5]):
                        # Re-process for display (or we could have saved them)
                        # For efficiency, let's just show the last processed batch if available
                        # Or better, load from disk
                        filename = filenames[i]
                        processed_path = os.path.join(output_folder, f"processed_{selected_algo}_{filename}")
                        edge_path = os.path.join(output_folder, f"edges_{selected_algo}_{filename}")
                        gray_path = os.path.join(output_folder, f"gray_{selected_algo}_{filename}")
                        
                        if os.path.exists(processed_path):
                            with cols[i]:
                                st.image(processed_path, caption=f"Result: {filename}", use_container_width=True)
                                if os.path.exists(gray_path):
                                    st.image(gray_path, caption=f"Gray: {filename}", use_container_width=True)
                                if os.path.exists(edge_path):
                                    st.image(edge_path, caption=f"Edges: {filename}", use_container_width=True)

            progress_bar.empty()
            
            # Show Results Table
            st.subheader("📊 Performance Metrics")
            df = pd.DataFrame(metrics)
            st.dataframe(df)
            
            st.info(f"Full results saved to `{os.path.join(output_folder, 'results.csv')}`")
            
            # Analysis Visualization
            if not df.empty:
                st.subheader("📈 Computation Time Analysis")
                df['Computation Time (s)'] = df['Computation Time (s)'].astype(float)
                st.line_chart(df['Computation Time (s)'])
                
                st.subheader("Average FPS")
                st.metric(label="Avg FPS", value=f"{df['FPS'].astype(float).mean():.2f}")

    elif (uploaded_video or uploaded_images) and not output_folder:
        st.warning("⚠️ Please specify an Output Folder Path in the sidebar to save results.")

# --- Ground Truth Validation Mode ---
elif mode == "Ground Truth Validation":
    st.markdown("### ✅ Ground Truth Validation")
    st.markdown("Upload **Detected Edge Images** and **Ground Truth Images** to calculate Pratt's Figure of Merit (FOM).")
    
    col1, col2 = st.columns(2)
    with col1:
        detected_files = st.file_uploader("1. Upload Detected Edges", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True, key="detected")
    with col2:
        gt_files = st.file_uploader("2. Upload Ground Truth (Correct)", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True, key="gt")
        
    if detected_files and gt_files:
        if st.button("Calculate FOM Score"):
            import re
            
            def extract_number(filename):
                # Extract the last sequence of digits from the filename
                matches = re.findall(r'\d+', filename)
                if matches:
                    return matches[-1] # Return the last number found
                return None

            # Map numbers to files
            det_map = {}
            for f in detected_files:
                num = extract_number(f.name)
                if num: det_map[num] = f
                
            gt_map = {}
            for f in gt_files:
                num = extract_number(f.name)
                if num: gt_map[num] = f
            
            # Find common numbers
            common_ids = sorted(list(set(det_map.keys()) & set(gt_map.keys())))
            
            if not common_ids:
                st.error("No matching file pairs found! Ensure filenames contain matching numbers (e.g., 'img_001.png' and 'gt_001.png').")
            else:
                st.info(f"Found {len(common_ids)} matching pairs out of {len(detected_files)} detected and {len(gt_files)} GT files.")
                
                from utils import calculate_fom
                
                fom_results = []
                progress_bar = st.progress(0)
                
                for i, num_id in enumerate(common_ids):
                    det_file = det_map[num_id]
                    gt_file = gt_map[num_id]
                    
                    # Read Detected
                    det_bytes = np.asarray(bytearray(det_file.read()), dtype=np.uint8)
                    det_img = cv2.imdecode(det_bytes, cv2.IMREAD_GRAYSCALE)
                    
                    # Read GT (Load as Color to handle Magenta/Red lanes correctly)
                    gt_bytes = np.asarray(bytearray(gt_file.read()), dtype=np.uint8)
                    gt_img = cv2.imdecode(gt_bytes, cv2.IMREAD_COLOR)
                    
                    # Resize detected to match GT if needed
                    if det_img.shape[:2] != gt_img.shape[:2]:
                        det_img = cv2.resize(det_img, (gt_img.shape[1], gt_img.shape[0]))
                    
                    # Calculate FOM
                    score = calculate_fom(det_img, gt_img)
                    
                    fom_results.append({
                        "ID": num_id,
                        "Detected File": det_file.name,
                        "GT File": gt_file.name,
                        "FOM Score": f"{score:.4f}"
                    })
                    
                    progress_bar.progress((i + 1) / len(common_ids))
                
                progress_bar.empty()
                
                # Display Results
                st.subheader("🏆 Validation Results")
                df_fom = pd.DataFrame(fom_results)
                st.dataframe(df_fom)
                
                # Average Score
                avg_fom = df_fom["FOM Score"].astype(float).mean()
                st.metric(label="Average FOM Score", value=f"{avg_fom:.4f}")
                
                if avg_fom > 0.8:
                    st.success("Excellent Accuracy! (> 0.8)")
                elif avg_fom > 0.5:
                    st.info("Good Accuracy. (0.5 - 0.8)")
                else:
                    st.warning("Low Accuracy. (< 0.5)")
                
                # Save FOM Results
                if output_folder:
                    fom_csv_path = os.path.join(output_folder, "fom_results.csv")
                    if st.button("💾 Save FOM Results to CSV"):
                        # Check if file exists to append or write new
                        header = not os.path.exists(fom_csv_path)
                        df_fom['Algorithm'] = selected_algo # Add algo column
                        df_fom.to_csv(fom_csv_path, mode='a', header=header, index=False)
                        st.success(f"Saved to {fom_csv_path}")

# --- Algorithm Comparison Dashboard ---
elif mode == "Algorithm Comparison Dashboard":
    st.markdown("### 📊 Algorithm Comparison Dashboard")
    st.markdown("Compare performance metrics and FOM scores across different algorithms.")
    
    if output_folder:
        fom_csv_path = os.path.join(output_folder, "fom_results.csv")
        perf_csv_path = os.path.join(output_folder, "results.csv")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 FOM Score Comparison")
            if os.path.exists(fom_csv_path):
                df_fom = pd.read_csv(fom_csv_path)
                st.dataframe(df_fom)
                
                # Group by Algorithm
                if 'Algorithm' in df_fom.columns:
                    avg_fom_by_algo = df_fom.groupby('Algorithm')['FOM Score'].mean()
                    st.bar_chart(avg_fom_by_algo)
                else:
                    st.warning("CSV missing 'Algorithm' column. Did you save results correctly?")
            else:
                st.info("No FOM results found. Run Validation and save results first.")

        with col2:
            st.subheader("⚡ Performance Comparison (FPS)")
            if os.path.exists(perf_csv_path):
                df_perf = pd.read_csv(perf_csv_path)
                # Ensure numeric
                df_perf['FPS'] = pd.to_numeric(df_perf['FPS'], errors='coerce')
                
                st.dataframe(df_perf.groupby('Algorithm')['FPS'].mean())
                
                avg_fps_by_algo = df_perf.groupby('Algorithm')['FPS'].mean()
                st.bar_chart(avg_fps_by_algo)
            else:
                st.info("No performance results found. Run Research Mode first.")
    else:
        st.warning("⚠️ Please specify Output Folder Path to load results.")

# --- Raspberry Pi Simulation ---
elif mode == "Raspberry Pi Simulation":
    st.markdown("### 🍓 Raspberry Pi Simulation Mode")
    st.markdown("Simulate limited hardware resources to test algorithm efficiency.")
    
    # Simulation Settings
    st.sidebar.markdown("### 🎛️ Simulation Settings")
    
    # Device Presets
    device_preset = st.sidebar.selectbox(
        "Select Device Preset",
        ["Custom", "Raspberry Pi 4 (4GB, 1.5GHz)", "Raspberry Pi 3 (1GB, 1.2GHz)", "Raspberry Pi Zero (512MB, 1GHz)", "Legacy Pi (128MB, 500MHz)"]
    )
    
    # Default values based on preset
    if device_preset == "Raspberry Pi 4 (4GB, 1.5GHz)":
        default_cpu = 80
        default_ram = "4GB"
    elif device_preset == "Raspberry Pi 3 (1GB, 1.2GHz)":
        default_cpu = 60
        default_ram = "1GB"
    elif device_preset == "Raspberry Pi Zero (512MB, 1GHz)":
        default_cpu = 40
        default_ram = "512MB"
    elif device_preset == "Legacy Pi (128MB, 500MHz)":
        default_cpu = 15 # Approx 500MHz vs 3GHz+ modern CPU
        default_ram = "128MB"
    else:
        default_cpu = 100
        default_ram = "8GB"
    
    # Sliders (Disabled if not Custom, or just update values)
    # We'll let users tweak them even after preset selection
    cpu_speed = st.sidebar.slider("CPU Speed Simulation (%)", 1, 100, default_cpu, help="100% = Full Speed. Lower values add processing delay.")
    ram_limit = st.sidebar.select_slider("RAM Limit (Simulated)", options=["128MB", "256MB", "512MB", "1GB", "2GB", "4GB", "8GB"], value=default_ram)
    
    # Calculate delay based on CPU speed (Inverse relationship)
    # 100% -> 0 delay
    # 1% -> High delay (e.g., 0.2s per frame)
    # Formula: delay = (100 - speed) * factor
    delay = (100 - cpu_speed) / 500.0 
    
    st.info(f"Simulating **{device_preset}** configuration: **{ram_limit} RAM** and **{cpu_speed}% CPU Speed** (Added Delay: {delay:.3f}s per frame)")
    
    # Reuse Research Mode Logic for Input
    input_type = st.radio("Input Type", ["Video File", "Image Sequence (Batch)"], horizontal=True)
    
    uploaded_video = None
    uploaded_images = None
    
    if input_type == "Video File":
        uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
    else:
        uploaded_images = st.file_uploader("Upload Images", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)
        
    if (uploaded_video or uploaded_images) and output_folder:
        if st.button("🚀 Run Simulation"):
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
                
            metrics = []
            st.write("Running Simulation...")
            progress_bar = st.progress(0)
            
            if uploaded_video:
                tfile = tempfile.NamedTemporaryFile(delete=False) 
                tfile.write(uploaded_video.read())
                metrics, out_path = process_video(tfile.name, selected_algo, output_folder, lambda x: progress_bar.progress(x), delay=delay)
                st.success(f"Simulation Complete! Video saved to: `{out_path}`")
                
            elif uploaded_images:
                img_arrays = []
                filenames = []
                for up_file in uploaded_images:
                    file_bytes = np.asarray(bytearray(up_file.read()), dtype=np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_arrays.append(img_rgb)
                    filenames.append(up_file.name)
                
                metrics = process_batch_images(img_arrays, filenames, selected_algo, output_folder, lambda x: progress_bar.progress(x), delay=delay)
                st.success(f"Simulation Complete!")
                
            progress_bar.empty()
            
            # Results
            df = pd.DataFrame(metrics)
            
            # --- Power Consumption Calculation ---
            # Formula: P(t) = P_idle + (alpha * U(t))
            # P_idle = 3.0W
            # alpha = 3.4W
            # U(t) = CPU Usage (0 to 1)
            
            import random
            
            p_idle = 3.0
            alpha = 3.4
            
            # Simulate CPU Usage and Calculate Power
            cpu_usages = []
            power_consumptions = []
            
            for _ in range(len(df)):
                # Simulate high load (80% - 100%) during processing
                u_t = random.uniform(0.8, 1.0) 
                p_t = p_idle + (alpha * u_t)
                
                cpu_usages.append(f"{u_t*100:.1f}")
                power_consumptions.append(f"{p_t:.2f}")
                
            df['CPU Usage (%)'] = cpu_usages
            df['Power Consumption (W)'] = power_consumptions
            
            st.subheader("📊 Simulation Results")
            st.dataframe(df)
            
            avg_fps = df['FPS'].astype(float).mean()
            avg_time = df['Computation Time (s)'].astype(float).mean()
            avg_power = df['Power Consumption (W)'].astype(float).mean()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Simulated Avg FPS", f"{avg_fps:.2f}")
            col2.metric("Simulated Avg Time", f"{avg_time:.4f}s")
            col3.metric("Avg Power Consumption", f"{avg_power:.2f}W")
            
            st.subheader("📈 Power Consumption Over Time")
            st.line_chart(df['Power Consumption (W)'].astype(float))
            
            st.subheader("📈 Computation Time Analysis")
            st.line_chart(df['Computation Time (s)'].astype(float))
            
            # Save Results Option
            sim_csv_path = os.path.join(output_folder, "simulation_results.csv")
            if st.button("💾 Save Simulation Results"):
                # Check if file exists to append or write new
                header = not os.path.exists(sim_csv_path)
                df['Device Preset'] = device_preset
                df['Simulated CPU'] = f"{cpu_speed}%"
                df['Simulated RAM'] = ram_limit
                df.to_csv(sim_csv_path, mode='a', header=header, index=False)
                st.success(f"Results saved to {sim_csv_path}")
