import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import io
import base64
import time

# Set page configuration
st.set_page_config(page_title="Cardiomyocyte Development Animation", layout="wide")

# Title and description
st.title("Cardiomyocyte Morphological Development (Day 1-8)")
st.markdown("This animation shows the continuous morphological and functional changes in cardiomyocytes over an 8-day period.")

# Cell characteristics for each day - simplified for faster processing
cell_characteristics = {
    1: {
        "title": "Immature Stage",
        "shape": "round",
        "elongation": 1.0,
        "alignment": 0.1,
        "connection": 0.1,
        "sarcomere_organization": 0.1,
        "beating_strength": 0.1,
        "beating_sync": 0.1,
        "color_base": (255, 214, 204),
        "nucleus_size": 0.7,
        "debris_level": 0.1,
        "cell_count": 8,
        "cell_clustering": 0.1,
    },
    2: {
        "title": "Initial Beating",
        "shape": "slightly_elongated",
        "elongation": 1.2,
        "alignment": 0.2,
        "connection": 0.2,
        "sarcomere_organization": 0.2,
        "beating_strength": 0.3,
        "beating_sync": 0.2,
        "color_base": (255, 204, 204),
        "nucleus_size": 0.65,
        "debris_level": 0.1,
        "cell_count": 12,
        "cell_clustering": 0.3,
    },
    3: {
        "title": "Mean Beating Begins",
        "shape": "elongated",
        "elongation": 1.5,
        "alignment": 0.4,
        "connection": 0.3,
        "sarcomere_organization": 0.4,
        "beating_strength": 0.5,
        "beating_sync": 0.4,
        "color_base": (255, 194, 194),
        "nucleus_size": 0.5,
        "debris_level": 0.2,
        "cell_count": 16,
        "cell_clustering": 0.5,
    },
    4: {
        "title": "Stronger Contractions",
        "shape": "well_elongated",
        "elongation": 1.8,
        "alignment": 0.6,
        "connection": 0.5,
        "sarcomere_organization": 0.6,
        "beating_strength": 0.7,
        "beating_sync": 0.6,
        "color_base": (255, 153, 153),
        "nucleus_size": 0.45,
        "debris_level": 0.2,
        "cell_count": 20,
        "cell_clustering": 0.7,
    },
    5: {
        "title": "Moderate Synchronization",
        "shape": "fully_elongated",
        "elongation": 2.0,
        "alignment": 0.8,
        "connection": 0.7,
        "sarcomere_organization": 0.8,
        "beating_strength": 0.85,
        "beating_sync": 0.8,
        "color_base": (255, 102, 102),
        "nucleus_size": 0.4,
        "debris_level": 0.3,
        "cell_count": 24,
        "cell_clustering": 0.8,
    },
    6: {
        "title": "Peak Contraction Activity",
        "shape": "fully_elongated",
        "elongation": 2.2,
        "alignment": 0.9,
        "connection": 0.9,
        "sarcomere_organization": 0.9,
        "beating_strength": 1.0,
        "beating_sync": 0.9,
        "color_base": (255, 51, 51),
        "nucleus_size": 0.4,
        "debris_level": 0.4,
        "cell_count": 28,
        "cell_clustering": 0.9,
    },
    7: {
        "title": "Damage & Fragmentation Begins",
        "shape": "fragmenting",
        "elongation": 1.6,
        "alignment": 0.5,
        "connection": 0.6,
        "sarcomere_organization": 0.5,
        "beating_strength": 0.6,
        "beating_sync": 0.5,
        "color_base": (204, 51, 51),
        "nucleus_size": 0.3,
        "debris_level": 0.7,
        "cell_count": 20,
        "cell_clustering": 0.6,
    },
    8: {
        "title": "Significant Cell Damage",
        "shape": "fragmented",
        "elongation": 1.2,
        "alignment": 0.2,
        "connection": 0.2,
        "sarcomere_organization": 0.1,
        "beating_strength": 0.2,
        "beating_sync": 0.1,
        "color_base": (153, 51, 51),
        "nucleus_size": 0.25,
        "debris_level": 0.9,
        "cell_count": 12,
        "cell_clustering": 0.2,
    }
}

# Function to interpolate characteristics between days
@st.cache_data
def interpolate_characteristics(day_float):
    day_lower = int(day_float)
    day_upper = min(8, day_lower + 1)
    fraction = day_float - day_lower
    
    # Handle day 8 (cap at day 8)
    if day_lower == 8:
        return cell_characteristics[8]
    
    # Interpolate between the two days
    interpolated = {}
    for key in cell_characteristics[day_lower]:
        if key == "title" or key == "shape":
            interpolated[key] = cell_characteristics[day_lower][key]
        elif key == "color_base":
            r1, g1, b1 = cell_characteristics[day_lower][key]
            r2, g2, b2 = cell_characteristics[day_upper][key]
            r = int(r1 + (r2 - r1) * fraction)
            g = int(g1 + (g2 - g1) * fraction)
            b = int(b1 + (b2 - b1) * fraction)
            interpolated[key] = (r, g, b)
        else:
            val1 = cell_characteristics[day_lower][key]
            val2 = cell_characteristics[day_upper][key]
            interpolated[key] = val1 + (val2 - val1) * fraction
    
    return interpolated

# Generate and display a preview frame - optimized for speed
@st.cache_data
def generate_frame(characteristics, time_point, frame_width=400, frame_height=300):
    # Create a blank white image
    image = Image.new('RGB', (frame_width, frame_height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    
    # Calculate beating phase based on time_point - simplified for speed
    beat_frequency = 1.0 + characteristics["beating_strength"]
    beat_phase = (time_point * beat_frequency) % 1.0
    beat_pulse = np.sin(beat_phase * np.pi)
    
    # Generate fewer cells for faster rendering
    # Reduce the cell count for preview
    reduced_cell_count = min(int(characteristics["cell_count"] * 0.8), 15)
    
    # Generate cell clusters - fewer clusters for speed
    num_clusters = min(3, max(2, int(characteristics["cell_clustering"] * 3)))
    cluster_centers = []
    for _ in range(num_clusters):
        cluster_centers.append((
            100 + np.random.random() * (frame_width - 200),
            100 + np.random.random() * (frame_height - 200)
        ))
    
    # Draw connections between clusters - simplified
    if characteristics["connection"] > 0.4:
        for i in range(len(cluster_centers)):
            for j in range(i+1, len(cluster_centers)):
                if np.random.random() < characteristics["connection"]:
                    x1, y1 = cluster_centers[i]
                    x2, y2 = cluster_centers[j]
                    conn_color = (*characteristics["color_base"], 120)
                    draw.line([(x1, y1), (x2, y2)], fill=conn_color, width=2)
    
    # Draw cells - using fewer, more distinct cells
    cells_drawn = 0
    clusters_used = 0
    
    # Distribute cells among clusters
    while cells_drawn < reduced_cell_count:
        # Select a cluster
        if clusters_used < len(cluster_centers):
            cluster_x, cluster_y = cluster_centers[clusters_used]
            clusters_used += 1
        else:
            cluster_index = np.random.randint(0, len(cluster_centers))
            cluster_x, cluster_y = cluster_centers[cluster_index]
        
        # Determine how many cells in this cluster - simplified
        cells_in_cluster = min(5, reduced_cell_count - cells_drawn)
        
        for _ in range(cells_in_cluster):
            # Calculate position within the cluster
            cluster_radius = 20 + 15 * characteristics["cell_clustering"]
            angle = np.random.random() * 2 * np.pi
            distance = np.random.random() * cluster_radius
            x = cluster_x + np.cos(angle) * distance
            y = cluster_y + np.sin(angle) * distance
            
            # Basic cell morphology by day
            base_width = 15
            
            # Apply beat effect - simplified
            beat_effect = 1.0 + beat_pulse * characteristics["beating_strength"] * 0.3
            
            # Calculate cell dimensions based on morphology stage
            if characteristics["shape"] == "round":
                cell_width = base_width * beat_effect
                cell_height = cell_width
            elif characteristics["shape"] in ["slightly_elongated", "elongated", "well_elongated", "fully_elongated"]:
                cell_width = base_width * beat_effect
                cell_height = base_width * characteristics["elongation"] * beat_effect
                
                # Apply alignment - simplified
                if np.random.random() < characteristics["alignment"]:
                    cell_width, cell_height = cell_height, cell_width
            else:  # fragmenting or fragmented
                if np.random.random() < 0.6 and characteristics["debris_level"] > 0.5:
                    # Draw fragments instead of whole cell
                    num_fragments = 3
                    for j in range(num_fragments):
                        frag_x = x + np.random.random() * 15 - 7
                        frag_y = y + np.random.random() * 15 - 7
                        frag_size = 4 + np.random.random() * 4
                        frag_color = (*characteristics["color_base"], 150)
                        draw.ellipse([frag_x, frag_y, frag_x+frag_size, frag_y+frag_size], fill=frag_color)
                    
                    cells_drawn += 1
                    continue
                
                cell_width = base_width * beat_effect * 0.8
                cell_height = base_width * characteristics["elongation"] * beat_effect * 0.8
            
            # Cell color
            cell_alpha = 180
            cell_color = (*characteristics["color_base"], cell_alpha)
            
            # Draw the cell
            draw.ellipse([x, y, x + cell_width, y + cell_height], fill=cell_color)
            
            # Add simplified sarcomere structure for more mature cells
            if characteristics["sarcomere_organization"] > 0.3:
                num_lines = 3
                for i in range(num_lines):
                    line_position = y + cell_height * (i + 1) / (num_lines + 1)
                    line_length = cell_width * 0.8
                    line_start = x + (cell_width - line_length) / 2
                    line_end = line_start + line_length
                    line_color = (max(characteristics["color_base"][0]-50, 0), 
                                 max(characteristics["color_base"][1]-50, 0), 
                                 max(characteristics["color_base"][2]-50, 0), 150)
                    
                    # Simplified sarcomere drawing
                    if characteristics["sarcomere_organization"] < 0.5 and np.random.random() > 0.5:
                        # Draw broken sarcomeres for damaged cells
                        segments = 2
                        for s in range(segments):
                            if np.random.random() < 0.7:
                                seg_start = line_start + (line_length * s / segments)
                                seg_end = line_start + (line_length * (s+1) / segments)
                                draw.line([(seg_start, line_position), (seg_end, line_position)], 
                                         fill=line_color, width=1)
                    else:
                        draw.line([(line_start, line_position), (line_end, line_position)], 
                                 fill=line_color, width=1)
            
            # Draw nucleus
            if np.random.random() < 0.8:  # Not all cells show nucleus in the image
                nucleus_size = characteristics["nucleus_size"]
                nucleus_width = cell_width * nucleus_size
                nucleus_height = cell_height * nucleus_size
                nucleus_x = x + (cell_width - nucleus_width) / 2
                nucleus_y = y + (cell_height - nucleus_height) / 2
                nucleus_color = (102, 102, 204, 180)
                
                # Draw nucleus
                draw.ellipse([nucleus_x, nucleus_y, nucleus_x + nucleus_width, nucleus_y + nucleus_height], 
                             fill=nucleus_color)
            
            cells_drawn += 1
    
    # Add simplified debris
    debris_count = int(characteristics["debris_level"] * 80)
    for _ in range(debris_count):
        debris_x = np.random.random() * frame_width
        debris_y = np.random.random() * frame_height
        debris_size = 1 + np.random.random() * 3
        
        # Simplified debris color
        if characteristics["debris_level"] < 0.5:
            debris_color = (180, 180, 180, 80)
        else:
            debris_color = (160, 100, 100, 100)
            
        draw.ellipse([debris_x, debris_y, debris_x + debris_size, debris_y + debris_size], 
                     fill=debris_color)
    
    # Add day indicator text
    day_decimal = 1 + (time_point / 60.0) * 7  # 60 second video covers 7 days
    day_int = int(day_decimal)
    
    # Display the day and title
    text_bg = (240, 240, 240, 180)
    draw.rectangle([10, 10, 160, 30], fill=text_bg)
    # We'd add text here in a complete implementation
    
    return image

# Streamlit app layout
st.markdown("### Animation Controls")

# Simplified UI
col1, col2 = st.columns([1, 2])

with col1:
    # Simplified controls for faster generation
    animation_speed = st.select_slider(
        "Animation Speed",
        options=["Slow", "Medium", "Fast"],
        value="Medium"
    )
    # Map the speed setting to actual duration values
    duration_map = {"Slow": 500, "Medium": 250, "Fast": 125}  # milliseconds per frame
    
    # Simplified day selection
    day_selection = st.radio(
        "Day Range to Show",
        ["All Days (1-8)", "Early (1-3)", "Middle (3-6)", "Late (6-8)"]
    )
    
    # Map the selection to day ranges
    day_range_map = {
        "All Days (1-8)": (1.0, 8.0),
        "Early (1-3)": (1.0, 3.0),
        "Middle (3-6)": (3.0, 6.0),
        "Late (6-8)": (6.0, 8.0)
    }

with col2:
    st.markdown("""
    This animation shows the continuous morphological changes in cardiomyocytes over 8 days:
    
    - **Days 1-2:** Small, immature cells with minimal beating
    - **Days 3-4:** Developing elongation and alignment with improved contractility
    - **Days 5-6:** Peak maturity with strong synchronization and organized sarcomeres
    - **Days 7-8:** Progressive damage, fragmentation and loss of function
    
    The animation demonstrates key features:
    - Cell shape changes (round → elongated → fragmented)
    - Beating patterns and synchronization
    - Sarcomere structure development
    - Cell-to-cell connections
    """)

# Initialize session state
if 'video_generated' not in st.session_state:
    st.session_state.video_generated = False
    st.session_state.video_path = None
    st.session_state.preview_cache = {}

# Show a preview frame
st.markdown("### Preview Frame")

# Create a more efficient preview system with caching
preview_day = st.slider("Preview Day", 
                       min_value=day_range_map[day_selection][0], 
                       max_value=day_range_map[day_selection][1], 
                       value=(day_range_map[day_selection][0] + day_range_map[day_selection][1])/2,
                       step=0.5)
preview_placeholder = st.empty()

# Round to nearest 0.5 to reduce number of generated frames
day_key = round(preview_day * 2) / 2

# Check if frame is already in cache
if day_key not in st.session_state.preview_cache:
    characteristics = interpolate_characteristics(day_key)
    time_point = (day_key - 1) * (60 / 7)  # Map day to time point in animation
    preview_frame = generate_frame(characteristics, time_point)
    st.session_state.preview_cache[day_key] = preview_frame
else:
    preview_frame = st.session_state.preview_cache[day_key]

preview_placeholder.image(preview_frame, caption=f"Day {day_key:.1f}")

# Add a more prominent generate button
generate_button = st.button("Generate Animation", use_container_width=True, type="primary")

# Generate video when button is clicked
if generate_button:
    with st.spinner("Generating animation..."):
        progress_bar = st.progress(0)
        
        # Use a much smaller number of frames for instant display
        # Generate more frames for the selected day range
        min_day, max_day = day_range_map[day_selection]
        
        # Calculate how many frames based on the day range
        # More frames for shorter ranges for better detail
        if max_day - min_day <= 2:
            total_keyframes = 12  # More detail for short ranges
        else:
            total_keyframes = int(16 * (max_day - min_day) / 7)  # Proportional to range size
            
        sample_frames = []
        
        # Pre-compute day positions to ensure even distribution across the day range
        day_positions = np.linspace(min_day, max_day, total_keyframes)
        
        # Generate only key frames
        for i, day_decimal in enumerate(day_positions):
            # Generate the frame for this day
            characteristics = interpolate_characteristics(day_decimal)
            time_point = (day_decimal - 1) * (60 / 7)  # Map day to seconds in 60-second timeline
            frame = generate_frame(characteristics, time_point)
            sample_frames.append(frame)
            
            # Update progress
            progress = (i + 1) / total_keyframes
            progress_bar.progress(progress)
        
        # Create an animated GIF from the frames
        # Use the selected animation speed
        gif_path = "cardiomyocyte_animation.gif"
        frame_duration = duration_map[animation_speed]
        sample_frames[0].save(
            gif_path,
            save_all=True,
            append_images=sample_frames[1:],
            optimize=False,
            duration=frame_duration,  # milliseconds between frames based on selected speed
            loop=0  # loop forever
        )
        
        st.session_state.video_generated = True
        st.session_state.video_path = gif_path
        progress_bar.progress(1.0)

# Display the video if it has been generated
if st.session_state.video_generated:
    st.markdown("### Cardiomyocyte Development Animation")
    
    # Create columns for better layout
    vid_col1, vid_col2 = st.columns([3, 1])
    
    with vid_col1:
        st.image(st.session_state.video_path)
    
    with vid_col2:
        st.download_button(
            label="Download Animation",
            data=open(st.session_state.video_path, "rb").read(),
            file_name="cardiomyocyte_development.gif",
            mime="image/gif"
        )
        
        # Add an option to regenerate with different settings
        if st.button("Regenerate Animation"):
            st.session_state.video_generated = False
            st.rerun()
    
    # Add detailed description of what's happening in the animation
    st.markdown("""
    #### Animation Details
    
    This animation shows the developmental changes in cardiomyocytes:
    
    **Morphological Changes:**
    - Cell shape evolution from round to elongated to fragmented
    - Development and breakdown of internal sarcomere structures
    - Changes in nuclear size relative to the cell
    - Formation of cell clusters and intercellular connections
    
    **Functional Changes:**
    - Increasing beating strength and coordination (Days 1-6)
    - Peak contractile activity (Day 6)
    - Deterioration of contractile function (Days 7-8)
    
    **Cell Organization:**
    - Alignment of cells into functional patterns
    - Development of cell-cell connections
    - Increased clustering in mature stages
    - Breakdown of organization in later stages
    
    The beating effect is visualized through rhythmic size changes of the cells, with stronger and more coordinated beats in the middle stages.
    """)

# Add sidebar with scientific information
st.sidebar.title("Scientific Background")
st.sidebar.markdown("""
### Cardiomyocyte Development

Cardiomyocytes are specialized muscle cells responsible for cardiac contraction. Their development involves significant morphological and functional changes:

**Morphological Changes:**
- Transition from round to rod-shaped cells
- Formation of sarcomeres (contractile units)
- Development of intercalated discs (cell-cell junctions)
- Alignment of cells into functional networks

**Functional Changes:**
- Initiation of spontaneous contraction
- Improved synchronization of beating
- Enhanced contractile force
- Formation of electromechanical coupling

**Late-Stage Deterioration:**
In vitro culture models often show deterioration after 6-7 days due to:
- Metabolic stress
- Lack of supporting cell types
- Accumulation of waste products
- Limited nutrient availability

This animation simulates these processes based on observations from cardiac cell culture experiments.
""")

# Add code attribution
st.sidebar.markdown("---")
st.sidebar.markdown("Animation created using Python, PIL, and Streamlit")
