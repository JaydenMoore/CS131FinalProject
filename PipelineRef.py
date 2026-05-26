# %% [markdown]
# # 1. Set up environment
# 
# Import all necessary libraries, datasets, and do any audio extraction if necessary.

# %%
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.io import wavfile
from scipy.spatial.transform import Rotation, Slerp
from scipy.spatial import cKDTree
import cv2
import plotly.graph_objects as go
import ast
import re
from sklearn.cluster import DBSCAN
import subprocess
import os
from IPython.display import display, HTML
import json
import itertools
import matplotlib.pyplot as plt

DataFolder = "Data/TestData5"

mov_path = None
wav_path = None
py_path = None
VIDEO_CLAP_FRAME = None

def extract_audio_from_mov(mov_path, wav_path):
    """
    Helper command to strictly extract .wav standard acoustics natively from Apple .mov containers using OSX afconvert.
    """
    subprocess.run(['afconvert', '-f', 'WAVE', '-d', 'LEI16', mov_path, wav_path])
    print(f"Extraction complete: Saved standalone wave audio to {wav_path}")

for filename in os.listdir(DataFolder):
    if filename.endswith('.mov'):
        mov_path = os.path.join(DataFolder, filename)
        wav_path = os.path.join(DataFolder, filename.replace('.mov', '.wav'))
        py_path = os.path.join(DataFolder, filename.replace('.mov', '.blender.py'))
        frame = os.path.join(DataFolder, "frame.txt")
        VIDEO_CLAP_FRAME = int(open(frame).read().strip())

        if not os.path.exists(wav_path):
            extract_audio_from_mov(mov_path, wav_path)
        break

# %% [markdown]
# # 2. Parsing and Synchronization
# 
# Parse the pose data and intrinsics from the blender.py file exported from the AR Camera app. 

# %%
def parse_blender_tracking_log(py_path, fps=60.0):
    with open(py_path, 'r') as f:
        content = f.read()
    fps_match = re.search(r'bpy\.context\.scene\.render\.fps\s*=\s*(\d+)', content)
    fps = float(fps_match.group(1)) if fps_match else fps
    
    frames = ast.literal_eval(re.search(r'movement_keyframes\s*=\s*(\[.*?\])', content, re.DOTALL).group(1))
    locations = ast.literal_eval(re.search(r'locations\s*=\s*(\[.*?\])', content, re.DOTALL).group(1))
    rotations = ast.literal_eval(re.search(r'rotations\s*=\s*(\[.*?\])', content, re.DOTALL).group(1))
    
    timestamps, translations, rotation_matrices = [], [], []
    for frame, loc, rot in zip(frames, locations, rotations):
        timestamps.append(frame / fps)
        translations.append(loc)
        r = Rotation.from_euler('ZXY', [rot[2], rot[0], rot[1]], degrees=False)
        rotation_matrices.append(r.as_matrix())
        
    return pd.DataFrame({'timestamp': timestamps, 'translation': translations, 'rotation_matrix': rotation_matrices})


def parse_blender_intrinsics(py_path):
    with open(py_path, 'r') as f:
        content = f.read()
    width = float(re.search(r'resolution_x\s*=\s*([\d.]+)', content).group(1))
    height = float(re.search(r'resolution_y\s*=\s*([\d.]+)', content).group(1))
    lens = float(re.search(r'lens\s*=\s*([\d.]+)', content).group(1))
    sensor_width = float(re.search(r'sensor_width\s*=\s*([\d.]+)', content).group(1))
    
    fx = (lens / sensor_width) * width
    return np.array([[fx, 0, width/2.0], [0, fx, height/2.0], [0, 0, 1]]), int(width), int(height)

# %% [markdown]
# # 3. Signal Processing
# 
# Computing the distances to the walls using time-of-flight calculations and sound-to-noise thresholding.

# %%
def compute_acoustic_distances(audio_sig, trajectory_df, sample_rate, reference_chirp, snr_threshold=3.0):
    SPEED_OF_SOUND = 343.0  
    if audio_sig.ndim > 1: audio_sig = audio_sig[:, 0]
    timestamps = trajectory_df['timestamp'].to_numpy()
    distances = np.full(len(timestamps), np.nan)
    window_samples = int(0.100 * sample_rate)
    
    center_indices = (timestamps * sample_rate).astype(int)
    for i, idx_center in enumerate(center_indices):
        if idx_center + window_samples > len(audio_sig): continue
        audio_window = audio_sig[idx_center : idx_center + window_samples]
        if len(audio_window) < len(reference_chirp): continue
            
        xcorr = signal.correlate(audio_window, reference_chirp, mode='valid')
        
        # SNR Calculation Constraint
        noise_floor = np.sqrt(np.mean(audio_window**2)) + 1e-6 
        peaks, _ = signal.find_peaks(xcorr, height=noise_floor * snr_threshold, distance=int(0.001 * sample_rate))
        
        if len(peaks) >= 2:
            delay = (peaks[1] - peaks[0]) / sample_rate
            dist = (SPEED_OF_SOUND * delay) / 2.0
            if dist <= 1.2: 
                distances[i] = dist
    return distances

# %% [markdown]
# # 4. Hough Accumulator (Placeholder)
# 
# Build the hough accumulator using the distances and camera pose data with selective filtering to remove noisy voxels. (This needs to be rewritten from scratch as AI did most of the heavy lifting).

# %%
def build_hough_accumulator(pose_df, acoustic_distances, voxel_size=0.04, margin=1.5):
    trajectory = np.array(pose_df['translation'].tolist())
    rotations = np.array(pose_df['rotation_matrix'].tolist())
    min_bounds = np.min(trajectory, axis=0) - margin
    max_bounds = np.max(trajectory, axis=0) + margin
    
    grid_dims = np.ceil((max_bounds - min_bounds) / voxel_size).astype(int)
    voxel_grid = np.zeros(grid_dims, dtype=np.int32)
    
    x_bins = min_bounds[0] + np.arange(grid_dims[0]) * voxel_size
    y_bins = min_bounds[1] + np.arange(grid_dims[1]) * voxel_size
    z_bins = min_bounds[2] + np.arange(grid_dims[2]) * voxel_size
    
    for i, (C_i, d_i) in enumerate(zip(trajectory, acoustic_distances)):
        if np.isnan(d_i) or d_i <= 0: continue
        
        idx_min = np.maximum(0, np.floor((C_i - (d_i + voxel_size) - min_bounds) / voxel_size)).astype(int)
        idx_max = np.minimum(grid_dims, np.ceil((C_i + (d_i + voxel_size) - min_bounds) / voxel_size)).astype(int)
        
        ix, iy, iz = np.arange(idx_min[0], idx_max[0]), np.arange(idx_min[1], idx_max[1]), np.arange(idx_min[2], idx_max[2])
        if len(ix)==0 or len(iy)==0 or len(iz)==0: continue

        X_s, Y_s, Z_s = np.meshgrid(x_bins[ix], y_bins[iy], z_bins[iz], indexing='ij')
        
        grid_vectors = np.stack([X_s - C_i[0], Y_s - C_i[1], Z_s - C_i[2]], axis=-1)
        dist_to_cam = np.linalg.norm(grid_vectors, axis=-1)
        
        forward_world = -rotations[i][:, 2] 
        dot_prod = np.dot(grid_vectors, forward_world) / (dist_to_cam + 1e-6)
        
        voxel_grid[idx_min[0]:idx_max[0], idx_min[1]:idx_max[1], idx_min[2]:idx_max[2]] += (np.abs(dist_to_cam - d_i) < voxel_size) & (dot_prod >= 0.707)
        
    return voxel_grid, min_bounds, voxel_size

# %% [markdown]
# # 5. Generate Rectangular Geometry (Placeholder)
# 
# Generates rectangular mesh with 4 enclosing planes to represent the walls of the room. (This needs to be rewritten from scratch as AI did most of the heavy lifting).

# %%
def generate_balanced_room_mesh(
    voxel_points,
    trajectory_df,
    residual_threshold=0.08,
    resolution=0.03,
    dbscan_eps=None,
    dbscan_min_samples=4,
    cluster_keep_ratio=0.35,
    max_clusters_to_keep=4,
    return_debug=False,
):
    """
    Orientation <- balanced/global Hough points
    Extents     <- full Hough cloud (so box follows global Hough orientation)
    RANSAC is deterministic (seeded RNG).
    """
    debug_info = {
        "input_point_count": int(len(voxel_points)),
        "ransac_point_count": 0,
        "dbscan_cluster_count": 0,
        "dbscan_noise_count": 0,
        "kept_point_count": 0,
        "wall_segment_count": 0,
        "dominant_wall_angle_deg": 0.0,
        "rect_width": 0.0,
        "rect_depth": 0.0,
        "rect_angle_deg": 0.0,
        "room_fit_source": "none",
    }

    if len(voxel_points) == 0:
        if return_debug:
            return np.empty((0, 3)), debug_info
        return np.empty((0, 3))

    # full Hough 2D (used for extents)
    points_2d = np.float32(voxel_points[:, [0, 2]])
    print(f"generate_balanced_room_mesh input point count: {len(points_2d)}")

    rng = np.random.default_rng(42)

    def diversify_points_spatial(points, cell_size=0.18, per_cell_cap=10, min_global=400):
        if len(points) <= 3:
            return points
        mins = points.min(axis=0)
        cell_idx = np.floor((points - mins) / max(cell_size, 1e-6)).astype(np.int32)
        hash_mult = np.array([73856093, 19349663], dtype=np.int64)
        cell_hash = np.dot(cell_idx, hash_mult)
        kept = []
        for h in np.unique(cell_hash):
            idx = np.where(cell_hash == h)[0]
            if len(idx) <= per_cell_cap:
                kept.extend(idx.tolist())
                continue
            local = points[idx]
            d = np.linalg.norm(local - local.mean(axis=0), axis=1)
            order = np.argsort(-d)
            boundary_idx = idx[order[: max(1, per_cell_cap // 3)]]
            remaining = np.setdiff1d(idx, boundary_idx, assume_unique=False)
            random_take = per_cell_cap - len(boundary_idx)
            if random_take > 0 and len(remaining) > random_take:
                remaining = rng.choice(remaining, size=random_take, replace=False)
            kept.extend(boundary_idx.tolist())
            if random_take > 0:
                kept.extend(np.atleast_1d(remaining).tolist())
        kept = np.unique(np.array(kept, dtype=np.int32))
        target_min = min(min_global, len(points))
        if len(kept) < target_min:
            d_global = np.linalg.norm(points - points.mean(axis=0), axis=1)
            extra_idx = np.argsort(-d_global)[: target_min - len(kept)]
            kept = np.unique(np.concatenate([kept, extra_idx.astype(np.int32)]))
        return np.float32(points[kept])

    def build_dense_mesh_from_box(box_2d):
        y_vals = trajectory_df["translation"].apply(lambda x: x[1])
        y_min, y_max = y_vals.min() - 0.2, y_vals.max() + 1.5
        height_grid = np.arange(y_min, y_max, resolution)
        dense_points = []
        for i in range(4):
            p1, p2 = box_2d[i], box_2d[(i + 1) % 4]
            num_steps = max(2, int(np.linalg.norm(p2 - p1) / resolution))
            t = np.linspace(0, 1, num_steps)
            xz_line = p1[None, :] + t[:, None] * (p2 - p1)[None, :]
            xz_repeated = np.repeat(xz_line, len(height_grid), axis=0)
            y_repeated = np.tile(height_grid, num_steps)
            dense_points.append(np.column_stack([xz_repeated[:, 0], y_repeated, xz_repeated[:, 1]]))
        return np.vstack(dense_points) if dense_points else np.empty((0, 3))

    # balanced Hough used only to compute stable orientation
    balance_cell = max(0.10, resolution * 6.0)
    balanced_points_2d = diversify_points_spatial(points_2d, cell_size=balance_cell, per_cell_cap=10)
    print(f"Balanced Hough point count for fitting: {len(balanced_points_2d)}")

    # ---- RANSAC denoiser on balanced set (deterministic) ----
    clean_wall_points = []
    pts_pool = balanced_points_2d.copy()
    for _ in range(10):
        if len(pts_pool) < 20:
            break
        best_inliers = []
        for _ in range(500):
            if len(pts_pool) < 2:
                break
            idx = rng.choice(len(pts_pool), 2, replace=False)
            p1, p2 = pts_pool[idx]
            vec = p2 - p1
            length = np.linalg.norm(vec)
            if length < 0.01:
                continue
            normal = np.array([-vec[1], vec[0]]) / length
            C = -np.dot(normal, p1)
            dist = np.abs(np.dot(pts_pool, normal) + C)
            inliers = np.where(dist < residual_threshold)[0]
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
        if len(best_inliers) >= 20:
            clean_wall_points.append(pts_pool[best_inliers])
            pts_pool = np.delete(pts_pool, best_inliers, axis=0)
        else:
            break

    combined_clean_points = balanced_points_2d if not clean_wall_points else np.vstack(clean_wall_points)
    debug_info["ransac_point_count"] = int(len(combined_clean_points))
    print(f"RANSAC-clean point count: {len(combined_clean_points)}")

    # ---- DBSCAN cleanup on the cleaned set ----
    if len(combined_clean_points) >= 3:
        clustering_eps = dbscan_eps if dbscan_eps is not None else max(resolution * 4, 0.12)
        clustering = DBSCAN(eps=clustering_eps, min_samples=dbscan_min_samples)
        cluster_labels = clustering.fit_predict(combined_clean_points)
        valid_mask = cluster_labels != -1
        valid_points = combined_clean_points[valid_mask]
        unique_labels, counts = (np.unique(cluster_labels[valid_mask], return_counts=True) if np.any(valid_mask) else (np.array([]), np.array([])))
        noise_count = int(np.sum(cluster_labels == -1))
        debug_info["dbscan_cluster_count"] = int(len(unique_labels))
        debug_info["dbscan_noise_count"] = noise_count
        print(f"DBSCAN clusters found: {len(unique_labels)} (noise points: {noise_count})")
        if len(unique_labels) > 0:
            cluster_sizes = {label: count for label, count in zip(unique_labels.tolist(), counts.tolist())}
            largest_cluster_size = max(cluster_sizes.values())
            min_cluster_size = max(3, int(np.ceil(largest_cluster_size * cluster_keep_ratio)))
            ordered_labels = sorted(cluster_sizes.keys(), key=lambda label: cluster_sizes[label], reverse=True)
            kept_clusters = [
                combined_clean_points[cluster_labels == label]
                for label in ordered_labels[:max_clusters_to_keep]
                if cluster_sizes[label] >= min_cluster_size
            ]
            if kept_clusters:
                combined_clean_points = np.vstack(kept_clusters)
            elif len(valid_points) >= 3:
                combined_clean_points = valid_points
            else:
                print("DBSCAN removed too many points; falling back to RANSAC-clean points.")
        else:
            print("DBSCAN found only noise; falling back to RANSAC-clean points.")
    else:
        print("DBSCAN skipped due to insufficient points; falling back to unclustered clean points.")
    debug_info["kept_point_count"] = int(len(combined_clean_points))

    if len(combined_clean_points) < 3:
        print("Not enough clustered points for room fitting; returning empty mesh.")
        if return_debug:
            return np.empty((0, 3)), debug_info
        return np.empty((0, 3))

    # -----------------------
    # Orientation from balanced Hough (stable)
    # -----------------------
    orient_src = balanced_points_2d if len(balanced_points_2d) >= 20 else points_2d
    try:
        rect = cv2.minAreaRect(np.float32(orient_src))
        w, h = rect[1]
        angle = float(rect[2])
        if w < h:
            angle += 90.0
    except Exception:
        centered = orient_src - orient_src.mean(axis=0)
        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        major_axis = eigvecs[:, int(np.argmax(eigvals))]
        angle = float(np.degrees(np.arctan2(major_axis[1], major_axis[0])))

    rect_angle_deg = ((angle + 180.0) % 360.0) - 180.0
    debug_info["dominant_wall_angle_deg"] = rect_angle_deg

    theta = np.radians(rect_angle_deg)
    cth, sth = float(np.cos(theta)), float(np.sin(theta))
    R_local_to_world = np.array([[cth, -sth], [sth, cth]], dtype=np.float32)
    R_world_to_local = R_local_to_world.T

    # -----------------------
    # Extents: compute from FULL Hough cloud (points_2d) rotated into Hough-local frame
    # -----------------------
    local_hough = (R_world_to_local @ points_2d.T).T
    pad = 0.01
    low = np.percentile(local_hough, 2.0, axis=0) - pad
    high = np.percentile(local_hough, 98.0, axis=0) + pad

    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)) or np.any(high <= low):
        # fallback to cleaned points axis-aligned
        mesh_xz = combined_clean_points
        mesh_min_x, mesh_max_x = np.min(mesh_xz[:, 0]), np.max(mesh_xz[:, 0])
        mesh_min_z, mesh_max_z = np.min(mesh_xz[:, 1]), np.max(mesh_xz[:, 1])
        box_world = np.array(
            [
                [mesh_min_x, mesh_min_z],
                [mesh_max_x, mesh_min_z],
                [mesh_max_x, mesh_max_z],
                [mesh_min_x, mesh_max_z],
            ],
            dtype=np.float32,
        )
        rect_width = float(mesh_max_x - mesh_min_x)
        rect_depth = float(mesh_max_z - mesh_min_z)
        rect_angle_deg = 0.0
        debug_info["room_fit_source"] = "axis_aligned_fallback"
    else:
        box_local = np.array(
            [
                [low[0], low[1]],
                [high[0], low[1]],
                [high[0], high[1]],
                [low[0], high[1]],
            ],
            dtype=np.float32,
        )
        box_world = (R_local_to_world @ box_local.T).T
        rect_width = float(high[0] - low[0])
        rect_depth = float(high[1] - low[1])
        rect_angle_deg = rect_angle_deg
        debug_info["room_fit_source"] = "hough_oriented_full_extents"

    debug_info["rect_width"] = rect_width
    debug_info["rect_depth"] = rect_depth
    debug_info["rect_angle_deg"] = rect_angle_deg
    debug_info["wall_segment_count"] = 0

    print(f"Room fit source={debug_info['room_fit_source']}: width={rect_width:.3f}, depth={rect_depth:.3f}, angle={rect_angle_deg:.2f} deg")

    dense_mesh = build_dense_mesh_from_box(box_world)

    if return_debug:
        return dense_mesh, debug_info
    return dense_mesh

# %% [markdown]
# # 6. Project Textures onto the Mesh (Placeholder)
# 
# Uses the camera pinhole model to project the pixels from sampled images onto 3D space. (This needs to be rewritten from scratch as AI did most of the heavy lifting).

# %%
def project_textures_to_mesh(mesh_points, video_path, pose_df, K, video_clap_frame, max_keyframes=45):
    cap = cv2.VideoCapture(video_path)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    point_colors = np.ones((len(mesh_points), 3)) * 0.15 
    best_depth = np.full(len(mesh_points), np.inf) 
    
    times = pose_df['timestamp'].values
    t_min, t_max = times.min(), times.max()
    
    candidates, frame_idx = [], 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        t_query = (frame_idx - video_clap_frame) / video_fps
        if t_min <= t_query <= t_max:
            if frame_idx % 10 == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                candidates.append((frame_idx, np.var(gray), t_query))
        frame_idx += 1
        
    candidates.sort(key=lambda x: x[1], reverse=True)
    keyframes = []
    
    for cand in candidates:
        if all(abs(cand[2] - kf[2]) >= 1 for kf in keyframes):
            keyframes.append(cand)
            if len(keyframes) >= max_keyframes:
                break
            
    translations = np.array(pose_df['translation'].tolist())
    rotations = np.array(pose_df['rotation_matrix'].tolist())
    
    for f_idx, _, t_query in keyframes:
        idx = np.clip(np.searchsorted(times, t_query) - 1, 0, len(times) - 2)
        weight_t = (t_query - times[idx]) / (times[idx+1] - times[idx])
        
        t_w = (1 - weight_t) * translations[idx] + weight_t * translations[idx+1]
        slerp = Slerp(times[idx:idx+2], Rotation.from_matrix(rotations[idx:idx+2]))
        R_cw = slerp([t_query]).as_matrix()[0]
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret: continue
            
        X_cam = np.dot(mesh_points - t_w, R_cw)
        
        X_opencv = X_cam * np.array([1, -1, -1])
        
        valid_idx = np.where(X_opencv[:, 2] > 0.1)[0]
        if len(valid_idx) == 0: continue
            
        X_v = X_opencv[valid_idx]
        u = (X_v[:, 0] * K[0,0] / X_v[:, 2]) + K[0,2]
        v = (X_v[:, 1] * K[1,1] / X_v[:, 2]) + K[1,2]
        
        in_frame = (u >= 0) & (u < width - 1) & (v >= 0) & (v < height - 1)
        in_frame_idx = valid_idx[in_frame]
        if len(in_frame_idx) == 0: continue
            
        current_depths = X_v[in_frame, 2]
        better_view_mask = current_depths < best_depth[in_frame_idx]
        
        if not np.any(better_view_mask): continue
            
        update_idx = in_frame_idx[better_view_mask]
        u_win = u[in_frame][better_view_mask].astype(np.float32)
        v_win = v[in_frame][better_view_mask].astype(np.float32)
        
        sampled = cv2.remap(
            frame, 
            u_win.reshape(1, -1), 
            v_win.reshape(1, -1), 
            interpolation=cv2.INTER_LINEAR
        ).reshape(-1, 3)
        
        point_colors[update_idx] = sampled[:, ::-1] / 255.0
        best_depth[update_idx] = current_depths[better_view_mask]
        
    cap.release()
    point_colors = np.clip(point_colors, 0.0, 1.0)
    
    colors_uint8 = (point_colors * 255).astype(np.uint8)
    plotly_colors = [f'rgb({c[0]}, {c[1]}, {c[2]})' for c in colors_uint8]
    
    return plotly_colors

# %% [markdown]
# # 7. Visualization
# 
# Outputs 3D interactive plotly models of the textured mesh and hough accumulator voxels.

# %%
def display_reconstruction_results(ideal_room_mesh, mesh_colors, hough_points, trajectory_df):
    # ---------------------------------------------------------
    # 0. EXTRACT CAMERA TRAJECTORY
    # ---------------------------------------------------------
    if 'x' in trajectory_df.columns and 'z' in trajectory_df.columns:
        cam_x, cam_y, cam_z = trajectory_df['x'], trajectory_df['y'], trajectory_df['z']
    elif 'translation' in trajectory_df.columns:
        traj = np.array(trajectory_df['translation'].tolist())
        cam_x, cam_y, cam_z = traj[:, 0], traj[:, 1], traj[:, 2]

    # Shared Dark Mode Styling
    dark_layout = dict(
        template="plotly_dark",
        paper_bgcolor='black',
        plot_bgcolor='black',
        scene=dict(
            aspectmode='data',
            xaxis=dict(showbackground=False, gridcolor='#333333'), 
            yaxis=dict(showbackground=False, gridcolor='#333333'),
            zaxis=dict(showbackground=False, gridcolor='#333333')
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # =========================================================
    # VIEW 1: PURE TEXTURED RECONSTRUCTION
    # =========================================================
    fig1 = go.Figure()

    # Textured Mesh
    fig1.add_trace(go.Scatter3d(
        x=ideal_room_mesh[:, 0], y=ideal_room_mesh[:, 1], z=ideal_room_mesh[:, 2],
        mode='markers',
        marker=dict(size=3, color=mesh_colors, opacity=0.9),
        name='Textured Walls'
    ))

    # Camera Trajectory
    fig1.add_trace(go.Scatter3d(
        x=cam_x, y=cam_y, z=cam_z,
        mode='lines',
        line=dict(color='white', width=4),
        name='Camera Trajectory'
    ))

    fig1.update_layout(**dark_layout, title=dict(text="1. Pure Textured Reconstruction", font=dict(color='white')))
    fig1.show()

    # =========================================================
    # VIEW 2: TEXTURED MESH + HOUGH POINTS OVERLAY
    # =========================================================
    fig2 = go.Figure()

    # Raw Hough Points (Cyan)
    if len(hough_points) > 0:
        fig2.add_trace(go.Scatter3d(
            x=hough_points[:, 0], y=hough_points[:, 1], z=hough_points[:, 2],
            mode='markers',
            marker=dict(size=2, color='cyan', opacity=0.55),
            name='Raw Hough Points'
        ))

    # Textured Mesh
    fig2.add_trace(go.Scatter3d(
        x=ideal_room_mesh[:, 0], y=ideal_room_mesh[:, 1], z=ideal_room_mesh[:, 2],
        mode='markers',
        marker=dict(size=3, color=mesh_colors, opacity=0.9),
        name='Textured Walls'
    ))

    # Camera Trajectory
    fig2.add_trace(go.Scatter3d(
        x=cam_x, y=cam_y, z=cam_z,
        mode='lines',
        line=dict(color='white', width=4),
        name='Camera Trajectory'
    ))

    fig2.update_layout(**dark_layout, title=dict(text="2. Textured Reconstruction + Hough Point Cloud", font=dict(color='white')))
    fig2.show()

# %% [markdown]
# # 8. Tune parameters
# 
# Tune the parameters to find the best parameters to apply per dataset (that gets a result closest to the camera maximum bounding box).

# %%
def find_dataset_files(data_folder):
    mov_file = None
    py_file = None
    wav_file = None
    frame_file = os.path.join(data_folder, "frame.txt")

    for filename in os.listdir(data_folder):
        if filename.endswith(".mov"):
            mov_file = os.path.join(data_folder, filename)
        elif filename.endswith(".blender.py"):
            py_file = os.path.join(data_folder, filename)
        elif filename.endswith(".wav"):
            wav_file = os.path.join(data_folder, filename)

    if mov_file is None or py_file is None:
        raise FileNotFoundError(f"Missing .mov or .blender.py in {data_folder}")

    if wav_file is None:
        wav_file = os.path.join(data_folder, os.path.basename(mov_file).replace(".mov", ".wav"))
        if not os.path.exists(wav_file):
            extract_audio_from_mov(mov_file, wav_file)

    if not os.path.exists(frame_file):
        raise FileNotFoundError(f"Missing frame.txt in {data_folder}")

    video_clap_frame = int(open(frame_file).read().strip())
    return mov_file, wav_file, py_file, video_clap_frame

def compute_camera_path_bounds(trajectory_df):
    trajectory = np.array(trajectory_df["translation"].tolist())
    x_min, _, z_min = trajectory.min(axis=0)
    x_max, _, z_max = trajectory.max(axis=0)
    camera_width = float(x_max - x_min)
    camera_depth = float(z_max - z_min)
    camera_area = float(camera_width * camera_depth)
    return {
        "x_min": float(x_min),
        "x_max": float(x_max),
        "z_min": float(z_min),
        "z_max": float(z_max),
        "camera_width": camera_width,
        "camera_depth": camera_depth,
        "camera_area": camera_area,
    }

def save_best_params(folder, params, save_to_data_folder=True):
    os.makedirs("diagnostics", exist_ok=True)
    base = os.path.basename(folder.rstrip("/"))
    diag_path = f"diagnostics/{base}_best_params.json"
    with open(diag_path, "w") as fh:
        json.dump(params, fh, indent=2)

    data_path = None
    if save_to_data_folder:
        try:
            data_path = os.path.join(folder, "best_params.json")
            with open(data_path, "w") as fh:
                json.dump(params, fh, indent=2)
        except Exception as e:
            print(f"Warning: failed saving best_params to {folder}: {e}")
            data_path = None

    return diag_path, data_path


def load_best_params(folder):
    data_path = os.path.join(folder, "best_params.json")
    if os.path.exists(data_path):
        try:
            with open(data_path, "r") as fh:
                return json.load(fh)
        except Exception:
            pass
    base = os.path.basename(folder.rstrip("/"))
    diag_path = f"diagnostics/{base}_best_params.json"
    if os.path.exists(diag_path):
        try:
            with open(diag_path, "r") as fh:
                params = json.load(fh)
            try:
                with open(data_path, "w") as fh:
                    json.dump(params, fh, indent=2)
            except Exception:
                pass
            return params
        except Exception:
            pass

    return None

def score_fit_against_camera(fit_area, camera_area, hough_points, fit_box=None):
    if camera_area <= 0:
        return -np.inf, np.nan, np.nan

    fit_ratio = fit_area / camera_area

    coverage = 0.0
    if fit_box is not None and len(hough_points) > 0:
        edge_pts = []
        for i in range(4):
            p1 = fit_box[i]
            p2 = fit_box[(i + 1) % 4]
            for t in np.linspace(0.0, 1.0, 25):
                edge_pts.append(p1 * (1.0 - t) + p2 * t)
        edge_pts = np.asarray(edge_pts)

        dists = np.sqrt(
            (edge_pts[:, None, 0] - hough_points[None, :, 0]) ** 2
            + (edge_pts[:, None, 1] - hough_points[None, :, 2]) ** 2
        )
        coverage = float(np.mean(np.any(dists < 0.18, axis=1)))

    score = coverage - 0.65 * abs(1.0 - fit_ratio)
    return score, fit_ratio, coverage

def run_parameter_sweep(compare_folders, voxel_size=None):
    voxel_size = voxel_size if voxel_size is not None else globals().get("HOUGH_VOXEL_SIZE", 0.04)
    video_fps = globals().get("VIDEO_FPS", 60.0)
    os.makedirs("diagnostics", exist_ok=True)

    dbscan_eps_list = [0.12, 0.18, 0.24]
    cluster_keep_ratio_list = [0.15, 0.25, 0.35]
    sparse_percentile_list = [85, 90, 95]
    residual_list = [0.06, 0.08, 0.10]

    results = []

    for folder in compare_folders:
        mov_file, wav_file, py_file, video_clap_frame = find_dataset_files(folder)
        trajectory_df = parse_blender_tracking_log(py_file, fps=video_fps)
        trajectory_df["timestamp"] = trajectory_df["timestamp"] - (video_clap_frame / video_fps)
        camera_bounds = compute_camera_path_bounds(trajectory_df)

        sr, audio_sig = wavfile.read(wav_file)
        t_chirp = np.linspace(0, 0.010, int(44100 * 0.010), endpoint=False)
        tapered_chirp = np.sin(
            2 * np.pi * (18000 + (22000 - 18000) * t_chirp / (2 * 0.010))
        ) * signal.windows.tukey(len(t_chirp), alpha=0.1)

        acoustic_distances = compute_acoustic_distances(audio_sig, trajectory_df, sr, tapered_chirp)
        voxel_grid, bounds, v_scale = build_hough_accumulator(
            trajectory_df, acoustic_distances, voxel_size=voxel_size
        )

        nonzero_voxels = voxel_grid[voxel_grid > 0]
        if len(nonzero_voxels) == 0:
            print(f"No Hough voxels for {folder}")
            continue

        for db_eps, keep_ratio, sparse_pct, resid in itertools.product(
            dbscan_eps_list,
            cluster_keep_ratio_list,
            sparse_percentile_list,
            residual_list,
        ):
            thresh = float(np.percentile(nonzero_voxels, sparse_pct))
            voxel_indices = np.argwhere(voxel_grid >= thresh)
            hough_points = bounds + (voxel_indices * v_scale) if len(voxel_indices) > 0 else np.empty((0, 3))

            if len(hough_points) < 3:
                results.append(
                    {
                        "folder": folder,
                        "dbscan_eps": db_eps,
                        "cluster_keep_ratio": keep_ratio,
                        "sparse_percentile": sparse_pct,
                        "residual_threshold": resid,
                        "score": -np.inf,
                        "fit_ratio": np.nan,
                        "coverage": 0.0,
                        "hough_count": int(len(hough_points)),
                    }
                )
                continue

            mesh, dbg = generate_balanced_room_mesh(
                hough_points,
                trajectory_df,
                residual_threshold=resid,
                resolution=0.03,
                dbscan_eps=db_eps,
                cluster_keep_ratio=keep_ratio,
                return_debug=True,
            )

            fit_area = float(dbg["rect_width"] * dbg["rect_depth"]) if dbg["rect_width"] > 0 and dbg["rect_depth"] > 0 else 0.0
            fit_box = None

            try:
                rect = cv2.minAreaRect(np.float32(hough_points[:, [0, 2]]))
                fit_box = cv2.boxPoints(rect)
            except Exception:
                pass

            score, fit_ratio, coverage = score_fit_against_camera(
                fit_area,
                camera_bounds["camera_area"],
                hough_points,
                fit_box=fit_box,
            )

            results.append(
                {
                    "folder": folder,
                    "dbscan_eps": float(db_eps),
                    "cluster_keep_ratio": float(keep_ratio),
                    "sparse_percentile": int(sparse_pct),
                    "residual_threshold": float(resid),
                    "score": float(score),
                    "fit_ratio": float(fit_ratio),
                    "coverage": float(coverage),
                    "hough_count": int(len(hough_points)),
                    "mesh_count": int(len(mesh)),
                }
            )

    sweep_df = pd.DataFrame(results)
    sweep_csv = "diagnostics/parameter_sweep_results.csv"
    sweep_df.to_csv(sweep_csv, index=False)
    print(f"Saved parameter sweep results -> {sweep_csv}")

    best_rows = []
    for folder in sweep_df["folder"].unique():
        best = sweep_df[sweep_df["folder"] == folder].sort_values("score", ascending=False).iloc[0]
        params = {
            "dbscan_eps": float(best["dbscan_eps"]),
            "cluster_keep_ratio": float(best["cluster_keep_ratio"]),
            "sparse_percentile": int(best["sparse_percentile"]),
            "residual_threshold": float(best["residual_threshold"]),
        }
        json_path = save_best_params(folder, params)
        best_rows.append(
            {
                "folder": folder,
                "json_path": json_path,
                "score": float(best["score"]),
                "fit_ratio": float(best["fit_ratio"]),
                "coverage": float(best["coverage"]),
                "params": params,
            }
        )
        print(f"Best params for {folder} -> {params}")

    return sweep_df, pd.DataFrame(best_rows)

compare_folders = ["Data/TestData1", "Data/TestData2", "Data/TestData3", "Data/TestData4", "Data/TestData5"]
sweep_df, best_params_df = run_parameter_sweep(compare_folders, voxel_size=globals().get("HOUGH_VOXEL_SIZE", 0.04))
print(best_params_df)

# %% [markdown]
# # 9. Apply and Visualize
# 
# Apply the tuned parameters found and then visualize the models with 3D Plotly graphs.

# %%
def auto_tune_and_apply(compare_folders, voxel_size=None):
    voxel_size = voxel_size if voxel_size is not None else globals().get("HOUGH_VOXEL_SIZE", 0.04)
    video_fps = globals().get("VIDEO_FPS", 60.0)
    os.makedirs("diagnostics", exist_ok=True)

    summary = []

    for folder in compare_folders:
        params = load_best_params(folder)
        if params is None:
            print(f"No saved params for {folder}; run the sweep first.")
            continue

        print(f"\n--- Auto-tuning {folder} with {params} ---")
        mov_file, wav_file, py_file, video_clap_frame = find_dataset_files(folder)

        trajectory_df = parse_blender_tracking_log(py_file, fps=video_fps)
        trajectory_df["timestamp"] = trajectory_df["timestamp"] - (video_clap_frame / video_fps)
        K, _, _ = parse_blender_intrinsics(py_file)

        sr, audio_sig = wavfile.read(wav_file)
        t_chirp = np.linspace(0, 0.010, int(44100 * 0.010), endpoint=False)
        tapered_chirp = np.sin(
            2 * np.pi * (18000 + (22000 - 18000) * t_chirp / (2 * 0.010))
        ) * signal.windows.tukey(len(t_chirp), alpha=0.1)

        acoustic_distances = compute_acoustic_distances(audio_sig, trajectory_df, sr, tapered_chirp)
        voxel_grid, bounds, v_scale = build_hough_accumulator(
            trajectory_df, acoustic_distances, voxel_size=voxel_size
        )

        nonzero = voxel_grid[voxel_grid > 0]
        if len(nonzero) == 0:
            print(f"No Hough voxels for {folder}")
            continue

        thresh = float(np.percentile(nonzero, params["sparse_percentile"]))
        voxel_indices = np.argwhere(voxel_grid >= thresh)
        hough_points = bounds + (voxel_indices * v_scale) if len(voxel_indices) > 0 else np.empty((0, 3))

        ideal_room_mesh, room_debug = generate_balanced_room_mesh(
            hough_points,
            trajectory_df,
            residual_threshold=params["residual_threshold"],
            resolution=0.03,
            dbscan_eps=params["dbscan_eps"],
            cluster_keep_ratio=params["cluster_keep_ratio"],
            return_debug=True,
        )

        mesh_path = f"diagnostics/{os.path.basename(folder)}_final_mesh.npy"
        np.save(mesh_path, ideal_room_mesh)

        camera_bounds = compute_camera_path_bounds(trajectory_df)
        fitted_area = (
            float(room_debug["rect_width"] * room_debug["rect_depth"])
            if room_debug["rect_width"] > 0 and room_debug["rect_depth"] > 0
            else 0.0
        )
        fit_ratio = fitted_area / camera_bounds["camera_area"] if camera_bounds["camera_area"] > 0 else np.nan

        try:
            mesh_colors = project_textures_to_mesh(
                ideal_room_mesh, mov_file, trajectory_df, K, video_clap_frame
            )
            colors_path = f"diagnostics/{os.path.basename(folder)}_mesh_colors.npy"
            np.save(colors_path, np.array(mesh_colors))
        except Exception as e:
            colors_path = None
            mesh_colors = None
            print(f"Texture projection failed for {folder}: {e}")

        traj_arr = np.array(trajectory_df["translation"].tolist())
        traj_2d = np.float32(traj_arr[:, [0, 2]])

        try:
            cam_rect = cv2.minAreaRect(traj_2d)
            cam_box = cv2.boxPoints(cam_rect)
        except Exception:
            cam_box = None

        try:
            hough_2d = np.float32(hough_points[:, [0, 2]])
            fit_rect = cv2.minAreaRect(hough_2d)
            fit_box = cv2.boxPoints(fit_rect)
        except Exception:
            fit_box = None

        plt.figure(figsize=(6, 6))
        if len(hough_points) > 0:
            plt.scatter(hough_points[:, 0], hough_points[:, 2], s=2, c="cyan", alpha=0.6, label="Hough points")
        plt.plot(traj_2d[:, 0], traj_2d[:, 1], "-k", lw=1, label="Camera path")
        if cam_box is not None:
            cam_box_closed = np.vstack([cam_box, cam_box[0]])
            plt.plot(cam_box_closed[:, 0], cam_box_closed[:, 1], "-r", lw=2, label="Camera rect")
        if fit_box is not None:
            fit_box_closed = np.vstack([fit_box, fit_box[0]])
            plt.plot(fit_box_closed[:, 0], fit_box_closed[:, 1], "-g", lw=2, label="Fitted rect")
        plt.gca().set_aspect("equal", "box")
        plt.title(f"{folder} auto-tuned overlay")
        plt.legend(loc="upper right")
        overlay_path = f"diagnostics/{os.path.basename(folder)}_auto_tuned_overlay.png"
        plt.tight_layout()
        plt.savefig(overlay_path, dpi=150, bbox_inches="tight")
        plt.close()

        print("Displaying Plotly reconstruction views...")
        display_reconstruction_results(ideal_room_mesh, mesh_colors, hough_points, trajectory_df)

        summary.append(
            {
                "data_folder": folder,
                "camera_width": round(camera_bounds["camera_width"], 3),
                "camera_depth": round(camera_bounds["camera_depth"], 3),
                "camera_area": round(camera_bounds["camera_area"], 3),
                "rect_width": round(room_debug["rect_width"], 3),
                "rect_depth": round(room_debug["rect_depth"], 3),
                "rect_area": round(fitted_area, 3),
                "hough_point_count": int(len(hough_points)),
                "mesh_point_count": int(len(ideal_room_mesh)),
                "hough_points": hough_points,
                "trajectory": traj_2d,
                "params": params,
                "mesh_path": mesh_path,
                "overlay_path": overlay_path,
                "colors_path": colors_path,
                "fit_ratio": fit_ratio,
                "room_debug": room_debug,
            }
        )

        print(f"Saved mesh -> {mesh_path}")
        print(f"Saved overlay -> {overlay_path}")

    return pd.DataFrame(summary)


# --- Execution Block ---
compare_folders = [
    "Data/TestData1",
    "Data/TestData2",
    "Data/TestData3",
    "Data/TestData4",
    "Data/TestData5",
]

voxel_size = globals().get("HOUGH_VOXEL_SIZE", 0.04)
VIDEO_FPS = globals().get("VIDEO_FPS", 60.0)

summary_df = auto_tune_and_apply(compare_folders, voxel_size=voxel_size)
summary_df.to_csv("diagnostics/auto_tune_summary.csv", index=False)
print(summary_df)

print("\n" + "=" * 85)
print(" BATCH RUN COMPLETE: DIMENSIONS & AREA SUMMARY")
print("=" * 85)

if summary_df.empty:
    print("No successful reconstructions to summarize.")
else:
    display_cols = [
        "data_folder",
        "camera_width",
        "camera_depth",
        "camera_area",
        "rect_width",
        "rect_depth",
        "rect_area",
        "hough_point_count",
        "mesh_point_count",
    ]

    summary_table = summary_df[display_cols].copy()

    summary_table = summary_table.rename(
        columns={
            "data_folder": "Dataset",
            "camera_width": "Cam Traj Width (m)",
            "camera_depth": "Cam Traj Depth (m)",
            "camera_area": "Cam Traj Area (m²)",
            "rect_width": "Acoustic Box Width (m)",
            "rect_depth": "Acoustic Box Depth (m)",
            "rect_area": "Acoustic Box Area (m²)",
            "hough_point_count": "Hough Points",
            "mesh_point_count": "Mesh Points",
        }
    )

    display(HTML(summary_table.to_html(index=False)))

# %% [markdown]
# # 10. Estimate Room Size
# 
# Aggregate all datasets into one unified point cloud by aligning their coordinate planes, then use this to estimate the room size.

# %%
def align_icp_2d(src_points, dst_points, max_iterations=20, tolerance=0.001):
    src = np.copy(src_points[:, [0, 2]])
    dst = np.copy(dst_points[:, [0, 2]])
    
    src_center = np.mean(src, axis=0)
    dst_center = np.mean(dst, axis=0)
    src -= src_center
    dst_centered = dst - dst_center
    
    tree = cKDTree(dst_centered)
    prev_error = float('inf')
    
    for _ in range(max_iterations):
        distances, indices = tree.query(src)
        matched_dst = dst_centered[indices]
        
        src_3d = np.column_stack((src, np.zeros(len(src))))
        dst_3d = np.column_stack((matched_dst, np.zeros(len(matched_dst))))
        
        rot, _ = Rotation.align_vectors(dst_3d, src_3d)
        
        src = rot.apply(src_3d)[:, [0, 1]]
        
        mean_error = np.mean(distances)
        if abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
        
    aligned_src_xz = src + dst_center
    aligned_3d = np.copy(src_points)
    aligned_3d[:, 0] = aligned_src_xz[:, 0]
    aligned_3d[:, 2] = aligned_src_xz[:, 1]
    
    return aligned_3d

# =========================================================
# THE GLOBAL FUSION PIPELINE
# =========================================================

# Optional: set this to a dataset path string to force a master, else leave None
MASTER_DATA_FOLDER = "Data/TestData1"

# 1) Load summary (prefer in-memory summary_df produced by auto_tune_and_apply)
if 'summary_df' in globals() and isinstance(summary_df, pd.DataFrame):
    df = summary_df.copy()
else:
    import os
    csv_path = "diagnostics/auto_tune_summary.csv"
    if not os.path.exists(csv_path):
        raise RuntimeError(f"No summary_df in memory and {csv_path} not found. Run auto_tune_and_apply first.")
    df = pd.read_csv(csv_path)

# Helper to parse stored list/array text
def _ensure_array(x, dtype=np.float32, expected_cols=3):
    if isinstance(x, np.ndarray):
        return x.astype(dtype)
    if isinstance(x, (list, tuple)):
        try:
            arr = np.asarray(x, dtype=dtype)
            if arr.ndim == 1 and expected_cols > 1:
                arr = arr.reshape((-1, expected_cols))
            return arr
        except Exception:
            return np.empty((0, expected_cols), dtype=dtype)
    if isinstance(x, str):
        try:
            parsed = ast.literal_eval(x)
            return _ensure_array(parsed, dtype=dtype, expected_cols=expected_cols)
        except Exception:
            return np.empty((0, expected_cols), dtype=dtype)
    return np.empty((0, expected_cols), dtype=dtype)

# 2) Build plotly_results list
plotly_results = []
for _, row in df.iterrows():
    hp = _ensure_array(row.get('hough_points', []), expected_cols=3)
    traj = _ensure_array(row.get('trajectory', []), expected_cols=2)
    plotly_results.append({
        'data_folder': row.get('data_folder', None),
        'hough_points': hp,
        'trajectory': traj
    })

# 3) Choose master (explicit or first non-empty)
nonempty = [p for p in plotly_results if p['hough_points'].shape[0] > 0]
if len(nonempty) == 0:
    raise RuntimeError("No non-empty hough clouds found in summary; run auto_tune_and_apply first.")

if MASTER_DATA_FOLDER:
    candidates = [p for p in nonempty if p['data_folder'] == MASTER_DATA_FOLDER]
    if len(candidates) == 0:
        print(f"Requested master {MASTER_DATA_FOLDER} not found or empty; falling back to first non-empty.")
        master_entry = nonempty[0]
    else:
        master_entry = candidates[0]
else:
    master_entry = nonempty[0]

print(f"Selected master dataset: {master_entry['data_folder']} (hough points: {len(master_entry['hough_points'])})")

# 4) Align all other clouds to master
master_points = master_entry['hough_points'].astype(np.float32)
fused_list = [master_points.copy()]

for entry in plotly_results:
    if entry['hough_points'].shape[0] == 0:
        continue
    if entry['data_folder'] == master_entry['data_folder']:
        continue
    src = entry['hough_points'].astype(np.float32)
    try:
        aligned = align_icp_2d(src, master_points)  # uses notebook function
        fused_list.append(aligned)
        print(f"Aligned {entry['data_folder']} -> master (src {len(src)} -> aligned {len(aligned)})")
    except Exception as e:
        print(f"Alignment failed for {entry['data_folder']}: {e}")

# 5) Concatenate and clean
master_cloud = np.vstack(fused_list) if fused_list else np.empty((0,3), dtype=np.float32)
print(f"\nFusion complete — combined points: {len(master_cloud)}")

if master_cloud.shape[0] == 0:
    raise RuntimeError("No fused points after alignment.")

db = DBSCAN(eps=0.3, min_samples=15).fit(master_cloud)
mask = db.labels_ != -1
master_clean = master_cloud[mask]
print(f"After DBSCAN cleanup -> {len(master_clean)} points (removed {np.sum(~mask)})")

# Robust room size estimate from fused cloud, emphasizing point agreement
room_pts = master_clean[:, [0, 2]].astype(np.float32)

if room_pts.shape[0] < 3:
    raise RuntimeError("Not enough points to estimate room size.")

# 1) Get a stable room orientation from the full cloud
rect = cv2.minAreaRect(room_pts)
(_, _), (w, h), angle = rect
if w < h:
    angle += 90.0

theta = np.radians(angle)
R_world_to_local = np.array([
    [ np.cos(theta), np.sin(theta)],
    [-np.sin(theta), np.cos(theta)],
], dtype=np.float32)

# 2) Rotate into room coordinates
local_pts = (R_world_to_local @ room_pts.T).T

# 3) Build a consensus mask using bin agreement
grid_size = 0.08  # adjust: smaller = stricter consensus, larger = looser
local_min = local_pts.min(axis=0)
bin_idx = np.floor((local_pts - local_min) / grid_size).astype(np.int32)

# Count how many points support each cell
bins, counts = np.unique(bin_idx, axis=0, return_counts=True)
count_map = {tuple(b): c for b, c in zip(bins, counts)}
support = np.array([count_map[tuple(b)] for b in bin_idx], dtype=np.int32)

# Keep only points in cells with strong agreement
max_support = int(support.max())
support_threshold = max(3, int(np.ceil(0.35 * max_support)))
consensus_mask = support >= support_threshold
consensus_pts = local_pts[consensus_mask]

# Fallback if threshold is too strict
if consensus_pts.shape[0] < 10:
    cutoff = np.percentile(support, 80)
    consensus_pts = local_pts[support >= cutoff]

if consensus_pts.shape[0] < 3:
    consensus_pts = local_pts

# 4) Trim only the high-consensus region
low = np.percentile(consensus_pts, 5.0, axis=0)
high = np.percentile(consensus_pts, 95.0, axis=0)

estimated_width = float(high[0] - low[0])
estimated_depth = float(high[1] - low[1])
estimated_area = float(estimated_width * estimated_depth)

print("=" * 50)
print(" CONSENSUS-BASED ROOM ESTIMATE")
print("=" * 50)
print(f"Estimated Width : {estimated_width:.3f} m")
print(f"Estimated Depth : {estimated_depth:.3f} m")
print(f"Estimated Area  : {estimated_area:.3f} m²")
print(f"Orientation     : {angle:.2f} deg")
print(f"Consensus Points: {len(consensus_pts)} / {len(room_pts)}")
print("=" * 50)

if master_clean.shape[0] == 0:
    raise RuntimeError("All fused points removed by DBSCAN. Consider lowering eps or min_samples.")

# 6) Texture projection using master dataset camera/video
master_folder = master_entry['data_folder']
mov_file, wav_file, py_file, video_clap_frame = find_dataset_files(master_folder)
VIDEO_FPS_LOCAL = globals().get("VIDEO_FPS", 60.0)
master_traj_df = parse_blender_tracking_log(py_file, fps=VIDEO_FPS_LOCAL)
master_traj_df['timestamp'] = master_traj_df['timestamp'] - (video_clap_frame / VIDEO_FPS_LOCAL)
master_K, _, _ = parse_blender_intrinsics(py_file)

print(f"Projecting textures from master video ({master_folder}) onto {len(master_clean)} points...")
try:
    fused_mesh_colors = project_textures_to_mesh(master_clean, mov_file, master_traj_df, master_K, video_clap_frame)
    fused_mesh_colors = np.array(fused_mesh_colors)
except Exception as e:
    fused_mesh_colors = None
    print(f"Texture projection failed: {e}")

# 7) Display final result
print("Displaying Final Fused 3D Reconstruction...")
display_reconstruction_results(ideal_room_mesh=master_clean, mesh_colors=fused_mesh_colors, hough_points=master_clean, trajectory_df=master_traj_df)

# 8) Save fused cloud and colors for later use
np.save("diagnostics/fused_master_cloud.npy", master_clean)
if fused_mesh_colors is not None:
    np.save("diagnostics/fused_master_colors.npy", fused_mesh_colors)
print("Saved diagnostics/fused_master_cloud.npy (and colors if available).")


