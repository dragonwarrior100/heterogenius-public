import streamlit as st
import numpy as np
import tempfile
import os
import io
import zipfile
import py3Dmol
from stmol import showmol as st_3dmol_show
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

try:
    from mp_api.client import MPRester
except ImportError:
    MPRester = None

# Set up Streamlit page
st.set_page_config(page_title="HeteroGenius — Public Edition", layout="wide")

def apply_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }

        /* Gradient Background */
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            color: #f8fafc;
        }

        /* Sidebar Glassmorphism */
        [data-testid="stSidebar"] {
            background: rgba(15, 23, 42, 0.5) !important;
            backdrop-filter: blur(12px) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        /* Vibrant Text Gradients for Headers */
        h1, h2, h3 {
            background: linear-gradient(45deg, #00f2fe, #4facfe, #00f2fe);
            background-size: 200% auto;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: shine 5s linear infinite;
        }

        @keyframes shine {
            to {
                background-position: 200% center;
            }
        }

        /* Beautiful Metric Cards */
        [data-testid="metric-container"] {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(4px);
            transition: all 0.3s ease;
        }

        [data-testid="metric-container"]:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 40px 0 rgba(0, 242, 254, 0.15);
            border-color: rgba(0, 242, 254, 0.3);
            background: rgba(255, 255, 255, 0.05);
        }
        
        [data-testid="metric-container"] label {
            color: #94a3b8 !important;
            font-weight: 600;
        }
        
        [data-testid="metric-container"] div[data-testid="stMetricValue"] {
            color: #f8fafc;
            font-weight: 800;
        }

        /* Premium Buttons */
        .stButton > button {
            background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%) !important;
            border: none !important;
            color: #0f172a !important;
            border-radius: 10px !important;
            padding: 0.6rem 1.5rem !important;
            font-weight: 800 !important;
            font-size: 1.1rem !important;
            box-shadow: 0 4px 15px rgba(0, 242, 254, 0.4) !important;
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
            width: 100%;
        }

        .stButton > button:hover {
            transform: translateY(-2px) scale(1.02) !important;
            box-shadow: 0 8px 25px rgba(0, 242, 254, 0.6) !important;
            filter: brightness(1.1);
        }

        .stButton > button:active {
            transform: translateY(1px) scale(0.98) !important;
        }

        /* File Upload Area */
        [data-testid="stFileUploadDropzone"] {
            background: rgba(255, 255, 255, 0.02) !important;
            border: 2px dashed rgba(255, 255, 255, 0.15) !important;
            border-radius: 12px !important;
            transition: all 0.3s ease !important;
        }

        [data-testid="stFileUploadDropzone"]:hover {
            border-color: #4facfe !important;
            background: rgba(255, 255, 255, 0.05) !important;
        }
        
        /* Sliders */
        .stSlider > div > div > div > div {
            background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%) !important;
        }
        
        /* 3D View Container Polish */
        iframe {
            display: block !important;
            margin: 0 auto !important;
            border-radius: 16px !important;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5) !important;
        }
        
        /* Alerts */
        .stAlert {
            border-radius: 12px !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            backdrop-filter: blur(5px);
        }
        </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# Auto-load Materials Project API key from secrets if available
if 'mp_api_key' not in st.session_state or not st.session_state.mp_api_key:
    try:
        secret_key = st.secrets.get("MP_API_KEY", "")
        if secret_key and secret_key != "PASTE_YOUR_KEY_HERE":
            st.session_state.mp_api_key = secret_key
    except Exception:
        pass

def orient_ab_to_xy(structure):
    """
    Orients a structure so that the a and b lattice vectors lie perfectly
    in the xy-plane. This ensures that the 2D interface is flat and 
    no out-of-plane lattice components are truncated.
    """
    a_vec = structure.lattice.matrix[0]
    b_vec = structure.lattice.matrix[1]
    
    # Normal to the a-b plane
    normal = np.cross(a_vec, b_vec)
    normal = normal / np.linalg.norm(normal)
    
    target_z = np.array([0, 0, 1])
    
    if np.allclose(normal, target_z):
        return structure
        
    axis = np.cross(normal, target_z)
    norm_axis = np.linalg.norm(axis)
    
    if norm_axis > 1e-8:
        axis = axis / norm_axis
        angle = np.arccos(np.clip(np.dot(normal, target_z), -1.0, 1.0))
        
        # Rodrigues' rotation formula
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        
        new_matrix = structure.lattice.matrix @ R.T
        new_coords = structure.cart_coords @ R.T
        
        return Structure(Lattice(new_matrix), structure.species, new_coords, coords_are_cartesian=True)
    elif np.allclose(normal, -target_z):
        # 180 degree rotation around x-axis
        R = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])
        new_matrix = structure.lattice.matrix @ R.T
        new_coords = structure.cart_coords @ R.T
        return Structure(Lattice(new_matrix), structure.species, new_coords, coords_are_cartesian=True)
        
    return structure

def extract_single_layer(structure, gap_threshold=2.5):
    """
    For bulk layered materials (e.g. 2H-WS2 with 2 formula units per primitive
    cell), detect the largest internal vdW gap in fractional z and keep only
    the first layer. Returns the structure unchanged if no gap is found.
    """
    # Work in absolute Cartesian z (c is always [0,0,c_len] after orient_along_c)
    c_len = structure.lattice.matrix[2, 2]
    z_carts = sorted(site.coords[2] for site in structure)
    if len(z_carts) < 2:
        return structure

    # Find the biggest gap between consecutive distinct z-planes
    gaps = []
    for i in range(len(z_carts) - 1):
        g = z_carts[i+1] - z_carts[i]
        gaps.append((g, z_carts[i], z_carts[i+1]))
    max_gap, gap_lo, gap_hi = max(gaps, key=lambda x: x[0])

    if max_gap < gap_threshold:
        return structure  # No clear vdW gap — already a monolayer / 3D bulk

    # Keep atoms in the first slab (z <= gap_lo + small tolerance)
    kept = [(site.species, site.coords) for site in structure
            if site.coords[2] <= gap_lo + 0.05]

    if not kept:
        return structure  # Safety

    kept_species = [k[0] for k in kept]
    kept_coords  = [k[1] for k in kept]

    # Normalise so the slab starts at z=0
    z_min = min(c[2] for c in kept_coords)
    shifted = [np.array([c[0], c[1], c[2] - z_min]) for c in kept_coords]

    # New c = slab_thickness + the vdW gap (preserves original interlayer distance)
    slab_h = max(c[2] for c in shifted)
    new_c  = slab_h + max_gap
    new_mat = structure.lattice.matrix.copy()
    new_mat[2] = [0, 0, new_c]

    return Structure(Lattice(new_mat), kept_species, shifted,
                     coords_are_cartesian=True)

def standardize_structure(structure, z_shift_cart=1.0):
    """
    Shifts the structure so the lowest atom is offset from z=0 by a fixed Cartesian amount
    to avoid cell boundary issues, and standardizes the lattice so 'a' is along x and 'b' is in the xy plane.
    """
    frac = structure.frac_coords.copy()
    min_z = np.min(frac[:, 2])
    c_len = structure.lattice.c
    z_shift_frac = z_shift_cart / c_len if c_len > 0 else 0
    
    frac[:, 2] = (frac[:, 2] - min_z + z_shift_frac) % 1.0
    
    l = structure.lattice
    std_lattice = Lattice.from_parameters(l.a, l.b, l.c, l.alpha, l.beta, l.gamma)
    
    # Clean up numerical noise (e.g., 1e-16) to exactly 0.0
    clean_matrix = np.where(np.abs(std_lattice.matrix) < 1e-10, 0.0, std_lattice.matrix)
    clean_lattice = Lattice(clean_matrix)
    
    return Structure(clean_lattice, structure.species, frac)

def get_structure_from_upload(uploaded_file):
    """Loads a pymatgen Structure from a Streamlit uploaded file."""
    if uploaded_file is None:
        return None
    
    suffix = ".cif" if uploaded_file.name.lower().endswith(".cif") else ".poscar"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
        
    try:
        struct = Structure.from_file(tmp_path)
    finally:
        os.remove(tmp_path)
    return struct

def find_diagonal_matches(sub_struct, film_struct, max_area_zsl, max_strain_pct, min_atoms, max_atoms, substrate_repeats):
    strain_tol = max_strain_pct / 100.0
    
    sub_matrix = sub_struct.lattice.matrix[:2, :2]
    film_matrix = film_struct.lattice.matrix[:2, :2]
    
    sub_area = np.abs(np.linalg.det(sub_matrix))
    film_area = np.abs(np.linalg.det(film_matrix))
    
    matches = []
    max_n = max(2, int(np.ceil(max_area_zsl / sub_area)) + 1)
    max_m = max(2, int(np.ceil(max_area_zsl / film_area)) + 1)
    
    for nx in range(1, max_n):
        for ny in range(1, max_n):
            if nx * ny * sub_area > max_area_zsl:
                continue
            for mx in range(1, max_m):
                for my in range(1, max_m):
                    if mx * my * film_area > max_area_zsl:
                        continue
                    
                    t_sub = np.array([[nx, 0], [0, ny]])
                    t_film = np.array([[mx, 0], [0, my]])
                    
                    s_sl = t_sub @ sub_matrix
                    f_sl = t_film @ film_matrix
                    
                    matches.append({
                        "sub_sl_transform": t_sub,
                        "film_sl_transform": t_film,
                        "substrate_sl_vectors": s_sl,
                        "film_sl_vectors": f_sl
                    })
    
    valid_matches = []
    rejected_matches = []
    
    for m in matches:
        try:
            t_film = m.get("film_sl_transform")
            t_sub = m.get("sub_sl_transform")
            f_sl = m.get("film_sl_vectors")
            s_sl = m.get("substrate_sl_vectors")
        except Exception:
            continue
            
        if t_sub is None or t_film is None or f_sl is None or s_sl is None:
            continue

        S = np.array(s_sl).T
        F = np.array(f_sl).T
        
        try:
            F_def = S @ np.linalg.inv(F)
            if np.linalg.det(F_def) < 0:
                continue

            E = 0.5 * (F_def + F_def.T) - np.eye(2)
            e_xx = E[0, 0]
            e_yy = E[1, 1]
            e_xy = E[0, 1]
            
            strain = float(np.sqrt(e_xx**2 - e_xx*e_yy + e_yy**2 + 3*e_xy**2))
        except Exception:
            continue

        T_sub_3x3 = np.eye(3)
        T_sub_3x3[:2, :2] = t_sub
        T_sub_3x3[2, 2] = substrate_repeats
        
        T_film_3x3 = np.eye(3)
        T_film_3x3[:2, :2] = t_film

        s_sl_3x3_eff = sub_struct.lattice.matrix.copy()
        s_sl_3x3_eff[:2, :2] = s_sl
        s_sl_3x3_eff[2] *= substrate_repeats
        t_sub_eff = np.rint(s_sl_3x3_eff @ np.linalg.inv(sub_struct.lattice.matrix)).astype(int)

        f_sl_3x3_eff = film_struct.lattice.matrix.copy()
        f_sl_3x3_eff[:2, :2] = f_sl
        t_film_eff = np.rint(f_sl_3x3_eff @ np.linalg.inv(film_struct.lattice.matrix)).astype(int)
        
        n_sub = len(sub_struct) * int(round(np.abs(np.linalg.det(T_sub_3x3))))
        n_film = len(film_struct) * int(round(np.abs(np.linalg.det(T_film_3x3))))
        total_atoms = n_sub + n_film
        
        match_info = {
            "t_sub": T_sub_3x3,
            "t_film": T_film_3x3,
            "t_sub_eff": t_sub_eff,
            "t_film_eff": t_film_eff,
            "strain": strain,
            "total_atoms": total_atoms,
            "n_sub": n_sub,
            "n_film": n_film,
            "f_sl": f_sl,
            "s_sl": s_sl
        }
        
        is_valid = True
        if strain is not None and strain > strain_tol:
            is_valid = False
        if total_atoms > max_atoms or total_atoms < min_atoms:
            is_valid = False
            
        if is_valid:
            valid_matches.append(match_info)
        else:
            rejected_matches.append(match_info)
            
    return valid_matches, rejected_matches

# ----------------- UI Sidebar -----------------
with st.sidebar:
    st.header("Upload Structures")
    sub_file = st.file_uploader("Substrate (Bottom) [.cif, POSCAR]", type=["cif", "poscar"])
    film_file = st.file_uploader("2D Layer (Top) [.cif, POSCAR]", type=["cif", "poscar"])
    
    st.header("Interface Parameters")
    vdw_gap = st.number_input("van der Waals Gap (Å)", value=3.3, step=0.1)
    top_vacuum = st.number_input("Top Vacuum Gap (Å)", value=15.0, step=1.0)
    substrate_repeats = st.slider("Substrate Thickness (Z Repeats)", min_value=1, max_value=10, value=1)
    
    termination_plane = None
    if sub_file is not None:
        try:
            # Use the fallback structure if one was found for this same file
            use_cached_sub = (
                st.session_state.get('fallback_used') and
                'active_sub_struct' in st.session_state and
                sub_file.name == st.session_state.get('sub_file_name', '')
            )
            if use_cached_sub:
                tmp_sub = st.session_state['active_sub_struct']
            else:
                tmp_sub = get_structure_from_upload(sub_file)
                tmp_sub = SpacegroupAnalyzer(tmp_sub).get_primitive_standard_structure()
                tmp_sub = orient_ab_to_xy(tmp_sub)
                tmp_sub = extract_single_layer(tmp_sub)
            
            c_len = tmp_sub.lattice.matrix[2, 2]
            z_coords = [site.coords[2] for site in tmp_sub]
            species = [site.species_string for site in tmp_sub]
            
            base_planes = []
            for z, sp in zip(z_coords, species):
                found = False
                for bp in base_planes:
                    if abs(bp['z'] - z) < 0.1:
                        bp['species'].add(sp)
                        # Average the z slightly
                        bp['z'] = (bp['z'] + z) / 2.0
                        found = True
                        break
                if not found:
                    base_planes.append({'z': z, 'species': {sp}})
                    
            base_planes.sort(key=lambda x: x['z'])
            
            all_planes = []
            for i in range(substrate_repeats):
                for bp in base_planes:
                    all_planes.append({
                        'z': bp['z'] + i * c_len,
                        'species': bp['species'],
                        'layer_idx': i + 1
                    })
            
            plane_options = []
            for idx, p in enumerate(all_planes):
                sp_str = ", ".join(sorted(list(p['species'])))
                label = f"Plane {idx+1}: Z = {p['z']:.2f} Å ({sp_str}) [Repeat {p['layer_idx']}]"
                plane_options.append((p['z'], label))
            
            if plane_options:
                selected_option = st.selectbox(
                    "Substrate Termination Plane",
                    options=plane_options,
                    format_func=lambda x: x[1],
                    index=len(plane_options)-1 # Default to highest plane
                )
                if selected_option:
                    termination_plane = selected_option[0]
        except Exception as e:
            st.warning(f"Could not parse substrate planes dynamically: {e}")

    max_area_zsl = st.slider("Maximum Interface Area (Å²)", min_value=50, max_value=800, value=250)
    min_atoms, max_atoms = st.slider("Allowed Atoms Range", min_value=10, max_value=1500, value=(50, 685))
    max_strain_pct = st.slider("Maximum Allowed Strain (%)", min_value=0.1, max_value=10.0, value=5.0, step=0.1)
    
    process_btn = st.button("Generate Heterostructure", type="primary")

# ----------------- Main Layout -----------------
st.markdown("<h1 style='text-align: center; font-size: 3.5rem; margin-bottom: 0;'>✨ HeteroGenius (Diagonal Edition) ✨</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #94a3b8; font-weight: 300; margin-top: -10px; margin-bottom: 30px;'>Strict Positive Integer Diagonal Expansions</h3>", unsafe_allow_html=True)

st.markdown("""
<div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); padding: 20px; border-radius: 12px; margin-bottom: 30px;">
    Automate the creation of 2D/3D coherent heterointerfaces using purely diagonal transformation matrices. 
    Simply upload your crystal structures, and the engine will discover the optimal diagonal supercell match with minimal strain.
</div>
""", unsafe_allow_html=True)

if process_btn:
    st.session_state['generation_successful'] = False
    st.session_state['generation_failed'] = False
    st.session_state['fallback_attempted'] = False
    st.session_state['rejected_matches'] = []
    
    if not sub_file or not film_file:
        st.error("Please upload both the Substrate and the 2D Layer files.")
    else:
        with st.spinner("Processing structures and searching for diagonal matching lattices..."):
            try:
                # 1. Load and orient structures
                # Reuse cached fallback substrate if available for the same file
                use_cached_sub = (
                    st.session_state.get('fallback_used') and
                    'active_sub_struct' in st.session_state and
                    sub_file.name == st.session_state.get('sub_file_name', '')
                )
                if use_cached_sub:
                    sub_struct = st.session_state['active_sub_struct']
                else:
                    sub_struct = get_structure_from_upload(sub_file)
                    sub_struct = SpacegroupAnalyzer(sub_struct).get_primitive_standard_structure()
                    sub_struct = orient_ab_to_xy(sub_struct)
                    sub_struct = extract_single_layer(sub_struct)

                film_struct = get_structure_from_upload(film_file)
                film_struct = SpacegroupAnalyzer(film_struct).get_primitive_standard_structure()
                film_struct = orient_ab_to_xy(film_struct)
                film_struct = extract_single_layer(film_struct)
                

                # 3. Find Diagonal Matches (Offshoot behavior)
                valid_matches, rejected_matches = find_diagonal_matches(
                    sub_struct, film_struct, max_area_zsl, max_strain_pct, min_atoms, max_atoms, substrate_repeats
                )
                
                # Preserve fallback state if we're reusing a cached MP structure
                if use_cached_sub:
                    fallback_used = True
                    fallback_mpid = st.session_state.get('fallback_mpid')
                    fallback_stable = st.session_state.get('fallback_stable', True)
                else:
                    fallback_used = False
                    fallback_mpid = None
                    fallback_stable = True
                
                if not valid_matches:
                    # 4. Fallback to Materials Project
                    has_key = 'mp_api_key' in st.session_state and bool(st.session_state.get('mp_api_key'))
                    has_mprester = MPRester is not None
                    stable_only = st.session_state.get('mp_stable_only', True)
                    if has_key and has_mprester:
                        st.session_state['fallback_attempted'] = True
                        formula = sub_struct.composition.reduced_formula
                        
                        progress_placeholder = st.empty()
                        progress_placeholder.info(f"🔍 No match found with your uploaded substrate. Querying Materials Project for all **{formula}** structures...")
                        
                        try:
                            with MPRester(st.session_state.mp_api_key) as mpr:
                                # Search for all materials with exact formula — request structure explicitly
                                mp_results = mpr.materials.summary.search(
                                    formula=formula,
                                    fields=["material_id", "formula_pretty", "energy_above_hull", "structure"]
                                )
                                # Filter exact formula match
                                mp_results = [m for m in mp_results if m.formula_pretty == formula]
                                # Sort by energy above hull (stability)
                                mp_results.sort(key=lambda x: getattr(x, 'energy_above_hull', float('inf')) or 0.0)
                                
                                progress_placeholder.info(f"🔍 Found **{len(mp_results)}** {formula} structures on Materials Project. Testing each one...")
                                
                                all_mp_matches = []
                                mp_progress = st.progress(0, text="Evaluating alternative structures...")
                                
                                for idx, mp_doc in enumerate(mp_results):
                                    mp_id = mp_doc.material_id
                                    e_hull = getattr(mp_doc, 'energy_above_hull', None)
                                    e_hull_str = f"{e_hull:.3f} eV/atom" if e_hull is not None else "N/A"
                                    
                                    # Skip metastable/unstable structures if filter is on
                                    if stable_only and e_hull is not None and e_hull > 0.001:
                                        mp_progress.progress(
                                            (idx + 1) / len(mp_results),
                                            text=f"Skipping {mp_id} ({idx+1}/{len(mp_results)}) — unstable (E_hull: {e_hull_str})"
                                        )
                                        continue
                                    
                                    mp_progress.progress(
                                        (idx + 1) / len(mp_results),
                                        text=f"Testing {mp_id} ({idx+1}/{len(mp_results)}) — E_hull: {e_hull_str}"
                                    )
                                    try:
                                        # Get the structure from the summary doc
                                        alt_struct = mp_doc.structure
                                        if alt_struct is None:
                                            # Fallback: fetch structure individually
                                            alt_struct = mpr.get_structure_by_material_id(mp_id)
                                        
                                        # Standardize it to match our pipeline
                                        alt_struct = SpacegroupAnalyzer(alt_struct).get_primitive_standard_structure()
                                        alt_struct = orient_ab_to_xy(alt_struct)
                                        alt_struct = extract_single_layer(alt_struct)
                                        
                                        alt_valid, _ = find_diagonal_matches(
                                            alt_struct, film_struct, max_area_zsl, max_strain_pct, min_atoms, max_atoms, substrate_repeats
                                        )
                                        
                                        for match in alt_valid:
                                            match['mp_id'] = str(mp_id)
                                            match['alt_struct'] = alt_struct
                                            match['energy_above_hull'] = e_hull if e_hull is not None else 0.0
                                            all_mp_matches.append(match)
                                            
                                    except Exception as e:
                                        continue
                                
                                mp_progress.empty()
                                
                                if all_mp_matches:
                                    # Pick the one with the lowest strain
                                    best_alt = min(all_mp_matches, key=lambda x: x["strain"])
                                    valid_matches = [best_alt]
                                    sub_struct = best_alt['alt_struct']
                                    fallback_used = True
                                    fallback_mpid = best_alt['mp_id']
                                    fallback_stable = best_alt.get('energy_above_hull', 0.0) <= 0.01
                                    progress_placeholder.success(f"✅ Found a matching alternative substrate: **{fallback_mpid}**")
                                else:
                                    progress_placeholder.error(f"Searched all {len(mp_results)} {formula} structures on Materials Project — none matched your constraints.")
                                        
                        except Exception as e:
                            progress_placeholder.error(f"Materials Project fallback search failed: {e}")
                
                if not valid_matches:
                    st.session_state['generation_failed'] = True
                    st.session_state['failed_rejected_matches'] = rejected_matches
                else:
                    # 4. Select the match with the absolute lowest strain
                    best_match = min(valid_matches, key=lambda x: x["strain"])
                        
                    T_sub = best_match["t_sub"].copy()
                    T_film = best_match["t_film"].copy()
                    s_sl = best_match["s_sl"].copy()
                    f_sl = best_match["f_sl"].copy()
                    
                    # 5. Build the Supercells
                    sub_super = sub_struct.copy()
                    sub_super.make_supercell(T_sub)

                    # --- TERMINATION SHIFT LOGIC ---
                    if fallback_used and not use_cached_sub:
                        termination_plane = None

                    if 'termination_plane' in locals() and termination_plane is not None:
                        c_vec = sub_super.lattice.matrix[2]
                        new_coords = []
                        for site in sub_super:
                            c_cart = site.coords.copy()
                            if c_cart[2] > termination_plane + 0.05:
                                c_cart -= c_vec
                            new_coords.append(c_cart)
                        
                        new_coords = np.array(new_coords)
                        z_min = np.min(new_coords[:, 2])
                        new_coords[:, 2] -= z_min
                        
                        sub_super = Structure(
                            sub_super.lattice,
                            sub_super.species,
                            new_coords,
                            coords_are_cartesian=True
                        )
                    # -------------------------------

                    film_super = film_struct.copy()
                    film_super.make_supercell(T_film)


                    # 6. Apply Epitaxial Strain via Pure Cartesian Math
                    # The substrate perfectly assumes the matched superlattice vectors `s_sl`
                    S = np.array(s_sl).T
                    F = np.array(f_sl).T
                    
                    # The 2D tensor mapping film superlattice to substrate superlattice
                    strain_tensor = S @ np.linalg.inv(F)
                    
                    # Strain the film's Cartesian coordinates directly (x, y only)
                    strained_film_coords = []
                    for coord in film_super.cart_coords:
                        xy = coord[:2]
                        new_xy = strain_tensor @ xy
                        strained_film_coords.append([new_xy[0], new_xy[1], coord[2]])
                    
                    strained_film_coords = np.array(strained_film_coords)
                    # Keep a pre-shift copy for the standalone strained POSCAR
                    strained_film_coords_standalone = strained_film_coords.copy()

                    # 7. Stack: translate strained film above substrate by VdW gap
                    sub_max_z  = max(sub_super.cart_coords[:, 2])
                    film_min_z = min(strained_film_coords[:, 2])
                    z_shift    = sub_max_z + vdw_gap - film_min_z

                    strained_film_coords[:, 2] += z_shift
                    
                    # Combine atoms
                    combined_species = ([site.species for site in sub_super] +
                                        [site.species for site in film_super])
                    combined_coords  = np.vstack([sub_super.cart_coords, strained_film_coords])
                    
                    # 8. Build final right-handed lattice box with vacuum
                    combined_max_z = max(combined_coords[:, 2])
                    final_c_length = combined_max_z + top_vacuum
                    
                    final_lattice_matrix = np.eye(3)
                    final_lattice_matrix[:2, :2] = S.T
                    final_lattice_matrix[2, 2] = final_c_length
                    
                    # --- CHIRALITY SAFEGUARD ---
                    a_vec = final_lattice_matrix[0]
                    b_vec = final_lattice_matrix[1]
                    if a_vec[0]*b_vec[1] - a_vec[1]*b_vec[0] < 0:
                        # Swap a and b vectors to enforce a right-handed cell
                        final_lattice_matrix[[0, 1]] = final_lattice_matrix[[1, 0]]
                    # ---------------------------
                    
                    final_lattice   = Lattice(final_lattice_matrix)
                    final_structure = Structure(final_lattice, combined_species,
                                                combined_coords, coords_are_cartesian=True)

                    # Build the strained film structure as a standalone structure
                    # Use the pre-shift (standalone) copy so z starts near 0
                    strained_film_min_z = min(strained_film_coords_standalone[:, 2])

                    # All standalone structures share the same c = final_c_length and
                    # the same xy lattice as the interface (S.T), so all four cells
                    # are fully commensurate for binding energy calculations.
                    shared_lat_mat = final_lattice_matrix.copy()  # already has final_c_length in [2,2]

                    # --- Strained 2D layer ---
                    sf_coords_zeroed = strained_film_coords_standalone.copy()
                    sf_coords_zeroed[:, 2] -= strained_film_min_z
                    strained_film_structure = Structure(
                        Lattice(shared_lat_mat),
                        [site.species for site in film_super],
                        sf_coords_zeroed,
                        coords_are_cartesian=True
                    )

                    # --- Substrate (rebuild with shared lattice + final_c_length) ---
                    sub_lat_mat = shared_lat_mat.copy()
                    sub_coords_zeroed = sub_super.cart_coords.copy()
                    sub_coords_zeroed[:, 2] -= min(sub_coords_zeroed[:, 2])
                    sub_super_export = Structure(
                        Lattice(sub_lat_mat),
                        [site.species for site in sub_super],
                        sub_coords_zeroed,
                        coords_are_cartesian=True
                    )

                    # --- Unstrained 2D layer ---
                    # CRITICAL: use the film's OWN native lattice (a, b from film_super),
                    # not the strained substrate-matched box. Only c is set to final_c_length
                    # so that E_bind = E_hetero - E_sub - E_strained + E_unstrained is
                    # computed with correctly referenced reference energies.
                    film_native_mat = film_super.lattice.matrix.copy()
                    film_native_mat[2] = [0.0, 0.0, final_c_length]  # same vacuum depth
                    film_unstrained_coords = film_super.cart_coords.copy()
                    film_unstrained_coords[:, 2] -= min(film_unstrained_coords[:, 2])
                    film_super_unstrained_export = Structure(
                        Lattice(film_native_mat),
                        [site.species for site in film_super],
                        film_unstrained_coords,
                        coords_are_cartesian=True
                    )

                    final_structure = standardize_structure(final_structure)
                    sub_super_export = standardize_structure(sub_super_export)
                    strained_film_structure = standardize_structure(strained_film_structure)
                    film_super_unstrained_export = standardize_structure(film_super_unstrained_export)

                    # ----------------- Save to Session State -----------------
                    st.session_state['generation_successful'] = True
                    st.session_state['best_match'] = best_match
                    st.session_state['final_structure'] = final_structure
                    st.session_state['sub_super'] = sub_super_export
                    st.session_state['strained_film_structure'] = strained_film_structure
                    st.session_state['film_super_unstrained'] = film_super_unstrained_export
                    st.session_state['sub_file_name'] = sub_file.name
                    st.session_state['film_file_name'] = film_file.name
                    
                    st.session_state['fallback_used'] = fallback_used
                    st.session_state['fallback_mpid'] = fallback_mpid
                    st.session_state['fallback_stable'] = fallback_stable
                    if fallback_used:
                        st.session_state['active_sub_struct'] = sub_struct
                        st.session_state['sub_file_name'] = sub_file.name
                        st.session_state['_fallback_needs_rerun'] = True
                    
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
                st.exception(e)

    # Force a rerun after first fallback so sidebar termination planes update immediately
    if st.session_state.get('_fallback_needs_rerun'):
        st.session_state.pop('_fallback_needs_rerun')
        st.rerun()

# ----------------- Render Output -----------------
if st.session_state.get('generation_failed', False):
    rejected_matches = st.session_state.get('failed_rejected_matches', [])
    fallback_attempted = st.session_state.get('fallback_attempted', False)
    
    if fallback_attempted:
        st.error("Materials Project fallback search completed, but no alternative structures matched your constraints. Try relaxing your constraints.")
    elif not rejected_matches:
        st.error("No coincidence site lattice found within the given parameters. Enter a Materials Project API key below to search for alternative substrate structures!")
    else:
        st.warning("Matches were found, but they all exceeded your constraints (Allowed Atoms Range or Maximum Allowed Strain). Enter a Materials Project API key below to search for alternative substrate structures!")
        
    if not fallback_attempted:
        has_embedded_key = bool(st.session_state.get('mp_api_key'))
        if not has_embedded_key:
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                mp_key_input = st.text_input("Materials Project API Key", type="password", key="mp_api_key_input_main", help="Press Enter to save, then click 'Generate Heterostructure' again.")
                if mp_key_input:
                    st.session_state.mp_api_key = mp_key_input
                    st.success("API Key saved! Click 'Generate Heterostructure' again to search the Materials Project.")
                stable_toggle = st.checkbox("Only search thermodynamically stable structures", value=True, key="mp_stable_only")
            st.markdown("<br>", unsafe_allow_html=True)
        else:
            st.info("🔑 A Materials Project API key is configured. Click 'Generate Heterostructure' again to automatically search for alternative substrate structures.")
            stable_toggle = st.checkbox("Only search thermodynamically stable structures", value=True, key="mp_stable_only")

    if rejected_matches:
        import pandas as pd
        st.write("Here are the closest matches that were rejected:")
        # Sort by lowest strain first
        sorted_matches = sorted(
            rejected_matches,
            key=lambda x: x['strain'] if x['strain'] is not None else float('inf')
        )
        df = pd.DataFrame(sorted_matches)
        df['strain_pct'] = df['strain'].apply(lambda x: f"{x*100:.2f}%" if x is not None else "N/A")
        df['rejected_by'] = df.apply(lambda row: "Strain/Atoms Constraint Exceeded", axis=1)
        st.dataframe(df[['n_sub', 'n_film', 'total_atoms', 'strain_pct', 'rejected_by']])

if st.session_state.get('generation_successful', False):
    best_match = st.session_state['best_match']
    final_structure = st.session_state['final_structure']
    
    if st.session_state.get('fallback_used', False):
        st.success(f"Successfully generated heterostructure using an alternative substrate from the Materials Project: **{st.session_state['fallback_mpid']}**")
        if not st.session_state.get('fallback_stable', True):
            st.warning("⚠️ **Note:** The alternative substrate used is theoretically predicted but has an energy above the convex hull > 0.01 eV/atom, meaning it might be metastable or unstable.")
    else:
        st.success("Successfully generated heterostructure!")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Epitaxial Strain on 2D Layer", f"{best_match['strain']*100:.2f}%")
    with col2:
        n_sub_det = int(round(np.abs(np.linalg.det(best_match['t_sub']))))
        n_film_det = int(round(np.abs(np.linalg.det(best_match['t_film']))))
        st.metric("Supercell Size", f"{n_sub_det}x Sub / {n_film_det}x Film")
    with col3:
        st.metric("Total Atoms", best_match['total_atoms'])

    st.markdown("<h3 style='text-align: center; margin-top: 30px; margin-bottom: 15px;'>Interactive 3D Visualization</h3>", unsafe_allow_html=True)
    
    # Prepare structure for visualization
    vis_structure = final_structure.copy()
    # Wrap coordinates to [0, 1) to ensure they are inside the visualization bounding box
    vis_frac = vis_structure.frac_coords % 1.0
    vis_structure = Structure(vis_structure.lattice, vis_structure.species, vis_frac)
    
    # Use py3Dmol for visualization
    
    # Use py3Dmol for visualization
    cif_str = vis_structure.to(fmt="cif")
    view = py3Dmol.view(width=800, height=450)
    view.addModel(cif_str, "cif")
    view.setStyle({'sphere': {'radius': 0.4}, 'stick': {'radius': 0.15}})
    view.addUnitCell()
    view.setBackgroundColor('#0f172a')
    view.zoomTo()
    
    # Render in Streamlit (centered)
    _, col_3d, _ = st.columns([1, 6, 1])
    with col_3d:
        st_3dmol_show(view, width=800, height=450)
    
    st.subheader("Download Structure")

    def species_label(structure):
        """Return a compact string of element symbols for file naming."""
        seen = []
        for site in structure:
            sym = site.specie.symbol
            if sym not in seen:
                seen.append(sym)
        return "_".join(seen)

    st.download_button(
        label="📥 Download POSCAR",
        data=final_structure.to(fmt="poscar"),
        file_name="POSCAR",
        mime="text/plain",
        type="primary"
    )

    with st.expander("Show Detailed Transformation Matrices"):
        st.write("**Substrate Transformation:**")
        st.write(best_match["t_sub_eff"])
        st.write("**2D Layer Transformation:**")
        st.write(best_match["t_film_eff"])

    # ---- POSCAR-only ZIP Bundle ----
    st.markdown("---")
    st.subheader("📦 Binding Energy Bundle")
    st.caption(
        "Downloads a ZIP with all four POSCARs — heterostructure, substrate, "
        "strained 2D layer, and unstrained 2D layer — ready for binding energy calculations."
    )

    sub_super        = st.session_state.get('sub_super')
    strained_film_st = st.session_state.get('strained_film_structure')
    film_unstrained  = st.session_state.get('film_super_unstrained')

    if sub_super is not None and strained_film_st is not None and film_unstrained is not None:
        structures_map = {
            f"POSCAR_compound_{species_label(final_structure)}": final_structure,
            f"POSCAR_substrate_{species_label(sub_super)}": sub_super,
            f"POSCAR_STRAINED2dlayer_{species_label(strained_film_st)}": strained_film_st,
            f"POSCAR_unstrained2dlayer_{species_label(film_unstrained)}": film_unstrained,
        }

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for poscar_name, struct in structures_map.items():
                zf.writestr(poscar_name, struct.to(fmt="poscar"))
        zip_buffer.seek(0)

        st.download_button(
            label="📦 Download All POSCARs (.zip)",
            data=zip_buffer.getvalue(),
            file_name="binding_energy_bundle.zip",
            mime="application/zip",
            type="primary",
        )
    else:
        st.info("Re-generate the heterostructure to enable the bundle download.")
