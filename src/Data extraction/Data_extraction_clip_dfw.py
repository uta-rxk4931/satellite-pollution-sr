import os
from pathlib import Path
import numpy as np
from pyhdf.SD import SD, SDC


INPUT_DIR = r"path\to\yours\maiac_2023"
OUTPUT_DIR = r"path\to\your\dfw_tifs_2023"
# DFW approximate pixel bounds inside h09v05 tile (1 km grid)


ROW_START, ROW_END = 200, 400   # latitude direction
COL_START, COL_END = 300, 500   # longitude direction


os.makedirs(OUTPUT_DIR, exist_ok=True)

hdf_files = sorted(Path(INPUT_DIR).glob("*.hdf"))
print(f"Found {len(hdf_files)} files")

for hdf_path in hdf_files:
    try:
        hdf = SD(str(hdf_path), SDC.READ)
        
        # Optical_Depth_055
        aod_sds = hdf.select("Optical_Depth_055")
        aod = aod_sds.get()         
        
        # Take the first orbit if 3D
        if aod.ndim == 3:
            aod = aod[0]
        
        # Scale
        aod = aod.astype(np.float32) * 0.001
        aod[(aod < -0.05) | (aod > 5)] = np.nan
        
        # Clip 
        clipped = aod[ROW_START:ROW_END, COL_START:COL_END]
        
        
        out_path = Path(OUTPUT_DIR) / (hdf_path.stem + "_dfw.npy")
        np.save(out_path, clipped)
        
        print(f"Saved: {out_path.name}  shape={clipped.shape}")
        hdf.end()
        
    except Exception as e:
        print(f"Failed {hdf_path.name}: {e}")

print("Done. Files saved as .npy in", OUTPUT_DIR)