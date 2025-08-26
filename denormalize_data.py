import numpy as np

def create_hrrr_statistics():
    """
    Create approximate HRRR normalization statistics based on typical weather data ranges.
    These are estimates - for production use, extract the actual stats from CorrDiff.
    
    Variables: ["u10m", "v10m", "t2m", "tp", "csnow", "cicep", "cfrzr", "crain"]
    """
    
    # Approximate means and standard deviations for HRRR variables
    # Based on typical US weather patterns
    
    hrrr_means = np.array([
        0.0,     # u10m (m/s) - wind u-component, mean ~0
        0.0,     # v10m (m/s) - wind v-component, mean ~0  
        285.0,   # t2m (K) - temperature, mean ~285K (12°C)
        0.0003,  # tp (m) - total precipitation, mean ~0.3mm
        0.0,     # csnow (0-1) - snow fraction, mean ~0
        0.0,     # cicep (0-1) - ice pellet fraction, mean ~0
        0.0,     # cfrzr (0-1) - freezing rain fraction, mean ~0
        0.0      # crain (0-1) - rain fraction, mean ~0
    ])
    
    hrrr_stds = np.array([
        5.0,     # u10m (m/s) - wind standard deviation
        5.0,     # v10m (m/s) - wind standard deviation
        15.0,    # t2m (K) - temperature standard deviation  
        0.002,   # tp (m) - precipitation standard deviation (~2mm)
        0.1,     # csnow - snow fraction std
        0.05,    # cicep - ice pellet fraction std
        0.05,    # cfrzr - freezing rain fraction std
        0.1      # crain - rain fraction std
    ])
    
    return hrrr_means, hrrr_stds

def denormalize_corrdiff_output(normalized_data, hrrr_means=None, hrrr_stds=None):
    """
    Denormalize CorrDiff output data back to physical units.
    
    Parameters:
    normalized_data: numpy array of shape (1, 1, 8, 1056, 1792)
    hrrr_means: array of means for each variable (if None, use estimates)
    hrrr_stds: array of standard deviations for each variable (if None, use estimates)
    
    Returns:
    denormalized_data: array in physical units
    """
    
    if hrrr_means is None or hrrr_stds is None:
        print("Using estimated HRRR statistics (not exact - extract actual stats for production)")
        hrrr_means, hrrr_stds = create_hrrr_statistics()
    
    # Copy the input data
    denormalized_data = normalized_data.copy()
    
    # Apply denormalization: output = std * normalized + mean
    for var_idx in range(8):
        denormalized_data[0, 0, var_idx, :, :] = (
            hrrr_stds[var_idx] * normalized_data[0, 0, var_idx, :, :] + hrrr_means[var_idx]
        )
    
    return denormalized_data

def load_actual_hrrr_stats(stats_dir="."):
    """
    Load actual HRRR statistics if they were extracted from CorrDiff.
    
    Parameters:
    stats_dir: directory containing the stats files
    
    Returns:
    (hrrr_means, hrrr_stds) or (None, None) if files not found
    """
    
    means_file = f"{stats_dir}/corrdiff_us_hrrr_means.npy"
    stds_file = f"{stats_dir}/corrdiff_us_hrrr_stds.npy"
    
    try:
        hrrr_means = np.load(means_file)
        hrrr_stds = np.load(stds_file)
        
        print(f"✓ Loaded actual HRRR statistics:")
        print(f"  Means: {hrrr_means}")
        print(f"  Stds: {hrrr_stds}")
        
        return hrrr_means, hrrr_stds
        
    except FileNotFoundError:
        print(f"⚠️  HRRR statistics files not found in {stats_dir}")
        print(f"   Looking for: {means_file}, {stds_file}")
        print(f"   Using estimated statistics instead.")
        return None, None

def check_data_ranges(data, variable_names=None):
    """
    Check if data ranges look reasonable after denormalization.
    """
    
    if variable_names is None:
        variable_names = ["u10m", "v10m", "t2m", "tp", "csnow", "cicep", "cfrzr", "crain"]
    
    print("\n=== Data Range Check ===")
    
    for var_idx, var_name in enumerate(variable_names):
        var_data = data[0, 0, var_idx, :, :]
        print(f"{var_name}: min={var_data.min():.6f}, max={var_data.max():.6f}, mean={var_data.mean():.6f}")
        
        # Check if ranges are reasonable
        if var_name == "t2m":
            if 200 < var_data.mean() < 320:
                print(f"  ✓ Temperature range looks reasonable (Kelvin)")
            else:
                print(f"  ⚠️  Temperature range might be incorrect")
                
        elif var_name == "tp":
            if 0 <= var_data.min() and var_data.max() < 0.1:  # < 100mm seems reasonable
                print(f"  ✓ Precipitation range looks reasonable (meters)")
            else:
                print(f"  ⚠️  Precipitation range might be incorrect")

# Example usage
if __name__ == "__main__":
    # Load your data
    print("Loading CorrDiff outputs...")
    arr1 = np.load("000_000.npy")
    arr2 = np.load("001_000.npy")
    
    print(f"Original data ranges:")
    print(f"arr1: min={arr1.min():.6f}, max={arr1.max():.6f}")
    print(f"arr2: min={arr2.min():.6f}, max={arr2.max():.6f}")
    
    # Try to load actual stats, fall back to estimates
    hrrr_means, hrrr_stds = load_actual_hrrr_stats()
    
    # Denormalize the data
    print("\nDenormalizing data...")
    arr1_denorm = denormalize_corrdiff_output(arr1, hrrr_means, hrrr_stds)
    arr2_denorm = denormalize_corrdiff_output(arr2, hrrr_means, hrrr_stds)
    
    # Check the ranges
    print("\nArray 1 after denormalization:")
    check_data_ranges(arr1_denorm)
    
    print("\nArray 2 after denormalization:")
    check_data_ranges(arr2_denorm)
    
    # Save denormalized data
    np.save("000_000_denormalized.npy", arr1_denorm)
    np.save("001_000_denormalized.npy", arr2_denorm)
    
    print(f"\n✅ Denormalized data saved!")
    print(f"   000_000_denormalized.npy: {arr1_denorm.shape}")
    print(f"   001_000_denormalized.npy: {arr2_denorm.shape}")
