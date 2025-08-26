#!/usr/bin/env python3
"""
Script to create animated GIF comparing CorrDiff and GEFS precipitation forecasts.

Usage:
    python make_precip_gif_comparison.py
    python make_precip_gif_comparison.py --output comparison_animation.gif --duration 0.8
    python make_precip_gif_comparison.py --pattern "output/*_f*.npy" --max-precip 80
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import os
import sys
import glob
import re
from PIL import Image
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta

# Import earth2studio for GEFS data
try:
    from earth2studio.data import GEFS_FX, GEFS_FX_721x1440
    EARTH2STUDIO_AVAILABLE = True
except ImportError:
    print("Warning: earth2studio not available. GEFS data will not be loaded.")
    EARTH2STUDIO_AVAILABLE = False

def create_ncar_precipitation_colormap():
    """
    Create NCAR-style precipitation colormap.
    Colors progress from white (no precip) through blues, greens, yellows, oranges to reds/purples.
    """
    # NCAR precipitation color scheme
    colors = [
        '#FFFFFF',  # 0 mm - White (no precipitation)
        '#E0E0FF',  # Very light blue
        '#C0C0FF',  # Light blue
        '#A0A0FF',  # Medium light blue
        '#8080FF',  # Blue
        '#6060E0',  # Medium blue
        '#4040C0',  # Dark blue
        '#20A020',  # Green
        '#40C040',  # Light green
        '#60E060',  # Bright green
        '#80FF80',  # Very bright green
        '#A0FF60',  # Yellow-green
        '#C0FF40',  # Light yellow-green
        '#E0FF20',  # Yellow
        '#FFFF00',  # Bright yellow
        '#FFE000',  # Yellow-orange
        '#FFC000',  # Orange
        '#FFA000',  # Dark orange
        '#FF8000',  # Red-orange
        '#FF6000',  # Red
        '#FF4000',  # Bright red
        '#FF2000',  # Dark red
        '#E00000',  # Very dark red
        '#C00040',  # Red-purple
        '#A00080',  # Purple
        '#8000C0',  # Dark purple
    ]
    
    n_bins = len(colors)
    cmap = mcolors.LinearSegmentedColormap.from_list('ncar_precip', colors, N=n_bins)
    return cmap

def load_coordinates():
    """Load latitude and longitude coordinates."""
    try:
        lat_grid = np.load('static/corrdiff_output_lat.npy').squeeze()
        lon_grid = np.load('static/corrdiff_output_lon.npy').squeeze()
        return lat_grid, lon_grid
    except FileNotFoundError as e:
        print(f"Error: Coordinate files not found: {e}")
        print("Make sure static/corrdiff_output_lat.npy and static/corrdiff_output_lon.npy exist")
        return None, None

def setup_gefs_data_source():
    """Initialize GEFS data source."""
    if not EARTH2STUDIO_AVAILABLE:
        return None
    
    try:
        # Try high-resolution GEFS first
        gefs_source = GEFS_FX_721x1440(cache=True, product="gec00")
        print("Using GEFS_FX_721x1440 (higher resolution)")
        return gefs_source
    except:
        try:
            # Fallback to regular GEFS
            gefs_source = GEFS_FX(cache=True)
            print("Using GEFS_FX (standard resolution)")
            return gefs_source
        except Exception as e:
            print(f"Error initializing GEFS data source: {e}")
            return None

def extract_forecast_hour(filename):
    """Extract forecast hour from filename."""
    # Try different patterns for forecast hour extraction
    patterns = [
        r'_f(\d+)',      # pattern like: file_f12.npy
        r'f(\d+)_',      # pattern like: f12_file.npy
        r'(\d+)h',       # pattern like: 12h.npy
        r'hour(\d+)',    # pattern like: hour12.npy
        r'(\d+)\.npy$'   # pattern like: 12.npy (last digits before .npy)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            return int(match.group(1))
    
    # If no pattern matches, try to extract any number in the filename
    numbers = re.findall(r'\d+', filename)
    if numbers:
        return int(numbers[-1])  # Use the last number found
    
    return 0  # Default to 0 if no number found

def find_forecast_files(pattern="output/*.npy"):
    """Find and sort forecast files by forecast hour."""
    files = glob.glob(pattern)
    
    if not files:
        print(f"No files found matching pattern: {pattern}")
        return []
    
    # Extract forecast hours and sort
    file_info = []
    for file in files:
        hour = extract_forecast_hour(os.path.basename(file))
        file_info.append((hour, file))
    
    # Sort by forecast hour
    file_info.sort(key=lambda x: x[0])
    
    print(f"Found {len(file_info)} forecast files:")
    for hour, file in file_info:
        print(f"  F{hour:02d}: {os.path.basename(file)}")
    
    return file_info

def load_gefs_precipitation(forecast_hour, base_time, gefs_source, target_lat_grid, target_lon_grid):
    """Load GEFS precipitation data for comparison."""
    if gefs_source is None:
        print(f"  GEFS data not available for F{forecast_hour:02d}")
        return None
    
    try:
        # Calculate forecast time
        forecast_time = base_time + timedelta(hours=forecast_hour)
        
        # Load GEFS total precipitation data
        print(f"  Loading GEFS data for F{forecast_hour:02d}h...")
        
        # Try different variable names for precipitation
        precip_vars = ["tp"]
        gefs_data = None
        
        for var in precip_vars:
            try:
                gefs_data = gefs_source(base_time, timedelta(hours=forecast_hour), [var])
                print(f"    Successfully loaded GEFS variable: {var}")
                break
            except Exception as e:
                print(f"    Failed to load GEFS variable {var}: {e}")
                continue
        
        if gefs_data is None:
            print(f"    Could not load GEFS precipitation data for any variable")
            return None
        
        # Extract precipitation data
        gefs_precip = gefs_data.values[0, 0, 0, :, :]  # Assuming shape (1, 1, 1, lat, lon)
        
        # Get GEFS coordinates
        gefs_lat = gefs_data.lat.values
        gefs_lon = gefs_data.lon.values
        
        # Interpolate GEFS data to CorrDiff grid
        from scipy.interpolate import RegularGridInterpolator
        
        # Convert GEFS lon to same convention as target grid if needed
        if gefs_lon.max() > 180 and target_lon_grid.max() <= 180:
            gefs_lon = np.where(gefs_lon > 180, gefs_lon - 360, gefs_lon)
        elif gefs_lon.max() <= 180 and target_lon_grid.max() > 180:
            target_lon_display = np.where(target_lon_grid > 180, target_lon_grid - 360, target_lon_grid)
        else:
            target_lon_display = target_lon_grid
        
        # Create interpolator
        interpolator = RegularGridInterpolator(
            (gefs_lat, gefs_lon),
            gefs_precip,
            method='linear',
            bounds_error=False,
            fill_value=0
        )
        
        # Create target coordinate meshgrid
        target_lat_flat = target_lat_grid.flatten()
        target_lon_flat = target_lon_display.flatten()
        target_points = np.stack([target_lat_flat, target_lon_flat], axis=-1)
        
        # Interpolate
        interpolated_precip = interpolator(target_points)
        gefs_precip_regridded = interpolated_precip.reshape(target_lat_grid.shape)
        
        # Convert to mm (assuming GEFS is in m or m/s - adjust as needed)
        # For precipitation rate (m/s), multiply by forecast interval (3600s for 1h)
        if gefs_precip_regridded.max() < 0.1:  # Likely in m or m/s
            gefs_precip_regridded = gefs_precip_regridded * 1000  # Convert to mm
        
        print(f"    GEFS precip range: {gefs_precip_regridded.min():.3f} to {gefs_precip_regridded.max():.3f} mm")
        
        return gefs_precip_regridded
        
    except Exception as e:
        print(f"  Error loading GEFS data for F{forecast_hour:02d}: {e}")
        return None

def create_comparison_frame(file_path, forecast_hour, lat_grid, lon_grid, ncar_cmap, 
                          gefs_source, base_time, max_precip=60, figsize=(12, 12)):
    """Create a single comparison frame for the animation."""
    
    # Load CorrDiff data
    try:
        corrdiff_data = np.load(file_path).squeeze()
        corrdiff_precip = corrdiff_data[3, :, :]  # Extract precipitation
    except Exception as e:
        print(f"Error loading CorrDiff data from {file_path}: {e}")
        return None
    
    # Load GEFS data
    gefs_precip = load_gefs_precipitation(forecast_hour, base_time, gefs_source, lat_grid, lon_grid)
    
    # Create the plot with two subplots
    fig = plt.figure(figsize=figsize)
    
    # Convert longitude to -180 to 180 range for display if needed
    lon_display = np.where(lon_grid > 180, lon_grid - 360, lon_grid)
    
    # Set map extent
    lon_min, lon_max = lon_display.min(), lon_display.max()
    lat_min, lat_max = lat_grid.min(), lat_grid.max()
    
    # Set up geographic projection
    proj = ccrs.PlateCarree()
    
    # Subplot 1: CorrDiff
    ax1 = fig.add_subplot(211, projection=proj)
    ax1.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
    
    # Add map features
    ax1.add_feature(cfeature.COASTLINE, linewidth=0.8, color='black')
    ax1.add_feature(cfeature.BORDERS, linewidth=0.6, color='black')
    ax1.add_feature(cfeature.STATES, linewidth=0.4, color='white', alpha=0.7)
    ax1.add_feature(cfeature.LAKES, color='lightblue', alpha=0.3)
    
    # Plot CorrDiff precipitation
    im1 = ax1.imshow(corrdiff_precip, 
                     extent=[lon_display.min(), lon_display.max(), 
                            lat_grid.min(), lat_grid.max()],
                     origin='upper', 
                     cmap=ncar_cmap, 
                     vmin=0, 
                     vmax=max_precip,
                     transform=proj,
                     alpha=0.8)
    
    # Add gridlines
    gl1 = ax1.gridlines(draw_labels=True, alpha=0.5, linestyle='--')
    gl1.top_labels = False
    gl1.right_labels = False
    gl1.xlabel_style = {'size': 8}
    gl1.ylabel_style = {'size': 8}
    
    ax1.set_title(f'CorrDiff Downscaled - F{forecast_hour:02d}h', fontsize=14, fontweight='bold')
    
    # Add statistics
    stats_text1 = f'Max: {corrdiff_precip.max():.1f} mm\nMean: {corrdiff_precip.mean():.1f} mm'
    ax1.text(0.02, 0.98, stats_text1, transform=ax1.transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
             verticalalignment='top', fontsize=10, fontweight='bold')
    
    # Subplot 2: GEFS
    ax2 = fig.add_subplot(212, projection=proj)
    ax2.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
    
    # Add map features
    ax2.add_feature(cfeature.COASTLINE, linewidth=0.8, color='black')
    ax2.add_feature(cfeature.BORDERS, linewidth=0.6, color='black')
    ax2.add_feature(cfeature.STATES, linewidth=0.4, color='white', alpha=0.7)
    ax2.add_feature(cfeature.LAKES, color='lightblue', alpha=0.3)
    
    if gefs_precip is not None:
        # Plot GEFS precipitation
        im2 = ax2.imshow(gefs_precip, 
                         extent=[lon_display.min(), lon_display.max(), 
                                lat_grid.min(), lat_grid.max()],
                         origin='upper', 
                         cmap=ncar_cmap, 
                         vmin=0, 
                         vmax=max_precip,
                         transform=proj,
                         alpha=0.8)
        
        stats_text2 = f'Max: {gefs_precip.max():.1f} mm\nMean: {gefs_precip.mean():.1f} mm'
    else:
        # Create empty plot if GEFS data not available
        im2 = ax2.imshow(np.zeros_like(corrdiff_precip), 
                         extent=[lon_display.min(), lon_display.max(), 
                                lat_grid.min(), lat_grid.max()],
                         origin='upper', 
                         cmap=ncar_cmap, 
                         vmin=0, 
                         vmax=max_precip,
                         transform=proj,
                         alpha=0.8)
        
        stats_text2 = 'GEFS data\nnot available'
    
    # Add gridlines
    gl2 = ax2.gridlines(draw_labels=True, alpha=0.5, linestyle='--')
    gl2.top_labels = False
    gl2.right_labels = False
    gl2.xlabel_style = {'size': 8}
    gl2.ylabel_style = {'size': 8}
    
    ax2.set_title(f'GEFS Original - F{forecast_hour:02d}h', fontsize=14, fontweight='bold')
    
    # Add statistics
    ax2.text(0.02, 0.98, stats_text2, transform=ax2.transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
             verticalalignment='top', fontsize=10, fontweight='bold')
    
    # Add shared colorbar
    # Create a separate axes for the colorbar on the right side
    cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im1, cax=cbar_ax, extend='max')
    cbar.set_label('Precipitation (mm)', fontsize=12, fontweight='bold')
    
    # Overall title
    
    plt.tight_layout()
    
    return fig

def create_comparison_gif(file_info, output_file="precipitation_comparison.gif", duration=1.0, 
                         max_precip=100, figsize=(16, 8), base_time=None):
    """Create animated GIF comparing CorrDiff and GEFS forecasts."""
    
    # Load coordinates
    lat_grid, lon_grid = load_coordinates()
    if lat_grid is None or lon_grid is None:
        return False
    
    # Setup GEFS data source
    gefs_source = setup_gefs_data_source()
    
    # Use default base time if not provided
    if base_time is None:
        base_time = datetime(2025, 7, 3, 18, 0, 0)  # Adjust as needed
        print(f"Using default base time: {base_time}")
    
    # Create NCAR colormap
    ncar_cmap = create_ncar_precipitation_colormap()
    
    # Create temporary directory for frames
    with tempfile.TemporaryDirectory() as temp_dir:
        frame_files = []
        
        print(f"Creating {len(file_info)} comparison animation frames...")
        
        # Create each frame
        for i, (hour, file_path) in enumerate(file_info):
            print(f"\nProcessing frame {i+1}/{len(file_info)}: F{hour:02d}h")
            
            fig = create_comparison_frame(file_path, hour, lat_grid, lon_grid, ncar_cmap, 
                                        gefs_source, base_time, max_precip, figsize)
            
            if fig is None:
                print(f"    Skipping frame {i+1} due to error")
                continue
            
            # Save frame as PNG
            frame_file = os.path.join(temp_dir, f"frame_{i:03d}.png")
            fig.savefig(frame_file, dpi=150, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close(fig)
            
            frame_files.append(frame_file)
        
        if not frame_files:
            print("Error: No frames were created successfully")
            return False
        
        print(f"\nCreating comparison GIF with {len(frame_files)} frames...")
        
        # Load all frames
        images = []
        for frame_file in frame_files:
            try:
                img = Image.open(frame_file)
                images.append(img)
            except Exception as e:
                print(f"Error loading frame {frame_file}: {e}")
        
        if not images:
            print("Error: No valid frames to create GIF")
            return False
        
        # Create the GIF
        duration_ms = int(duration * 1000)  # Convert to milliseconds
        
        # Save the GIF
        images[0].save(
            output_file,
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,  # Infinite loop
            optimize=True
        )
        
        print(f"✅ Comparison GIF saved to: {output_file}")
        print(f"   Frames: {len(images)}")
        print(f"   Duration per frame: {duration:.1f}s")
        print(f"   Total animation time: {len(images) * duration:.1f}s")
        
        return True

def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(
        description='Create animated GIF comparing CorrDiff and GEFS precipitation forecasts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python make_precip_gif_comparison.py
    python make_precip_gif_comparison.py --output comparison.gif
    python make_precip_gif_comparison.py --pattern "output/*_f*.npy" --duration 0.5
    python make_precip_gif_comparison.py --max-precip 80 --base-time "2023-07-03T18:00:00"
        """
    )
    
    parser.add_argument('--output', '-o', default='precipitation_comparison.gif',
                       help='Output GIF file path (default: precipitation_comparison.gif)')
    
    parser.add_argument('--pattern', '-p', default='output/*.npy',
                       help='File pattern to search for forecast files (default: output/*.npy)')
    
    parser.add_argument('--duration', '-d', type=float, default=1.0,
                       help='Duration of each frame in seconds (default: 1.0)')
    
    parser.add_argument('--max-precip', '-m', type=float, default=60,
                       help='Maximum precipitation for colorbar in mm (default: 60)')
    
    parser.add_argument('--figsize', nargs=2, type=float, default=[16, 8],
                       help='Figure size as width height (default: 16 8)')
    
    parser.add_argument('--base-time', '-t', 
                       help='Base forecast time in ISO format (default: 2023-07-03T18:00:00)')
    
    args = parser.parse_args()
    
    print("🎬 CorrDiff vs GEFS Precipitation Comparison Creator")
    print("=" * 60)
    
    # Parse base time
    base_time = None
    if args.base_time:
        try:
            base_time = datetime.fromisoformat(args.base_time)
            print(f"Using specified base time: {base_time}")
        except ValueError:
            print(f"Error: Invalid base time format: {args.base_time}")
            print("Use ISO format like: 2023-07-03T18:00:00")
            sys.exit(1)
    
    # Find forecast files
    file_info = find_forecast_files(args.pattern)
    
    if not file_info:
        print(f"❌ No forecast files found with pattern: {args.pattern}")
        print("\nTry different patterns:")
        print("  --pattern 'output/*.npy'")
        print("  --pattern '*.npy'") 
        print("  --pattern 'data/forecast_*.npy'")
        sys.exit(1)
    
    if len(file_info) < 2:
        print(f"⚠️  Only {len(file_info)} file found. Need at least 2 files for animation.")
        sys.exit(1)
    
    # Create the comparison GIF
    success = create_comparison_gif(
        file_info=file_info,
        output_file=args.output,
        duration=args.duration,
        max_precip=args.max_precip,
        figsize=tuple(args.figsize),
        base_time=base_time
    )
    
    if success:
        print(f"\n🎉 Comparison animation created successfully!")
        print(f"📁 File: {args.output}")
        
        # Show file size
        if os.path.exists(args.output):
            size_mb = os.path.getsize(args.output) / (1024 * 1024)
            print(f"📊 Size: {size_mb:.1f} MB")
    else:
        print(f"\n❌ Failed to create comparison animation")
        sys.exit(1)

if __name__ == "__main__":
    main()
