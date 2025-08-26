#!/usr/bin/env python3
"""
Script to create animated GIF from CorrDiff precipitation forecast files.

Usage:
    python make_precip_gif.py
    python make_precip_gif.py --output forecast_animation.gif --duration 0.8
    python make_precip_gif.py --pattern "output/*_f*.npy" --max-precip 80
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

def create_frame(file_path, forecast_hour, lat_grid, lon_grid, ncar_cmap, 
                max_precip=100, figsize=(12, 8)):
    """Create a single frame for the animation."""
    
    # Load the data
    try:
        data = np.load(file_path).squeeze()
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None
    
    # Extract precipitation (index 3 = "tp" = total precipitation)
    precip = data[3, :, :]
    
    # Create the plot
    fig = plt.figure(figsize=figsize)
    
    # Set up geographic projection
    proj = ccrs.PlateCarree()
    ax = fig.add_subplot(111, projection=proj)
    
    # Convert longitude to -180 to 180 range for display if needed
    lon_display = np.where(lon_grid > 180, lon_grid - 360, lon_grid)
    
    # Set map extent
    lon_min, lon_max = lon_display.min(), lon_display.max()
    lat_min, lat_max = lat_grid.min(), lat_grid.max()
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='black')
    ax.add_feature(cfeature.BORDERS, linewidth=0.6, color='black')
    ax.add_feature(cfeature.STATES, linewidth=0.4, color='white', alpha=0.7)
    ax.add_feature(cfeature.LAKES, color='lightblue', alpha=0.3)
    ax.add_feature(cfeature.RIVERS, color='lightblue', alpha=0.3)
    
    # Plot precipitation
    im = ax.imshow(precip, 
                   extent=[lon_display.min(), lon_display.max(), 
                          lat_grid.min(), lat_grid.max()],
                   origin='upper', 
                   cmap=ncar_cmap, 
                   vmin=0, 
                   vmax=max_precip,
                   transform=proj,
                   alpha=0.8)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    
    # Set title with forecast hour
    title = f'Precipitation Forecast - Hour {forecast_hour:02d}'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02, extend='max')
    cbar.set_label('Precipitation (mm)', fontsize=14, fontweight='bold')
    
    # Add precipitation statistics as text
    stats_text = f'F{forecast_hour:02d}h\nMax: {precip.max():.1f} mm\nMean: {precip.mean():.1f} mm'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
            verticalalignment='top', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    return fig

def create_gif(file_info, output_file="precipitation_forecast.gif", duration=1.0, 
               max_precip=100, figsize=(12, 8)):
    """Create animated GIF from forecast files."""
    
    # Load coordinates
    lat_grid, lon_grid = load_coordinates()
    if lat_grid is None or lon_grid is None:
        return False
    
    # Create NCAR colormap
    ncar_cmap = create_ncar_precipitation_colormap()
    
    # Create temporary directory for frames
    with tempfile.TemporaryDirectory() as temp_dir:
        frame_files = []
        
        print(f"Creating {len(file_info)} animation frames...")
        
        # Create each frame
        for i, (hour, file_path) in enumerate(file_info):
            print(f"  Processing frame {i+1}/{len(file_info)}: F{hour:02d}h")
            
            fig = create_frame(file_path, hour, lat_grid, lon_grid, ncar_cmap, 
                             max_precip, figsize)
            
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
        
        print(f"Creating GIF with {len(frame_files)} frames...")
        
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
        
        print(f"✅ GIF saved to: {output_file}")
        print(f"   Frames: {len(images)}")
        print(f"   Duration per frame: {duration:.1f}s")
        print(f"   Total animation time: {len(images) * duration:.1f}s")
        
        return True

def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(
        description='Create animated GIF from CorrDiff precipitation forecast files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python make_precip_gif.py
    python make_precip_gif.py --output my_forecast.gif
    python make_precip_gif.py --pattern "output/*_f*.npy" --duration 0.5
    python make_precip_gif.py --max-precip 80 --duration 1.2
        """
    )
    
    parser.add_argument('--output', '-o', default='precipitation_forecast.gif',
                       help='Output GIF file path (default: precipitation_forecast.gif)')
    
    parser.add_argument('--pattern', '-p', default='output/*.npy',
                       help='File pattern to search for forecast files (default: output/*.npy)')
    
    parser.add_argument('--duration', '-d', type=float, default=1.0,
                       help='Duration of each frame in seconds (default: 1.0)')
    
    parser.add_argument('--max-precip', '-m', type=float, default=60,
                       help='Maximum precipitation for colorbar in mm (default: 60)')
    
    parser.add_argument('--figsize', nargs=2, type=float, default=[12, 8],
                       help='Figure size as width height (default: 12 8)')
    
    args = parser.parse_args()
    
    print("🎬 CorrDiff Precipitation Animation Creator")
    print("=" * 50)
    
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
    
    # Create the GIF
    success = create_gif(
        file_info=file_info,
        output_file=args.output,
        duration=args.duration,
        max_precip=args.max_precip,
        figsize=tuple(args.figsize)
    )
    
    if success:
        print(f"\n🎉 Animation created successfully!")
        print(f"📁 File: {args.output}")
        
        # Show file size
        if os.path.exists(args.output):
            size_mb = os.path.getsize(args.output) / (1024 * 1024)
            print(f"📊 Size: {size_mb:.1f} MB")
    else:
        print(f"\n❌ Failed to create animation")
        sys.exit(1)

if __name__ == "__main__":
    main()
