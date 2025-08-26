#!/usr/bin/env python3
"""
Script to visualize precipitation from CorrDiff output files.

Usage:
    python plot_precipitation.py 000_000.npy
    python plot_precipitation.py 001_000.npy --output precip_plot.png
    python plot_precipitation.py data.npy --max-precip 50 --title "Custom Title"
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



def plot_precipitation(input_file, output_file=None, max_precip=100, title=None, show_plot=True):
    """
    Plot precipitation from CorrDiff output file.
    
    Parameters:
    input_file: str - path to input .npy file
    output_file: str - path to save output plot (optional)
    max_precip: float - maximum precipitation for colorbar (mm)
    title: str - custom title (optional)
    show_plot: bool - whether to display the plot
    """
    
    # Load coordinate grids
    lat_grid, lon_grid = load_coordinates()
    if lat_grid is None or lon_grid is None:
        return False
    
    
    # Load the data
    try:
        data = np.load(input_file).squeeze()
        print(f"Loaded data from {input_file}: shape {data.shape}")
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_file}")
        return False
    except Exception as e:
        print(f"Error loading data: {e}")
        return False
    

    
    
    # Extract precipitation (index 3 = "tp" = total precipitation)
    precip = data[3, :, :]
    
    print(f"Precipitation range: {precip.min():.3f} to {precip.max():.3f} mm")
    print(f"Mean precipitation: {precip.mean():.3f} mm")
    
    # Create the plot
    fig = plt.figure(figsize=(12, 8))
    
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
    
    # Create NCAR precipitation colormap
    ncar_cmap = create_ncar_precipitation_colormap()
    
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
    
    # Set title
    if title is None:
        title = f'Precipitation - {os.path.basename(input_file)}'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02, extend='max')
    cbar.set_label('Precipitation (mm)', fontsize=14, fontweight='bold')
    
    # Add precipitation statistics as text
    stats_text = f'Min: {precip.min():.1f} mm\nMax: {precip.max():.1f} mm\nMean: {precip.mean():.1f} mm'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            verticalalignment='top', fontsize=10)
    
    plt.tight_layout()
    
    # Save the plot if output file specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Plot saved to: {output_file}")
    
    # Show the plot
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return True

def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(
        description='Visualize precipitation from CorrDiff output files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python plot_precipitation.py 000_000.npy
    python plot_precipitation.py 001_000.npy --output rainfall_plot.png
    python plot_precipitation.py data.npy --max-precip 50 --title "24h Rainfall Forecast"
    python plot_precipitation.py data.npy --no-show --output plot.png
        """
    )
    
    parser.add_argument('input_file', 
                       help='Input .npy file containing CorrDiff output data')
    
    parser.add_argument('--output', '-o', 
                       help='Output file path to save the plot (optional)')
    
    parser.add_argument('--max-precip', '-m', type=float, default=60,
                       help='Maximum precipitation for colorbar in mm (default: 100)')
    
    parser.add_argument('--title', '-t', 
                       help='Custom title for the plot')
    
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display the plot (useful for batch processing)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file does not exist: {args.input_file}")
        sys.exit(1)
    
    # Create the plot
    success = plot_precipitation(
        input_file=args.input_file,
        output_file=args.output,
        max_precip=args.max_precip,
        title=args.title,
        show_plot=not args.no_show
    )
    
    if success:
        print("✅ Precipitation visualization completed successfully!")
    else:
        print("❌ Failed to create precipitation visualization")
        sys.exit(1)

if __name__ == "__main__":
    main()
