from pathlib import Path
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon
from matplotlib.patches import Rectangle, FancyArrow, Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib as mpl
import requests
from shapely.ops import unary_union
from matplotlib.colors import LogNorm, LinearSegmentedColormap

# ----- Configurations ----- #
FONT_FAMILY = "DejaVu Sans"
BASE_FONTSIZE = 9
TITLE_FONTSIZE = 14
COUNTRY_LABEL_FONTSIZE = 13
mpl.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.size": BASE_FONTSIZE,
    "axes.titlesize": TITLE_FONTSIZE,
    "axes.titleweight": "bold"
})

LEFT_ANCHOR = 0.01

# ----- State and county shapefile paths ----- #
SCRIPT_DIR = Path(__file__).parent  # ../national_embeddings/

# shapefiles obtained from: https://www.census.gov/cgi-bin/geo/shapefiles/index.php on Feb. 4, 2026
CENSUS_STATES = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_500k.zip"
CENSUS_COUNTIES = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_500k.zip"
# this is for connecticut, to be able to plot it up until 2023. 
# connecticut used counties for WNV case data up until 2022. 
# and it also used geoids for keeping track of the case data, 
# thus I need an older version to plot it 
LEGACY_COUNTIES_CT = "https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_county_500k.zip"

# this contains:
#   - 2017 to 2024 embedding, wnv case counts, population, normalized case counts for each county
ALL_DATA = SCRIPT_DIR / "national_wnv_case_data/all_data_normalized.csv"

# ----- Helper functions ----- #
def get_country_boundary(country_code):
	"""
	Get country boundary using REST Countries API + overpass/nominatim
	country_code: 'CA' for Canada, 'MX' for Mexico
	"""
	# openstreetmap nomatim
	url = f"https://nominatim.openstreetmap.org/search?country={country_code}&format=geojson&polygon_geojson=1"
	response = requests.get(url, headers={'User-Agent': 'WNV-Mapping/1.0'})
	
	if response.status_code == 200:
		gdf = gpd.read_file(response.text)
		return gdf
	return None

def remove_holes(geom):
    if geom.geom_type == "Polygon":
        return Polygon(geom.exterior)
    elif geom.geom_type == "MultiPolygon":
        return MultiPolygon([Polygon(p.exterior) for p in geom.geoms])
    else:
        return geom

def add_scalebar_miles_left_endlabel(
    ax,
    anchor=(0.03, 0.055),
    bar_h=26_000,
    tick_len_frac=0.5,
    label_fontsize=BASE_FONTSIZE,
    width_frac=0.42
):
    """Scale bar with non-uniform segments (500 miles for national map)."""
    M_PER_MILE = 1609.344
    tick_values = [0, 125, 250, 500]
    seg_lengths = np.diff(tick_values)
    
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    width_m = x1 - x0
    height_m = y1 - y0
    
    total_m_draw = width_m * width_frac
    total_miles = tick_values[-1] - tick_values[0]
    unit_scale = total_m_draw / total_miles
    tick_len = bar_h * tick_len_frac
    
    axfx, axfy = anchor
    x_left = x0 + axfx * width_m
    y_base = y0 + axfy * height_m
    
    edge = "#1E2933"
    dark = "#2F3B46"
    light = "#FFFFFF"
    
    # Outer frame
    ax.add_patch(Rectangle(
        (x_left, y_base), total_m_draw, bar_h,
        facecolor="none", edgecolor=edge, linewidth=1.6, zorder=60
    ))
    
    # Draw segments
    x_curr = x_left
    for i, seg_len_miles in enumerate(seg_lengths):
        seg_m = seg_len_miles * unit_scale
        face = dark if i % 2 == 0 else light
        
        ax.add_patch(Rectangle(
            (x_curr, y_base), seg_m, bar_h,
            facecolor=face, edgecolor=edge, linewidth=1.2, zorder=61
        ))
        
        # Tick at start
        ax.plot([x_curr, x_curr], [y_base, y_base - tick_len],
                color=edge, lw=2.0, solid_capstyle="round", zorder=62)
        ax.text(x_curr, y_base - tick_len - 11_000, f"{tick_values[i]}",
                ha="center", va="top", fontsize=label_fontsize, 
                color=edge, zorder=63)
        x_curr += seg_m
    
    # Final tick
    ax.plot([x_curr, x_curr], [y_base, y_base - tick_len],
            color=edge, lw=2.0, solid_capstyle="round", zorder=62)
    ax.text(x_curr, y_base - tick_len - 11_000, f"{tick_values[-1]}",
            ha="center", va="top", fontsize=label_fontsize, 
            color=edge, zorder=63)
    
    # "Miles" label 
    ax.text(x_curr + unit_scale * 30, y_base - bar_h * 0.05, "Miles",
            ha="left", va="center", fontsize=label_fontsize, 
            color=edge, zorder=63)

def add_compass(ax, center_frac=(0.12, 0.18), size=180_000, color="#2F3B46"):
    """Compass rose."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    cx = x0 + center_frac[0] * (x1 - x0)
    cy = y0 + center_frac[1] * (y1 - y0)
    
    # Cross lines
    ax.plot([cx - size*0.8, cx + size*0.8], [cy, cy], 
            color=color, lw=1.1, zorder=70)
    ax.plot([cx, cx], [cy - size*0.8, cy + size*0.8], 
            color=color, lw=1.1, zorder=70)
    
    # North arrow
    ax.add_patch(FancyArrow(
        cx, cy, 0, size*0.95, 
        width=size*0.12, head_width=size*0.35, head_length=size*0.35,
        color=color, length_includes_head=True, zorder=71
    ))
    
    fs = BASE_FONTSIZE
    ax.text(cx, cy + size*1.05, "N", ha="center", va="bottom", 
            fontsize=fs, color=color, zorder=71)
    ax.text(cx, cy - size*0.95, "S", ha="center", va="top", 
            fontsize=fs, color=color, zorder=71)
    ax.text(cx + size*0.95, cy, "E", ha="left", va="center", 
            fontsize=fs, color=color, zorder=71)
    ax.text(cx - size*0.95, cy, "W", ha="right", va="center", 
            fontsize=fs, color=color, zorder=71)

# ----------------------------- COLORBAR CLASSES ----------------------------
class _ColorbarProxy:
    def __init__(self, cmap, norm, ticks, ticklabels=None, nsteps=64):
        self.cmap = cmap
        self.norm = norm
        self.ticks = ticks
        self.ticklabels = ticklabels if ticklabels else [f"{t:g}" for t in ticks]
        self.nsteps = nsteps

class _NonLogVerticalColorbarHandler(mpl.legend_handler.HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, 
                      width, height, fontsize, trans):
        cmap = orig_handle.cmap
        norm = orig_handle.norm
        ticks = orig_handle.ticks
        n = orig_handle.nsteps
        artists = []
        
        # Geometry
        pad_x = 0.68 * width
        bar_w = 2.0 * width
        bar_h = height * 7.5
        x0 = xdescent + pad_x
        top_pad_frac = -0.4
        y0 = ydescent + (height - bar_h) / 2 + top_pad_frac * bar_h
        
        # Gradient rectangles - SIMPLIFIED FOR LINEAR SCALE
        for i in range(n):
            y = y0 + (i / n) * bar_h
            y2 = y0 + ((i + 1) / n) * bar_h
            
            # LINEAR interpolation (removed LogNorm logic)
            val = norm.vmin + ((i + 0.5) / n) * (norm.vmax - norm.vmin)
            
            rect = Rectangle(
                (x0, y), bar_w, y2 - y, 
                facecolor=cmap(norm(val)), edgecolor="none", lw=0
            )
            rect.set_transform(trans)
            artists.append(rect)
        
        # Vertical label
        title = mpl.text.Text(
            x=x0 - 0.5 * width, y=y0 + bar_h / 2,
            text="Cases per\n100K", rotation=90,  # Updated label
            va="center", ha="center",
            fontsize=fontsize - 3.2, color="#000000"
        )
        title.set_transform(trans)
        artists.append(title)
        
        # Tick labels
        label_x = x0 + bar_w + 0.14 * width
        PAD_TOP_FRAC = 0.09
        PAD_LOW_FRAC = 0.06
        
        for i, (lab, t) in enumerate(zip(orig_handle.ticklabels, ticks)):
            frac = (t - norm.vmin) / (norm.vmax - norm.vmin)  # Linear calculation
            
            if i == 0:
                frac = min(1.0, frac + PAD_LOW_FRAC)
            elif i == len(ticks) - 1:
                frac = max(0.0, frac - PAD_TOP_FRAC)
            
            ytick = y0 + frac * bar_h
            txt = mpl.text.Text(
                x=label_x, y=ytick, text=lab,
                va="center", ha="left",
                fontsize=fontsize-1, color="#111"
            )
            txt.set_transform(trans)
            artists.append(txt)
        
        return artists

class _VerticalColorbarHandler(mpl.legend_handler.HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, 
                      width, height, fontsize, trans):
        cmap = orig_handle.cmap
        norm = orig_handle.norm
        ticks = orig_handle.ticks
        n = orig_handle.nsteps
        artists = []
        
        # Geometry
        pad_x = 0.68 * width
        # adjusted to make it less wide
        bar_w = 2.0 * width
        bar_h = height * 7.5
        x0 = xdescent + pad_x
        top_pad_frac = -0.4
        y0 = ydescent + (height - bar_h) / 2 + top_pad_frac * bar_h
        
        # Gradient rectangles
        for i in range(n):
            y = y0 + (i / n) * bar_h
            y2 = y0 + ((i + 1) / n) * bar_h
            if isinstance(norm, LogNorm):
                frac = (i + 0.5) / n
                val = np.exp(np.log(norm.vmin) * (1 - frac) + 
                           np.log(norm.vmax) * frac)
            else:
                val = norm.vmin + ((i + 0.5) / n) * (norm.vmax - norm.vmin)
            rect = Rectangle(
                (x0, y), bar_w, y2 - y, 
                facecolor=cmap(norm(val)), edgecolor="none", lw=0
            )
            rect.set_transform(trans)
            artists.append(rect)
            
        title = mpl.text.Text(
            x=x0 - 0.5 * width, y=y0 + bar_h / 2,
            text="WNV Cases Per 100k", rotation=90,
            va="center", ha="center",
            fontsize=fontsize - 3.2, color="#000000"
        )
        title.set_transform(trans)
        artists.append(title)
        
        # Tick labels
        label_x = x0 + bar_w + 0.14 * width
        PAD_TOP_FRAC = 0.09
        PAD_LOW_FRAC = 0.06
        
        for i, (lab, t) in enumerate(zip(orig_handle.ticklabels, ticks)):
            frac = norm(t)
            if i == 0:
                frac = min(1.0, frac + PAD_LOW_FRAC)
            elif i == len(ticks) - 1:
                frac = max(0.0, frac - PAD_TOP_FRAC)
            ytick = y0 + frac * bar_h
            txt = mpl.text.Text(
                x=label_x, y=ytick, text=lab,
                va="center", ha="left",
                fontsize=fontsize-1, color="#111"
            )
            txt.set_transform(trans)
            artists.append(txt)
        
        return artists

# ----- Get state and county geographies (and great lakes), neighboring countries, embeddings + WNV data ----- #
states = gpd.read_file(CENSUS_STATES)
counties = gpd.read_file(CENSUS_COUNTIES)

mexico = get_country_boundary("MX")
canada = get_country_boundary("CA")

legacy_counties = gpd.read_file(LEGACY_COUNTIES_CT)
legacy_ct_counties = legacy_counties[legacy_counties["STATEFP"] == "09"].copy()

df_all = pd.read_csv(ALL_DATA)
df_all["GEOID"] = df_all["GEOID"].astype(str).str.zfill(5)

# ----- Project to EPSG:3857 and adjust boundaries ----- #

# ignore Alaska, Hawaii, Guam, Puerto Rico, Commonwealth of the Northern Mariana Islands, American Samoa, Virgin Islands (no cases)
# including these also unneccessarily enlarge the zoom on the US map 
exclude = ["AK","HI","GU","PR","MP","AS","VI"]
exclude_sfips = ['02', '60', '15', '78', '72', '69', '66']

# historically have had no cases: # https://health.hawaii.gov/docd/disease_listing/west-nile-virus/
# https://www.usgs.gov/faqs/where-united-states-has-west-nile-virus-been-detected-wildlife
# no geoid matched with CNMI in the cases data frame

mexico = mexico.to_crs(3857)
canada = canada.to_crs(3857)
states = states.to_crs(3857)

states = states[~states["STUSPS"].isin(exclude)]
counties = counties[~counties["STATEFP"].isin(exclude_sfips)]
counties = counties.to_crs(3857)
legacy_ct_counties = legacy_ct_counties.to_crs(3857)

# ----- Merge previous long dataframe with geographies ----- #

# both use "GEOID" as unique identifier
counties_geom = counties[["GEOID","geometry"]]
df_merged = pd.merge(df_all, counties_geom, on="GEOID", how="inner")
# convert df_merged to a GeoDataFrame (I need to inspect it visually)
df_merged = gpd.GeoDataFrame(df_merged, geometry=df_merged.geometry, crs=states.crs)

# only keep cases columns (embedding data is irrelevant for these visualizations)
df_merged = df_merged[['GEOID', 'Cases_2017', 'Cases_2018', 
                       'Cases_2019', 'Cases_2020', 'Cases_2021', 
                       'Cases_2022', 'Cases_2023', 'Cases_2024',
                       'Normalized_2017','Normalized_2018','Normalized_2019',
                       'Normalized_2020','Normalized_2021','Normalized_2022',
                       'Normalized_2023','Normalized_2024', 'geometry']]

# ----------------------------- CREATE COLORMAP ----------------------------
cmap = LinearSegmentedColormap.from_list(
    "wnv_cases", ["#FFF7EF", "#8C1A3C"]
)

# # ----------------------------- PLOT LOOP ----------------------------

# # unfiltered boundaries, for alaska and hawaii again
# states_all = gpd.read_file(CENSUS_STATES).to_crs(3857)
# counties_all = gpd.read_file(CENSUS_COUNTIES).to_crs(3857)

# us_union = unary_union(states.geometry)
# us_outline = gpd.GeoDataFrame(geometry=[remove_holes(us_union)],crs=states.crs)

# for year in range(2017, 2025):
#     col = f"Cases_{year}"
#     # IMPORTANT: 2017-2022 CT data plotting
#     if year <= 2022:
#         # Use legacy CT counties for this year
#         plot_df_base = df_merged[~df_merged["GEOID"].str.startswith("09")].copy()
        
#         # Merge legacy CT counties with case data
#         legacy_ct_geom = legacy_ct_counties[["GEOID", "geometry"]]
#         ct_cases = df_all[df_all["GEOID"].str.startswith("09")][["GEOID", col]].copy()
#         ct_merged = pd.merge(ct_cases, legacy_ct_geom, on="GEOID", how="inner")
#         ct_merged = gpd.GeoDataFrame(ct_merged, geometry=ct_merged.geometry, crs=states.crs)
#         ct_merged = ct_merged.rename(columns={col: "plot_col"})
        
#         # Combine non-CT data with legacy CT data
#         plot_df_base = plot_df_base.rename(columns={col: "plot_col"})
#         plot_df = pd.concat([plot_df_base, ct_merged], ignore_index=True)
#         plot_df = gpd.GeoDataFrame(plot_df, geometry=plot_df.geometry, crs=states.crs)
#         #Create county boundaries that use legacy CT only
#         counties_to_plot = counties[~counties["STATEFP"].str.startswith("09")].copy()
#         counties_to_plot = pd.concat([counties_to_plot, legacy_ct_counties], ignore_index=True)
#         counties_to_plot = gpd.GeoDataFrame(counties_to_plot, geometry=counties_to_plot.geometry, crs=states.crs)
#     else:
#         # For 2023+, use current counties
#         plot_df = df_merged.copy()
#         plot_df = plot_df.rename(columns={col: "plot_col"})
#         # use the new county boundaries 
#         counties_to_plot = counties.copy()
    
#     # Handle zeros/missing
#     has_pos = (plot_df["plot_col"] > 0).any()
#     if has_pos:
#         vmax = int(plot_df["plot_col"].max())
#         eps = 0.8
#         plot_df["plot_col_adj"] = np.where(
#             plot_df["plot_col"] <= 0, eps, plot_df["plot_col"].astype(float)
#         )
#         norm = LogNorm(vmin=eps, vmax=vmax)
#         data_min = int(plot_df.loc[plot_df["plot_col"] > 0, "plot_col"].min())
#         data_max = vmax
#     else:
#         plot_df["plot_col_adj"] = 1.0
#         norm = mpl.colors.Normalize(vmin=0, vmax=1)
#         data_min = 0
#         data_max = 0
    
#     # Create figure
#     fig, ax = plt.subplots(figsize=(15, 8), dpi=300)
    
#     # Canada and Mexico backdrops
#     if canada is not None and not canada.empty:
#         canada.plot(ax=ax, facecolor="#E6E8EB", edgecolor="#D1D5DB", 
#                    lw=0.6, zorder=1)
#     if mexico is not None and not mexico.empty:
#         mexico.plot(ax=ax, facecolor="#E6E8EB", edgecolor="#D1D5DB", 
#                    lw=0.6, zorder=1)
    
#     # US outline backdrop
#     us_outline.plot(ax=ax, facecolor="#E6E8EB", edgecolor="#D1D5DB", 
#                    lw=0.6, zorder=1)
    
#     # Choropleth - census tracts
#     plot_df.plot(
#         ax=ax, column="plot_col_adj", cmap=cmap, norm=norm,
#         linewidth=0.05, edgecolor="#9AA3AD", alpha=1.0, zorder=2.1
#     )
    
#     # Add gray rectangle over Great Lakes region (approximate coverage)
#     # coordinates are in EPSG:3857 (Web Mercator)
#     gl_xmin, gl_xmax = -10_300_000, -8_200_000  # Longitude range
#     gl_ymin, gl_ymax = 5_050_000, 6_400_000     # Latitude range
    
#     # zorder 0 so it goes all the way in the bottom
#     ax.add_patch(Rectangle(
#         (gl_xmin, gl_ymin), gl_xmax - gl_xmin, gl_ymax - gl_ymin,
#         facecolor="#E6E8EB", edgecolor="#D1D5DB", 
#         linewidth=0, zorder=0, alpha=1.0
#     ))
    
#     # State boundaries
#     states.boundary.plot(ax=ax, edgecolor="#333333", linewidth=0.8, zorder=4)
    
#     # County boundaries (lighter)
#     # this boundary changes according to year:
#     #   2017-2022 -> use legacy boundaries for connecticut
#     #   2023-2024 -> use new boundaries for connecticut planning regions
#     counties_to_plot.boundary.plot(ax=ax, edgecolor="#9AA3AD", linewidth=0.2, zorder=3.5)
    
#     # Country labels
#     xmin, ymin, xmax, ymax = us_outline.total_bounds
#     dx, dy = xmax - xmin, ymax - ymin
#     xmid, ymid = (xmin + xmax) / 2, (ymin + ymax) / 2
    
#     txt_can = ax.text(
#         xmid + 0.12, ymax + 0.02*dy, "CANADA",
#         fontsize=COUNTRY_LABEL_FONTSIZE, fontweight="bold",
#         color="#6B7280", ha="center", va="center", zorder=5
#     )
#     # fix -> shift to the left more
#     txt_mex = ax.text(
#         xmid - 0.12*dx, ymin - 0.01*dy, "MEXICO",
#         fontsize=COUNTRY_LABEL_FONTSIZE, fontweight="bold",
#         color="#6B7280", ha="center", va="center", zorder=5
#     )
    
#     for t in [txt_can, txt_mex]:
#         t.set_path_effects([
#             pe.withStroke(linewidth=2.2, foreground="white", alpha=0.9)
#         ])
    
#     # Set extent
#     # ADJUSTED: More padding on left, less on right to shift map right
#     pad_left = 0.10 * dx
#     pad_right = 0.05 * dx
#     pad_bottom = 0.12 * dy
#     pad_top = 0.05 * dy
    
#     ax.set_xlim(xmin - pad_left, xmax + pad_right)
#     ax.set_ylim(ymin - pad_bottom, ymax + pad_top)
#     ax.set_aspect("equal", adjustable="box")
#     ax.set_axis_off()

#     # ----------------------------- ALASKA & HAWAII INSETS ----------------------------
    
#     # Load Alaska and Hawaii separately (don't filter them out)
    
#     alaska_state = states_all[states_all["STUSPS"] == "AK"]
#     hawaii_state = states_all[states_all["STUSPS"] == "HI"]
#     alaska_counties = counties_all[counties_all["STATEFP"] == "02"]
#     hawaii_counties = counties_all[counties_all["STATEFP"] == "15"]
    
#     # Merge with case data
#     alaska_data = alaska_counties.merge(plot_df[["GEOID", "plot_col_adj"]], on="GEOID", how="left")
#     hawaii_data = hawaii_counties.merge(plot_df[["GEOID", "plot_col_adj"]], on="GEOID", how="left")
    
#     # Fill NaN with eps for Alaska/Hawaii (if no data)
#     alaska_data["plot_col_adj"] = alaska_data["plot_col_adj"].fillna(eps if has_pos else 1.0)
#     hawaii_data["plot_col_adj"] = hawaii_data["plot_col_adj"].fillna(eps if has_pos else 1.0)
    
#     # ALASKA INSET (bottom left)
#     ax_ak = inset_axes(ax, width="22%", height="20%", 
#                        loc="lower left", 
#                        bbox_to_anchor=(0.01, 0.02, 1, 1),
#                        bbox_transform=ax.transAxes,
#                        borderpad=0)
    
#     # Light blue background for ocean
#     ax_ak.set_facecolor("#D4E9F7")
    
#     # Plot Alaska counties
#     alaska_data.plot(ax=ax_ak, column="plot_col_adj", cmap=cmap, norm=norm,
#                      linewidth=0.15, edgecolor="#777777", alpha=1.0, zorder=2)
    
#     # Alaska state boundary (thicker)
#     alaska_state.boundary.plot(ax=ax_ak, edgecolor="#000000", linewidth=1.2, zorder=3)
    
#     # Alaska bounds and formatting
#     ak_bounds = alaska_state.total_bounds
#     ak_dx = ak_bounds[2] - ak_bounds[0]
#     ak_dy = ak_bounds[3] - ak_bounds[1]

#     # Center on mainland Alaska (show middle portion, excluding far west Aleutians and far east islands)
#     center_x = ak_bounds[0] + 0.06*ak_dx  # Center point shifted toward mainland
#     view_width_ak = 0.18*ak_dx  # Show 25% of total width
    
#     west_limit = center_x - view_width_ak/2
#     east_limit = center_x + view_width_ak/2
    
#     ax_ak.set_xlim(west_limit, east_limit)
#     ax_ak.set_ylim(ak_bounds[1] - 0.05*ak_dy, ak_bounds[3] + 0.05*ak_dy)
#     ax_ak.set_aspect("equal")

#     # Turn off tick labels but keep the frame
#     ax_ak.set_xticks([])
#     ax_ak.set_yticks([])
    
#     # Add black border to Alaska inset
#     for spine in ax_ak.spines.values():
#         spine.set_visible(True)
#         spine.set_edgecolor("#000000")
#         spine.set_linewidth(1.5)
    
#     # Hawaii inset
#     ax_hi = inset_axes(ax, width="15%", height="10%", 
#                        loc="lower left",
#                        bbox_to_anchor=(0.20, 0.02, 1, 1),
#                        bbox_transform=ax.transAxes,
#                        borderpad=0)
    
#     # Light blue background for ocean
#     ax_hi.set_facecolor("#D4E9F7")
    
#     # Plot Hawaii counties
#     hawaii_data.plot(ax=ax_hi, column="plot_col_adj", cmap=cmap, norm=norm,
#                      linewidth=0.15, edgecolor="#777777", alpha=1.0, zorder=2)
    
#     # Hawaii state boundary (thicker)
#     hawaii_state.boundary.plot(ax=ax_hi, edgecolor="#000000", linewidth=1.2, zorder=3)
    
#     # Turn off tick labels but keep the frame
#     ax_hi.set_xticks([])
#     ax_hi.set_yticks([])

#     # Hawaii bounds and formatting
#     hi_bounds = hawaii_state.total_bounds
#     hi_dx = hi_bounds[2] - hi_bounds[0]
#     hi_dy = hi_bounds[3] - hi_bounds[1]

#     # Center on main Hawaiian island chain
#     center_x = hi_bounds[0] + 0.85*hi_dx  # Center on main islands (toward the east)
#     view_width_hi = 0.60*hi_dx  # Show 45% of total width
    
#     west_limit = center_x - view_width_hi/2
#     east_limit = center_x + view_width_hi/2
    
#     ax_hi.set_xlim(west_limit, east_limit)
#     ax_hi.set_ylim(hi_bounds[1] - 0.05*hi_dy, hi_bounds[3] + 0.05*hi_dy)
#     ax_hi.set_aspect("equal")
    
#     # Add black border to Hawaii inset
#     for spine in ax_hi.spines.values():
#         spine.set_visible(True)
#         spine.set_edgecolor("#000000")
#         spine.set_linewidth(1.5)
    
#     # ----------------------------- LEGEND ----------------------------
#     # Build colorbar proxy
#     if has_pos:
#         low_pos = float(norm.vmin)
#         high_pos = float(norm.vmax)
#         ticks = [low_pos, high_pos]
#         ticklabels = [f"Low: {data_min}", f"High: {data_max}"]
#     else:
#         ticks = [0.0, 1.0]
#         ticklabels = [f"Low: {data_min}", f"High: {data_max}"]
    
#     colorbar_proxy = _ColorbarProxy(
#         cmap=cmap, norm=norm, ticks=ticks, ticklabels=ticklabels
#     )
		
# 		# adding to legend - county box handle 
#     county_handle = Rectangle(
#         (0, 0), width=1.0, height=0.6,
#         facecolor='none', edgecolor='#9AA3AD', linewidth=0.8
#     )
#     # adding to legend - state box handle 
#     state_handle = Rectangle(
#         (0, 0), width=1.0, height=0.6,
#         facecolor='none', edgecolor='#333333', linewidth=0.8
#     )
    
#     # Legend items
#     num_colorbar_rows = 3
#     invisible_handles = [Patch(alpha=0)] * num_colorbar_rows
    
#     handles = [county_handle] + [state_handle] + [colorbar_proxy] + invisible_handles
#     labels = ["County", "State", ""] + [""] * num_colorbar_rows 
    
#     # Remove old legend if present
#     for child in ax.get_children():
#         if isinstance(child, mpl.legend.Legend):
#             child.remove()
    
#     leg = ax.legend(
#         handles=handles, labels=labels,
#         handler_map={_ColorbarProxy: _VerticalColorbarHandler()},
#         title="Legend",
#         loc="lower right",
#         # no longer need LEFT_ANCHOR (first param)
#         bbox_to_anchor=(0.835, 0.081, 0.165, 0.6),
#         frameon=True, framealpha=1.0,
#         edgecolor="#B8BEC5", facecolor="#FFFFFF",
#         fontsize=BASE_FONTSIZE, title_fontsize=10,
#         alignment="left", mode="expand",
#         borderpad=1.0, labelspacing=1.0,
#         handlelength=1.6, handletextpad=0.5
#     )
    
#     # Bold legend title and header
#     if leg.get_title() is not None:
#         leg.get_title().set_fontweight("bold")
#         leg.get_title().set_ha("left")
    
#     # Title above legend
#     fig.canvas.draw()
#     renderer = fig.canvas.get_renderer()
#     bbox_px = leg.get_window_extent(renderer=renderer)
#     (x0_ax, y0_ax) = ax.transAxes.inverted().transform((bbox_px.x0, bbox_px.y0))
#     (x1_ax, y1_ax) = ax.transAxes.inverted().transform((bbox_px.x1, bbox_px.y1))
    
#     title_x = x0_ax
#     title_y = y1_ax - 0.03 
#     title_text = f"West Nile Virus\nCases by County ({year})"
    
#     t = ax.text(
#         title_x, title_y, title_text,
#         transform=ax.transAxes,
#         ha="left", va="bottom",
#         fontsize=10.5, fontweight="bold",
#         color="#111", zorder=200
#     )
#     t.set_path_effects([
#         pe.withStroke(linewidth=3, foreground="white", alpha=0.9)
#     ])
    
#     # Scale bar and compass
#     add_scalebar_miles_left_endlabel(
#         ax, anchor=(0.845, 0.04), # right below the legend
#         bar_h=40_000, width_frac=0.15
#     )
#     add_compass(
#         ax, center_frac=(0.94, 0.55), 
#         size=200_000
#     )
    
#     # Save
#     plt.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.06)
#     plt.savefig(
#         SCRIPT_DIR / f"wnv_case_maps/wnv_cases_map_{year}.png",
#         dpi=300, bbox_inches="tight", facecolor="white"
#     )
#     plt.close(fig)
#     print(f"Saved map for {year}")


# ----------------------------- PLOT LOOP - NORMALIZED DATA UPDATE ----------------------------

states_all = gpd.read_file(CENSUS_STATES).to_crs(3857)
counties_all = gpd.read_file(CENSUS_COUNTIES).to_crs(3857)

us_union = unary_union(states.geometry)
us_outline = gpd.GeoDataFrame(geometry=[remove_holes(us_union)],crs=states.crs)

for year in range(2017, 2025):
    col = f"Normalized_{year}"
    # IMPORTANT: 2017-2022 CT data plotting
    if year <= 2022:
        # Use legacy CT counties for this year
        plot_df_base = df_merged[~df_merged["GEOID"].str.startswith("09")].copy()
        
        # Merge legacy CT counties with case data
        legacy_ct_geom = legacy_ct_counties[["GEOID", "geometry"]]
        ct_cases = df_all[df_all["GEOID"].str.startswith("09")][["GEOID", col]].copy()
        ct_merged = pd.merge(ct_cases, legacy_ct_geom, on="GEOID", how="inner")
        ct_merged = gpd.GeoDataFrame(ct_merged, geometry=ct_merged.geometry, crs=states.crs)
        ct_merged = ct_merged.rename(columns={col: "plot_col"})
        
        # Combine non-CT data with legacy CT data
        plot_df_base = plot_df_base.rename(columns={col: "plot_col"})
        plot_df = pd.concat([plot_df_base, ct_merged], ignore_index=True)
        plot_df = gpd.GeoDataFrame(plot_df, geometry=plot_df.geometry, crs=states.crs)
        #Create county boundaries that use legacy CT only
        counties_to_plot = counties[~counties["STATEFP"].str.startswith("09")].copy()
        counties_to_plot = pd.concat([counties_to_plot, legacy_ct_counties], ignore_index=True)
        counties_to_plot = gpd.GeoDataFrame(counties_to_plot, geometry=counties_to_plot.geometry, crs=states.crs)
    else:
        # For 2023+, use current counties
        plot_df = df_merged.copy()
        plot_df = plot_df.rename(columns={col: "plot_col"})
        # use the new county boundaries 
        counties_to_plot = counties.copy()

    # Handle zeros/missing
    has_pos = (plot_df["plot_col"] > 0).any()
    if has_pos:
        data_min = float(plot_df.loc[plot_df["plot_col"] > 0, "plot_col"].min())
        data_max = float(plot_df["plot_col"].max())
        
        # Use LINEAR normalization (not log)
        plot_df["plot_col_adj"] = plot_df["plot_col"].copy()
        norm = mpl.colors.Normalize(vmin=0, vmax=data_max)
    else:
        # No positive values
        print(f"Warning: Year {year} has no positive values")
        plot_df["plot_col_adj"] = 0.0
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        data_min = 0
        data_max = 0
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 8), dpi=300)
    
    # Canada and Mexico backdrops
    if canada is not None and not canada.empty:
        canada.plot(ax=ax, facecolor="#E6E8EB", edgecolor="#D1D5DB", 
                   lw=0.6, zorder=1)
    if mexico is not None and not mexico.empty:
        mexico.plot(ax=ax, facecolor="#E6E8EB", edgecolor="#D1D5DB", 
                   lw=0.6, zorder=1)
    
    # US outline backdrop
    us_outline.plot(ax=ax, facecolor="#E6E8EB", edgecolor="#D1D5DB", 
                   lw=0.6, zorder=1)
    
    # Choropleth - census tracts
    plot_df.plot(
        ax=ax, column="plot_col_adj", cmap=cmap, norm=norm,
        linewidth=0.05, edgecolor="#9AA3AD", alpha=1.0, zorder=2.1
    )
    
    # Add gray rectangle over Great Lakes region (approximate coverage)
    # coordinates are in EPSG:3857 (Web Mercator)
    gl_xmin, gl_xmax = -10_300_000, -8_200_000  # Longitude range
    gl_ymin, gl_ymax = 5_050_000, 6_400_000     # Latitude range
    
    # zorder 0 so it goes all the way in the bottom
    ax.add_patch(Rectangle(
        (gl_xmin, gl_ymin), gl_xmax - gl_xmin, gl_ymax - gl_ymin,
        facecolor="#E6E8EB", edgecolor="#D1D5DB", 
        linewidth=0, zorder=0, alpha=1.0
    ))
    
    # State boundaries
    states.boundary.plot(ax=ax, edgecolor="#333333", linewidth=0.8, zorder=4)
    
    # County boundaries (lighter)
    # this boundary changes according to year:
    #   2017-2022 -> use legacy boundaries for connecticut
    #   2023-2024 -> use new boundaries for connecticut planning regions
    counties_to_plot.boundary.plot(ax=ax, edgecolor="#9AA3AD", linewidth=0.2, zorder=3.5)
    
    # Country labels
    xmin, ymin, xmax, ymax = us_outline.total_bounds
    dx, dy = xmax - xmin, ymax - ymin
    xmid, ymid = (xmin + xmax) / 2, (ymin + ymax) / 2
    
    txt_can = ax.text(
        xmid + 0.12, ymax + 0.02*dy, "CANADA",
        fontsize=COUNTRY_LABEL_FONTSIZE, fontweight="bold",
        color="#6B7280", ha="center", va="center", zorder=5
    )
    # fix -> shift to the left more
    txt_mex = ax.text(
        xmid - 0.12*dx, ymin - 0.01*dy, "MEXICO",
        fontsize=COUNTRY_LABEL_FONTSIZE, fontweight="bold",
        color="#6B7280", ha="center", va="center", zorder=5
    )
    
    for t in [txt_can, txt_mex]:
        t.set_path_effects([
            pe.withStroke(linewidth=2.2, foreground="white", alpha=0.9)
        ])
    
    # Set extent
    # ADJUSTED: More padding on left, less on right to shift map right
    pad_left = 0.10 * dx
    pad_right = 0.05 * dx
    pad_bottom = 0.12 * dy
    pad_top = 0.05 * dy
    
    ax.set_xlim(xmin - pad_left, xmax + pad_right)
    ax.set_ylim(ymin - pad_bottom, ymax + pad_top)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()

    # ----------------------------- ALASKA & HAWAII INSETS ----------------------------
    
    # Load Alaska and Hawaii separately (don't filter them out)
    
    alaska_state = states_all[states_all["STUSPS"] == "AK"]
    hawaii_state = states_all[states_all["STUSPS"] == "HI"]
    alaska_counties = counties_all[counties_all["STATEFP"] == "02"]
    hawaii_counties = counties_all[counties_all["STATEFP"] == "15"]
    
    # Merge with ORIGINAL data (df_all, not plot_df) since AK/HI were excluded from plot_df
    alaska_cases = df_all[df_all["GEOID"].str.startswith("02")][["GEOID", col]].copy()
    hawaii_cases = df_all[df_all["GEOID"].str.startswith("15")][["GEOID", col]].copy()

    # Merge with case data
    alaska_data = alaska_counties.merge(alaska_cases, on="GEOID", how="left")
    hawaii_data = hawaii_counties.merge(hawaii_cases, on="GEOID", how="left")

    # Rename column to match and create adjusted column
    alaska_data = alaska_data.rename(columns={col: "plot_col"})
    hawaii_data = hawaii_data.rename(columns={col: "plot_col"})

    # Create plot_col_adj for Alaska/Hawaii using the same normalization
    if has_pos:
        alaska_data["plot_col_adj"] = alaska_data["plot_col"].fillna(0)
        hawaii_data["plot_col_adj"] = hawaii_data["plot_col"].fillna(0)
    else:
        alaska_data["plot_col_adj"] = 0.0
        hawaii_data["plot_col_adj"] = 0.0
    
    # ALASKA INSET (bottom left)
    ax_ak = inset_axes(ax, width="22%", height="20%", 
                       loc="lower left", 
                       bbox_to_anchor=(0.01, 0.02, 1, 1),
                       bbox_transform=ax.transAxes,
                       borderpad=0)
    
    # Light blue background for ocean
    ax_ak.set_facecolor("#D4E9F7")
    
    # Plot Alaska counties
    alaska_data.plot(ax=ax_ak, column="plot_col_adj", cmap=cmap, norm=norm,
                     linewidth=0.15, edgecolor="#777777", alpha=1.0, zorder=2)
    
    # Alaska state boundary (thicker)
    alaska_state.boundary.plot(ax=ax_ak, edgecolor="#000000", linewidth=1.2, zorder=3)
    
    # Alaska bounds and formatting
    ak_bounds = alaska_state.total_bounds
    ak_dx = ak_bounds[2] - ak_bounds[0]
    ak_dy = ak_bounds[3] - ak_bounds[1]

    # Center on mainland Alaska (show middle portion, excluding far west Aleutians and far east islands)
    center_x = ak_bounds[0] + 0.06*ak_dx  # Center point shifted toward mainland
    view_width_ak = 0.18*ak_dx  # Show 25% of total width
    
    west_limit = center_x - view_width_ak/2
    east_limit = center_x + view_width_ak/2
    
    ax_ak.set_xlim(west_limit, east_limit)
    ax_ak.set_ylim(ak_bounds[1] - 0.05*ak_dy, ak_bounds[3] + 0.05*ak_dy)
    ax_ak.set_aspect("equal")

    # Turn off tick labels but keep the frame
    ax_ak.set_xticks([])
    ax_ak.set_yticks([])
    
    # Add black border to Alaska inset
    for spine in ax_ak.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#000000")
        spine.set_linewidth(1.5)
    
    # Hawaii inset
    ax_hi = inset_axes(ax, width="15%", height="10%", 
                       loc="lower left",
                       bbox_to_anchor=(0.20, 0.02, 1, 1),
                       bbox_transform=ax.transAxes,
                       borderpad=0)
    
    # Light blue background for ocean
    ax_hi.set_facecolor("#D4E9F7")
    
    # Plot Hawaii counties
    hawaii_data.plot(ax=ax_hi, column="plot_col_adj", cmap=cmap, norm=norm,
                     linewidth=0.15, edgecolor="#777777", alpha=1.0, zorder=2)
    
    # Hawaii state boundary (thicker)
    hawaii_state.boundary.plot(ax=ax_hi, edgecolor="#000000", linewidth=1.2, zorder=3)
    
    # Turn off tick labels but keep the frame
    ax_hi.set_xticks([])
    ax_hi.set_yticks([])

    # Hawaii bounds and formatting
    hi_bounds = hawaii_state.total_bounds
    hi_dx = hi_bounds[2] - hi_bounds[0]
    hi_dy = hi_bounds[3] - hi_bounds[1]

    # Center on main Hawaiian island chain
    center_x = hi_bounds[0] + 0.85*hi_dx  # Center on main islands (toward the east)
    view_width_hi = 0.60*hi_dx  # Show 45% of total width
    
    west_limit = center_x - view_width_hi/2
    east_limit = center_x + view_width_hi/2
    
    ax_hi.set_xlim(west_limit, east_limit)
    ax_hi.set_ylim(hi_bounds[1] - 0.05*hi_dy, hi_bounds[3] + 0.05*hi_dy)
    ax_hi.set_aspect("equal")
    
    # Add black border to Hawaii inset
    for spine in ax_hi.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#000000")
        spine.set_linewidth(1.5)
    
    # ----------------------------- LEGEND ----------------------------
    # Build colorbar proxy
    if has_pos:
        ticks = [0.0, data_max]
        ticklabels = [f"Low: {data_min:.2f}", f"High: {data_max:.2f}"]
    else:
        ticks = [0.0, 1.0]
        ticklabels = ["Low: 0.00", "High: 0.00"]
    
    colorbar_proxy = _ColorbarProxy(
        cmap=cmap, norm=norm, ticks=ticks, ticklabels=ticklabels
    )
		
		# adding to legend - county box handle 
    county_handle = Rectangle(
        (0, 0), width=1.0, height=0.6,
        facecolor='none', edgecolor='#9AA3AD', linewidth=0.8
    )
    # adding to legend - state box handle 
    state_handle = Rectangle(
        (0, 0), width=1.0, height=0.6,
        facecolor='none', edgecolor='#333333', linewidth=0.8
    )
    
    # Legend items
    num_colorbar_rows = 3
    invisible_handles = [Patch(alpha=0)] * num_colorbar_rows
    
    handles = [county_handle] + [state_handle] + [colorbar_proxy] + invisible_handles
    labels = ["County", "State", ""] + [""] * num_colorbar_rows 
    
    # Remove old legend if present
    for child in ax.get_children():
        if isinstance(child, mpl.legend.Legend):
            child.remove()
    
    leg = ax.legend(
        handles=handles, labels=labels,
        handler_map={_ColorbarProxy: _NonLogVerticalColorbarHandler()},
        title="Legend",
        loc="lower right",
        # no longer need LEFT_ANCHOR (first param)
        bbox_to_anchor=(0.835, 0.081, 0.165, 0.6),
        frameon=True, framealpha=1.0,
        edgecolor="#B8BEC5", facecolor="#FFFFFF",
        fontsize=BASE_FONTSIZE, title_fontsize=10,
        alignment="left", mode="expand",
        borderpad=1.0, labelspacing=1.0,
        handlelength=1.6, handletextpad=0.5
    )
    
    # Bold legend title and header
    if leg.get_title() is not None:
        leg.get_title().set_fontweight("bold")
        leg.get_title().set_ha("left")
    
    # Title above legend
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox_px = leg.get_window_extent(renderer=renderer)
    (x0_ax, y0_ax) = ax.transAxes.inverted().transform((bbox_px.x0, bbox_px.y0))
    (x1_ax, y1_ax) = ax.transAxes.inverted().transform((bbox_px.x1, bbox_px.y1))
    
    title_x = x0_ax
    title_y = y1_ax - 0.03 
    title_text = f"West Nile Virus\nCases per 100k by \nCounty ({year})"
    
    t = ax.text(
        title_x, title_y, title_text,
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=10.5, fontweight="bold",
        color="#111", zorder=200
    )
    t.set_path_effects([
        pe.withStroke(linewidth=3, foreground="white", alpha=0.9)
    ])
    
    # Scale bar and compass
    add_scalebar_miles_left_endlabel(
        ax, anchor=(0.845, 0.04), # right below the legend
        bar_h=40_000, width_frac=0.15
    )
    add_compass(
        ax, center_frac=(0.94, 0.55), 
        size=200_000
    )
    
    # Save
    plt.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.06)
    plt.savefig(
        SCRIPT_DIR / f"norm_wnv_case_maps/norm_wnv_cases_map_{year}.png",
        dpi=300, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)
    print(f"Saved map for {year}")