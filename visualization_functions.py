import matplotlib.pyplot as plt
import seaborn as sns
import ee # Google Earth Engine
import geemap # Google Earth Engine
import xarray # for nc file
import folium
from folium.plugins import MarkerCluster
import leafmap.foliumap as leafmap
import os
import geopandas as gpd


def plot_cluster_map(
    facility_gdf,
    tailing_gdf,
    polygon_gdf,
    facility_id_col="main_id",
    tailing_id_col="tailing_id",
    polygon_id_col="tang_id",
    cluster_col="cluster_id",
    output_html="cluster_map.html",
    simplify_tolerance=0.0005,
    open_in_browser=False,
    zoom_start=4,
    center=(56.1304, -106.3468)  # Approximate center of Canada
):
    """
    Creates an interactive map of facilities, tailings, and polygons with cluster IDs in tooltips.

    Parameters:
        facility_gdf, tailing_gdf, polygon_gdf (GeoDataFrames): Input data
        *_id_col (str): ID column name for each type
        cluster_col (str): Shared column storing the cluster label
        output_html (str): Path to output HTML map
        simplify_tolerance (float): Simplify polygon geometries to reduce map size
        open_in_browser (bool): Open the map in a browser after saving
    """

    def _prepare_gdf(gdf, simplify=False):
        gdf = gdf.copy()
        if gdf.crs is None or gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs("EPSG:4326")
        if simplify and "geometry" in gdf.columns:
            gdf["geometry"] = gdf.geometry.simplify(simplify_tolerance)
        return gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].reset_index(drop=True)

    # Reproject and simplify as needed
    facility_gdf = _prepare_gdf(facility_gdf)
    tailing_gdf = _prepare_gdf(tailing_gdf)
    polygon_gdf = _prepare_gdf(polygon_gdf, simplify=True)

    m = folium.Map(location=center, zoom_start=zoom_start, tiles="CartoDB positron")

    # Facility markers
    facility_cluster = MarkerCluster(name="Facilities").add_to(m)
    for _, row in facility_gdf.iterrows():
        tooltip = f"{facility_id_col}: {row.get(facility_id_col, 'Unknown')}<br>Cluster: {row.get(cluster_col, 'No cluster')}"
        folium.Marker(
            location=[row.geometry.y, row.geometry.x],
            tooltip=tooltip,
            icon=folium.Icon(color="blue", icon="industry", prefix="fa")
        ).add_to(facility_cluster)

    # Tailing markers
    tailing_cluster = MarkerCluster(name="Tailings").add_to(m)
    for _, row in tailing_gdf.iterrows():
        tooltip = f"{tailing_id_col}: {row.get(tailing_id_col, 'Unknown')}<br>Cluster: {row.get(cluster_col, 'No cluster')}"
        folium.Marker(
            location=[row.geometry.y, row.geometry.x],
            tooltip=tooltip,
            icon=folium.Icon(color="red", icon="trash", prefix="fa")
        ).add_to(tailing_cluster)

    # Polygons with tooltip
    folium.GeoJson(
        polygon_gdf,
        name="Polygons",
        style_function=lambda x: {
            "fillColor": "#FFD700",  # Gold
            "color": "#333333",
            "weight": 1.2,
            "fillOpacity": 0.3
        },
        tooltip=folium.GeoJsonTooltip(
            fields=[polygon_id_col, cluster_col],
            aliases=["Polygon ID", "Cluster ID"],
            localize=True
        )
    ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save(output_html)
    print(f"✅ Map saved to {output_html}")

    if open_in_browser:
        import webbrowser
        webbrowser.open(output_html)


def plot_polygons_per_facility(df):
    """
    Plots a histogram showing the distribution of the number of polygons per facility.
    """
    polygons_per_facility = df.groupby("main_id")["wpda_id"].nunique()

    # Adjusted histogram with each count clearly positioned on the x-axis
    plt.figure(figsize=(12, 6))
    bins = range(0, polygons_per_facility.max() + 2)  # Ensure each count has its own bin
    sns.histplot(polygons_per_facility, bins=bins, kde=True, discrete=True)  # Ensure bars align properly
    plt.xlabel("Number of Polygons per Facility")
    plt.ylabel("Count of Facilities")
    plt.title("Dispersion of Polygons per Facility (Properly Aligned Bars)")
    plt.xticks(bins)  # Ensure each integer is labeled
    plt.show()


def plot_mine_status_mincan(df):
    """
    Plots two figures:
    1. Mine openings and closures over time, with active mine count.
    2. Current active vs. inactive mines as a pie chart.

    Parameters:
    df (pd.DataFrame): DataFrame containing columns 'open1', 'close1', 'open2', 'close2', 'open3', 'close3', and 'mine_status'.
    """

    # Extract relevant columns related to opening and closing years
    open_close_columns = ['open1', 'close1', 'open2', 'close2', 'open3', 'close3']

    # Convert columns to numeric, handling non-numeric values (like 'open')
    df_filtered = df[open_close_columns].apply(pd.to_numeric, errors='coerce')

    # Reshape data into long format
    df_melted = df_filtered.melt(value_name='year', var_name='event').dropna()

    # Determine whether the event is an opening (+1) or closing (-1)
    df_melted['count'] = df_melted['event'].apply(lambda x: 1 if 'open' in x else -1)

    # Aggregate counts per year
    df_yearly = df_melted.groupby('year')['count'].sum().reset_index()

    # Compute cumulative sum for active mines
    df_yearly['active_mines'] = df_yearly['count'].cumsum()

    # --- Figure 1: Mine Openings and Closures Over Time ---
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(df_yearly['year'], df_yearly['count'], color=['green' if x > 0 else 'red' for x in df_yearly['count']],
            label="Openings/Closures")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Number of Openings (+) / Closures (-)")
    ax1.axhline(0, color='black', linewidth=0.8)
    ax1.legend(loc="upper left")

    # --- Figure 2: Current Active vs. Inactive Mines ---
    mine_status_counts = df['mine_status'].value_counts()

    plt.figure(figsize=(6, 6))
    plt.pie(mine_status_counts, labels=mine_status_counts.index, autopct='%1.1f%%', colors=['green', 'red'])
    plt.title("Current Active vs. Inactive Mines")
    plt.show()