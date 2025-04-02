def plot_map(full_facility_gdf, full_polygon_gdf,
             output_html="full_facility_polygon_map.html",
             open_in_browser=False):
    """
    Generates a Folium map displaying all facilities (points) and polygons.

    Parameters:
        full_facility_gdf (gpd.GeoDataFrame): Facility dataset.
        full_polygon_gdf (gpd.GeoDataFrame): Polygon dataset.
        output_html (str): File path to save the map.
        open_in_browser (bool): Whether to open the map in a browser automatically.

    Returns:
        None (Map is saved as an HTML file).
    """

    # Ensure both GeoDataFrames are in EPSG:4326 for Folium compatibility
    full_facility_gdf = full_facility_gdf.to_crs("EPSG:4326")
    full_polygon_gdf = full_polygon_gdf.to_crs("EPSG:4326")

    # Set default map view centered on Canada
    canada_center = (56.1304, -106.3468)
    m = folium.Map(location=canada_center, zoom_start=4)

    # Marker cluster for facilities
    marker_cluster = MarkerCluster().add_to(m)

    # Plot all facilities (black markers)
    for _, row in full_facility_gdf.iterrows():
        folium.Marker(
            location=[row.geometry.y, row.geometry.x],
            popup=f"Facility ID: {row.get('facility_id', 'Unknown')}",
            icon=folium.Icon(color="black"),
        ).add_to(marker_cluster)

    # Plot all polygons (hashed black)
    for _, row in full_polygon_gdf.iterrows():
        folium.GeoJson(
            row.geometry,
            tooltip=f"Polygon ID: {row.get('polygon_id', 'Unknown')}",
            style_function=lambda feature: {
                "fillColor": "#999999",  # Gray fill
                "color": "#000000",  # Black outline
                "weight": 1.5,
                "fillOpacity": 0.4,
            },
        ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save(output_html)
    print(f"Full Facility & Polygon Map saved to {output_html}")

    # Avoid rendering large maps in Jupyter, but allow browser opening
    if open_in_browser:
        import webbrowser
        webbrowser.open(output_html)


def visualize_clusters_folium(facility_gdf, polygon_gdf,
                              facility_id_col="facility_id", polygon_id_col="polygon_id",
                              cluster_id_col="cluster_id",
                              output_html="cluster_map.html", open_in_browser=False):
    """
    Creates an interactive Folium map displaying facilities and polygons:
    - Facilities and polygons **are colored by their cluster_id**.
    - Clicking on a feature **displays facility_id, polygon_id, and cluster_id**.
    - Saves the map as an **HTML file for easy viewing**.

    Parameters:
        facility_gdf (gpd.GeoDataFrame): Facility dataset (must include cluster IDs).
        polygon_gdf (gpd.GeoDataFrame): Polygon dataset (must include cluster IDs).
        facility_id_col (str): Column name for facility IDs.
        polygon_id_col (str): Column name for polygon IDs.
        cluster_id_col (str): Column name for cluster assignments.
        output_html (str): File path to save the map.
        open_in_browser (bool): Whether to open the map in a browser automatically.

    Returns:
        folium.Map: The generated interactive map.
    """

    # Convert GeoDataFrames to EPSG:4326 for Folium compatibility
    facility_gdf = facility_gdf.to_crs("EPSG:4326")
    polygon_gdf = polygon_gdf.to_crs("EPSG:4326")

    # Create a base map centered on Canada
    canada_center = (56.1304, -106.3468)
    m = folium.Map(location=canada_center, zoom_start=4)

    # Assign unique colors for each cluster (random colors)
    unique_clusters = sorted(polygon_gdf[cluster_id_col].dropna().unique())
    cluster_colors = {cid: "#{:06x}".format(random.randint(0, 0xFFFFFF)) for cid in unique_clusters}
    cluster_colors[None] = "#999999"  # Gray for unclustered features

    # Plot facilities as markers
    marker_cluster = MarkerCluster().add_to(m)

    for _, row in facility_gdf.iterrows():
        cluster = row.get(cluster_id_col)
        color = cluster_colors.get(cluster, "#999999")  # Default gray for unclustered

        folium.Marker(
            location=[row.geometry.y, row.geometry.x],
            popup=f"<b>Facility ID:</b> {row.get(facility_id_col, 'Unknown')}<br>"
                  f"<b>Cluster ID:</b> {cluster if cluster is not None else 'Unassigned'}",
            icon=folium.Icon(color="blue" if cluster is None else "red"),
        ).add_to(marker_cluster)

    # Plot polygons as colored shapes
    for _, row in polygon_gdf.iterrows():
        cluster = row.get(cluster_id_col)
        color = cluster_colors.get(cluster, "#999999")  # Default gray for unclustered

        folium.GeoJson(
            row["geometry"],
            tooltip=f"<b>Polygon ID:</b> {row.get(polygon_id_col, 'Unknown')}<br>"
                    f"<b>Cluster ID:</b> {cluster if cluster is not None else 'Unassigned'}",
            style_function=lambda feature, color=color: {
                "fillColor": color,
                "color": color,
                "weight": 1.5,
                "fillOpacity": 0.5,
            },
        ).add_to(m)

    # Save the map
    m.save(output_html)
    print(f"\n✅ Map saved to {output_html}")

    # Open in browser (optional)
    if open_in_browser:
        import webbrowser
        webbrowser.open(output_html)

    return m


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

