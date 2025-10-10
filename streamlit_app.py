# app.py
import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely import wkt
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import os

st.set_page_config(page_title="MetalliCan Explorer", layout="wide")

# ------- CONFIG -------
CSV_PATH = "database/CSV/main_table.csv"  # adapte si besoin

# ------- UTIL - LOAD DATA -------
@st.cache_data
def load_main(csv_path=CSV_PATH):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Le fichier '{csv_path}' est introuvable.")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    # Ensure geometry column exists
    if "geometry" not in df.columns:
        # try lat/lon fallback
        if {"latitude", "longitude"}.issubset(df.columns):
            df["geometry"] = df.apply(lambda r: f"POINT ({r['longitude']} {r['latitude']})", axis=1)
        else:
            raise ValueError("Aucune colonne 'geometry' ni 'latitude'/'longitude' trouvée dans le CSV.")
    # Convert WKT -> shapely -> extract lon/lat
    def safe_point_to_xy(g):
        try:
            p = wkt.loads(g)
            return p.x, p.y
        except Exception:
            return None, None

    coords = df["geometry"].fillna("").apply(safe_point_to_xy)
    df["longitude"] = coords.apply(lambda v: v[0])
    df["latitude"] = coords.apply(lambda v: v[1])

    # Drop rows without valid coordinates
    df = df[~(df["longitude"].isna() | df["latitude"].isna())].copy()
    # Make GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326")
    return gdf

gdf = load_main()

# ------- SIDEBAR FILTERS -------
st.sidebar.title("Filtres")
# dynamic options
province_opts = sorted(gdf["province"].dropna().unique().tolist())
commodity_opts = sorted(gdf["commodities"].dropna().unique().tolist())
mining_opts = sorted(gdf["mining_processing_type"].dropna().unique().tolist())
status_opts = sorted(gdf["status"].dropna().unique().tolist())

provinces = st.sidebar.multiselect("Province", options=province_opts, default=province_opts)
commodities = st.sidebar.multiselect("Commodity (commodities)", options=commodity_opts, default=[])
mining_types = st.sidebar.multiselect("Mining / Processing type", options=mining_opts, default=[])
statuses = st.sidebar.multiselect("Status", options=status_opts, default=[])

# quick search
text_search = st.sidebar.text_input("Recherche (nom d'installation, entreprise...)")

# ------- APPLY FILTERS -------
filtered = gdf.copy()
if provinces:
    filtered = filtered[filtered["province"].isin(provinces)]
if commodities:
    filtered = filtered[filtered["commodities"].isin(commodities)]
if mining_types:
    filtered = filtered[filtered["mining_processing_type"].isin(mining_types)]
if statuses:
    filtered = filtered[filtered["status"].isin(statuses)]
if text_search:
    q = text_search.lower()
    filtered = filtered[filtered.apply(lambda r:
                                       q in str(r.get("facility_name", "")).lower()
                                       or q in str(r.get("company_name", "")).lower(), axis=1)]

# ------- LAYOUT -------
st.title("MetalliCan — Explorer les installations minières (Main table)")

col1, col2 = st.columns([2, 1])

# Map in col1
with col1:
    st.subheader("Carte des installations")
    # base map centered on Canada
    # if filtered not empty set center to median of filtered
    if len(filtered) > 0:
        center_lat = float(filtered["latitude"].median())
        center_lon = float(filtered["longitude"].median())
        zoom_start = 5
    else:
        center_lat, center_lon, zoom_start = 56.0, -96.0, 3

    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, tiles="CartoDB positron")
    marker_cluster = MarkerCluster().add_to(m)

    # Add markers
    for _, row in filtered.iterrows():
        try:
            lat = float(row["latitude"])
            lon = float(row["longitude"])
        except Exception:
            continue

        # construct html popup
        popup_html = f"""
        <b>{row.get('facility_name','')}</b><br/>
        Company: {row.get('company_name','')}<br/>
        Province: {row.get('province','')}<br/>
        Commodities: {row.get('commodities','')}<br/>
        Type: {row.get('mining_processing_type','')}<br/>
        Status: {row.get('status','')}<br/>
        <small style='color:gray'>main_id: {row.get('main_id','')}</small>
        """
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=350),
            tooltip=row.get("facility_name", "")
        ).add_to(marker_cluster)

    # render map with streamlit_folium (captures clicks)
    st_map = st_folium(m, width=700, height=600)

# Details / table in col2
with col2:
    st.subheader("Liste des installations (filtrées)")
    # show a few columns
    table_cols = ["main_id", "facility_name", "company_name", "province", "commodities", "mining_processing_type", "status"]
    display_df = filtered[table_cols].reset_index(drop=True)
    st.dataframe(display_df, height=350)

    # facility selection
    st.write("---")
    sel_name = st.selectbox("Sélectionner une facility pour détails", options=[""] + display_df["facility_name"].tolist())
    selected_row = None
    if sel_name:
        selected_row = filtered[filtered["facility_name"] == sel_name].iloc[0]

    # try capture click on map (user clicked a marker)
    clicked = None
    # old/new versions of streamlit_folium return different keys
    if isinstance(st_map, dict):
        clicked = st_map.get("last_clicked") or st_map.get("last_object_clicked") or st_map.get("clicked")
    if clicked and "lat" in clicked and "lng" in clicked:
        # find nearest facility to click location
        click_lat, click_lon = clicked.get("lat"), clicked.get("lng")
        if click_lat and click_lon:
            # compute simple euclidean to find the row
            filtered["__dist"] = (filtered["latitude"] - click_lat)**2 + (filtered["longitude"] - click_lon)**2
            nearest = filtered.sort_values("__dist").iloc[0]
            selected_row = nearest
            # cleanup
            filtered.drop(columns="__dist", inplace=True, errors='ignore')

    if selected_row is not None:
        st.markdown("### Détails de la facility")
        # show key fields
        st.write(f"**Facility name:** {selected_row.get('facility_name','')}")
        st.write(f"**Company:** {selected_row.get('company_name','')}")
        st.write(f"**Province:** {selected_row.get('province','')}")
        st.write(f"**Commodities:** {selected_row.get('commodities','')}")
        st.write(f"**Mining / processing type:** {selected_row.get('mining_processing_type','')}")
        st.write(f"**Status:** {selected_row.get('status','')}")
        st.write(f"**main_id:** {selected_row.get('main_id','')}")
        # show links if present
        if selected_row.get("company_URL"):
            st.write(f"[Company URL]({selected_row.get('company_URL')})")
        if selected_row.get("facility_URL"):
            st.write(f"[Facility page]({selected_row.get('facility_URL')})")
    else:
        st.info("Sélectionne une facility dans la liste ou clique sur une icône sur la carte pour voir les détails.")

    st.write("---")
    st.download_button("Exporter les données filtrées (CSV)", data=filtered.to_csv(index=False).encode("utf-8-sig"),
                       file_name="main_filtered.csv", mime="text/csv")

# ------- FOOTER: instructions pour integrer d'autres tables -------
st.write("")
st.markdown(
    """
    **Notes / next steps**
    - Si tu as d'autres CSV (Production, Environmental_flows, Intensity, etc.) dans `database/CSV/`, on peut charger ces tables et afficher les entrées liées via `main_id` dans le panneau de détails.
    - Pour une intégration interactive (cliquer sur une facility → afficher ses données liés), nous joindrons les tables par `main_id` et afficherons graphiques (plotly) / tableaux.
    - Si le rendu Folium ne capte pas le clic dans ta version, la sélection via la liste déroulante reste disponible.
    """
)
