"""
Fiber-optic cable geometry: KMZ vertices -> DAS channels -> 50-channel roadway segments.

    load_channels()          one point per DAS channel (EPSG:4326), with channel numbers
    segment_lines()          LineString per roadway segment (``seg_id`` matches traffic tables)
    export_segments()        segments joined with a value column -> GeoJSON (for GIS / maps)
"""
import re
import zipfile

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString

import config


def parse_kmz_lonlat(kmz_path=config.FIBER_KMZ):
    """All lon/lat pairs in a KMZ (supports one or many points per <coordinates> block)."""
    with zipfile.ZipFile(kmz_path) as zf:
        names = zf.namelist()
        kml_name = "doc.kml" if "doc.kml" in names else next((n for n in names if n.lower().endswith(".kml")), None)
        if kml_name is None:
            raise FileNotFoundError("No .kml found inside KMZ.")
        kml = zf.read(kml_name).decode("utf-8", errors="ignore")

    lons, lats = [], []
    for block in re.findall(r"<coordinates>\s*([^<]+?)\s*</coordinates>", kml, flags=re.IGNORECASE):
        for token in block.split():
            vals = token.split(",")
            try:
                lons.append(float(vals[0]))
                lats.append(float(vals[1]))
            except (ValueError, IndexError):
                continue
    if len(lons) < 2:
        raise ValueError("Parsed too few coordinates from the KMZ.")
    return np.array(lons), np.array(lats)


def load_channels(kmz_path=config.FIBER_KMZ, first_channel=config.KMZ_FIRST_CHANNEL):
    lon, lat = parse_kmz_lonlat(kmz_path)
    gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon, lat), crs="EPSG:4326")
    gdf["channel"] = np.arange(len(gdf)) + first_channel
    return gdf


def fiber_line(channels):
    """The whole fiber as a single LineString."""
    return gpd.GeoDataFrame(geometry=[LineString(zip(channels.geometry.x, channels.geometry.y))], crs=channels.crs)


def segment_lines(channels, start_ch=config.SEG_START_CH, end_ch=config.SEG_END_CH, size=config.SEG_SIZE):
    """Connect the channels of each ``size``-channel bin in [start_ch, end_ch] into a LineString."""
    sel = channels[(channels["channel"] >= start_ch) & (channels["channel"] <= end_ch)].sort_values("channel").copy()
    sel["seg_id"] = ((sel["channel"] - start_ch) // size).astype(int)
    rows = [dict(seg_id=int(sid), ch0=int(g["channel"].iloc[0]), ch1=int(g["channel"].iloc[-1]),
                 geometry=LineString(zip(g.geometry.x, g.geometry.y)))
            for sid, g in sel.groupby("seg_id", sort=True) if len(g) >= 2]
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=channels.crs)


def export_segments(values, value_cols, out_path, seg_col="seg"):
    """Join per-segment values (DataFrame with ``seg_col``) to segment geometry and write GeoJSON."""
    segs = segment_lines(load_channels())
    out = segs.merge(values.rename(columns={seg_col: "seg_id"})[["seg_id", *value_cols]], on="seg_id", how="left")
    out.to_file(out_path, driver="GeoJSON")
    return out
