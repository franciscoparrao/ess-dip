"""
Fetch a small Sentinel-2 L2A scene + co-registered ESA WorldCover labels from
Microsoft Planetary Computer (STAC). Saves aligned arrays to data/ so the
analysis (exp 12) does not need to refetch.

Bands: the multispectral set commonly used for land-cover classification
(10 m + 20 m, resampled to 10 m): B02 B03 B04 B05 B06 B07 B08 B8A B11 B12.
Ground truth: ESA WorldCover 2021 (10 m, 11 classes).
"""
import numpy as np
import planetary_computer as pc
from pystac_client import Client
import stackstac
import rioxarray  # noqa: F401  (enables .rio accessor)

# small AOI in the Po Valley (agriculture + urban + water), ~2.5 x 2.5 km
BBOX = [9.150, 45.300, 9.180, 45.322]
BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
DATERANGE = "2021-06-01/2021-09-15"

cat = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                  modifier=pc.sign_inplace)

def get_epsg(item):
    p = item.properties
    for key in ("proj:epsg", "proj:code"):
        v = p.get(key)
        if v:
            return int(str(v).split(":")[-1])
    for a in item.assets.values():
        for key in ("proj:epsg", "proj:code"):
            v = a.extra_fields.get(key)
            if v:
                return int(str(v).split(":")[-1])
    return None


print("querying Sentinel-2 L2A ...", flush=True)
s2 = cat.search(collections=["sentinel-2-l2a"], bbox=BBOX, datetime=DATERANGE,
                query={"eo:cloud_cover": {"lt": 5}}).item_collection()
print(f"  {len(s2)} items; picking least-cloudy", flush=True)
item = min(s2, key=lambda i: i.properties["eo:cloud_cover"])
epsg = get_epsg(item)
print(f"  {item.id}  cloud={item.properties['eo:cloud_cover']:.1f}%  "
      f"epsg={epsg}", flush=True)

cube = (stackstac.stack(item, assets=BANDS, resolution=10,
                        bounds_latlon=BBOX, epsg=epsg)
        .squeeze())
cube = cube.transpose("y", "x", "band")
cube.rio.write_crs(epsg, inplace=True)
arr = cube.compute().values.astype(np.float32)
print(f"  S2 cube -> {arr.shape}", flush=True)

print("querying ESA WorldCover ...", flush=True)
wc_item = cat.search(collections=["esa-worldcover"], bbox=BBOX).item_collection()[0]
wc = rioxarray.open_rasterio(pc.sign(wc_item.assets["map"].href)).squeeze()
wc_match = wc.rio.reproject_match(cube)            # align to S2 grid
labels = wc_match.values.astype(np.int16)
print(f"  WorldCover -> {labels.shape}", flush=True)

# drop any rows/cols with NaN in S2 (edge) by simple nan->0 mask report
valid = np.isfinite(arr).all(-1)
print(f"  valid pixels: {valid.mean():.3f}")
vals, cnt = np.unique(labels[valid], return_counts=True)
WC = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built",
      60: "bare", 70: "snow", 80: "water", 90: "wetland", 95: "mangrove",
      100: "moss"}
print("  WorldCover class histogram:")
for v, c in sorted(zip(vals.tolist(), cnt.tolist()), key=lambda t: -t[1]):
    print(f"    {v:>3} {WC.get(v,'?'):8s} {c:>7} ({100*c/valid.sum():.1f}%)")

np.save("data/s2_cube.npy", arr)
np.save("data/s2_labels.npy", labels)
print("saved -> data/s2_cube.npy, data/s2_labels.npy")
