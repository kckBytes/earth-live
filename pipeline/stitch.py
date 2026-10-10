"""Global cloud map from five geostationary satellites (one time slot).

Physics, in order:
  1. Infrared (10.3-10.8 um) from every satellite -> brightness temperature BT (deg C), blended by viewing angle.
  2. Clear-sky reference temperature, i.e. what the ground/sea would read with no cloud:
       land: local warm envelope, computed on terrain-corrected temperature (BT + 5.5 K/km * height),
             so high cold plateaus (Tibet, Andes, Rockies) are judged against their own expected temperature;
       sea:  measured sea-surface temperature minus the atmosphere's own effect (estimated from clear
             patches nearby), so a low warm cloud deck only a few degrees colder than the sea still shows.
     IR cloudiness = how far BT falls below that reference.
  3. Visible red (0.6-0.64 um) in daylight -> reflectance above the known surface brightness (Blue Marble),
     corrected for sun angle, with sunglint and terminator excluded. Catches low clouds the IR can't.
  4. cloud = max(IR, visible); poles fade to the 3-hourly composite.
usage: python stitch.py <slot dir> <YYYY-MM-DDTHH:MMZ> <day texture> <polar-fill global map> <out prefix> [static dir]
"""
import sys, glob, re, math
import numpy as np
from PIL import Image
from scipy import ndimage

D, SLOT, DAYTEX, OLDMAP, OUT = sys.argv[1:6]
SD = sys.argv[6] if len(sys.argv) > 6 else D        # where sst/, dem/ and the colour maps live
W, H = 5120, 2560
lon = (np.arange(W) + 0.5) / W * 360 - 180
lat = 90 - (np.arange(H) + 0.5) / H * 180
LON, LAT = np.meshgrid(np.radians(lon).astype(np.float32), np.radians(lat).astype(np.float32))
lonA, latA = np.degrees(LON), np.degrees(LAT)

def log(*a): print(*a, flush=True)

# ------------------------------------------------------------------ colour-map inversion (GIBS)
def load_cmap(path):
    x = open(path, encoding='utf-8').read()
    keys, vals = [], []
    for ent in re.findall(r'<ColorMapEntry([^>]*)/>', x):
        if 'nodata="true"' in ent or 'transparent="true"' in ent: continue
        rgb = re.search(r'rgb="(\d+),(\d+),(\d+)"', ent); val = re.search(r'\bvalue="([^"]+)"', ent)
        if not rgb or not val: continue
        nums = [float(v) for v in re.findall(r'-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?', val.group(1))]
        if not nums: continue
        r, g, b = map(int, rgb.groups())
        keys.append((r << 16) | (g << 8) | b); vals.append(sum(nums) / len(nums))
    keys = np.array(keys, np.int64); vals = np.array(vals, np.float32)
    o = np.argsort(keys); keys, vals = keys[o], vals[o]
    pal = np.stack([(keys >> 16) & 255, (keys >> 8) & 255, keys & 255], 1).astype(np.float32)
    return keys, vals, pal

def mosaic(name, base=None):
    img = np.zeros((H, W, 4), np.uint8)
    for f in glob.glob(f'{base or D}/{name}/*.png'):
        r, c = map(int, re.findall(r'(\d+)_(\d+)\.png', f)[0])
        try:
            t = np.array(Image.open(f).convert('RGBA'))
        except Exception:                                # an error page instead of a tile: treat as no data
            continue
        y0, x0 = r * 512, c * 512
        hh, ww = min(512, H - y0), min(512, W - x0)
        if hh > 0 and ww > 0: img[y0:y0 + hh, x0:x0 + ww] = t[:hh, :ww]
    if base is None: img[dropouts(img), 3] = 0               # satellite layers only (not the static sst / dem)
    return img

def dropouts(img, run=256):
    """Scan-line dropouts in GIBS tiles: long horizontal runs (18+ degrees) of one identical colour. Real imagery
    always varies; a missing swath arrives as a flat band (GOES sends pure white, which the IR scale reads as -92 C).
    Not used on the EUMETSAT images, which are upsampled and legitimately flat in places."""
    ok = img[..., 3] > 0
    p = (img[..., 0].astype(np.int32) << 16) | (img[..., 1].astype(np.int32) << 8) | img[..., 2]
    eq = np.zeros(ok.shape, bool); eq[:, 1:] = (p[:, 1:] == p[:, :-1]) & ok[:, 1:] & ok[:, :-1]
    lab, n = ndimage.label(eq, structure=[[0, 0, 0], [1, 1, 1], [0, 0, 0]])
    if n == 0: return np.zeros(ok.shape, bool)
    big = np.bincount(lab.ravel()) >= run; big[0] = False
    m = big[lab]; m[:, :-1] |= m[:, 1:]
    if m.any(): log(f'   dropout band(s) masked: {m.sum()} px')
    return m

def invert(img, cm, nearest=True):
    keys, vals, pal = cm
    ok = img[..., 3] > 0
    packed = (img[..., 0].astype(np.int64) << 16) | (img[..., 1].astype(np.int64) << 8) | img[..., 2]
    out = np.full((H, W), np.nan, np.float32)
    p = packed[ok]
    idx = np.clip(np.searchsorted(keys, p), 0, len(keys) - 1)
    exact = keys[idx] == p
    v = np.where(exact, vals[idx], np.nan).astype(np.float32)
    if nearest and (~exact).any():
        rgb = img[..., :3][ok][~exact].astype(np.float32)
        best = np.empty(len(rgb), np.int64)
        for i in range(0, len(rgb), 100000):
            d = ((rgb[i:i + 100000, None, :] - pal[None]) ** 2).sum(-1); best[i:i + 100000] = d.argmin(1)
        v[~exact] = vals[best]
    out[ok] = v
    return out, exact.mean() if len(p) else 0

def grey(name):
    import os
    if not os.path.exists(f'{D}/{name}.png'):
        return np.full((H, W), np.nan, np.float32)
    try:
        a = np.array(Image.open(f'{D}/{name}.png').convert('RGBA'))
    except Exception:                                    # e.g. a WMS error document for a missing time
        log(f'{name}: unreadable, skipped')
        return np.full((H, W), np.nan, np.float32)
    g = np.full((H, W), np.nan, np.float32); ok = a[..., 3] > 0; g[ok] = a[..., 0][ok]
    return g

def fit_curve(x_img, ref, region, step=8, decreasing=True):
    m = region & np.isfinite(x_img) & np.isfinite(ref)
    if m.sum() < 5000:                                  # no overlap to calibrate against: generic curve
        log('   not enough overlap, using the generic grey curve')
        return np.where(np.isfinite(x_img), 25.0 - 0.40 * x_img, np.nan).astype(np.float32)
    x, y = x_img[m], ref[m]
    cs, ms = [], []
    for lo in range(0, 256, step):
        sel = (x >= lo) & (x < lo + step)
        if sel.sum() > 300: cs.append(lo + (step - 1) / 2); ms.append(np.median(y[sel]))
    ms = np.minimum.accumulate(ms) if decreasing else np.maximum.accumulate(ms)
    out = np.interp(x_img, cs, ms).astype(np.float32)
    out[~np.isfinite(x_img)] = np.nan
    fin = m & np.isfinite(out)
    log(f'   curve from {m.sum()} px, median abs err {np.median(np.abs(out[fin] - ref[fin])):.2f}')
    return out

# ------------------------------------------------------------------ geometry
SUBLON = {'goes_e': -75.2, 'goes_w': -137.2, 'hima': 140.7, 'mtg': 0.0, 'iodc': 45.5}
def view_weight(k):
    cosg = np.cos(LAT) * np.cos(LON - np.radians(SUBLON[k]))
    r = 6.6227; d = np.sqrt(1 + r * r - 2 * r * cosg); cosz = (r * cosg - 1) / d
    return np.clip(cosz - 0.12, 0, None) ** 4, cosz

def sun_vector(slot):
    from datetime import datetime, timezone
    t = datetime.strptime(slot, '%Y-%m-%dT%H:%MZ').replace(tzinfo=timezone.utc).timestamp() * 1000
    d = t / 86400000 + 2440587.5 - 2451545.0
    g = math.radians(357.529 + 0.98560028 * d); q = 280.459 + 0.98564736 * d
    L = math.radians(q + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g)); e = math.radians(23.439 - 0.00000036 * d)
    ra = math.atan2(math.cos(e) * math.sin(L), math.cos(L)); dec = math.asin(math.sin(e) * math.sin(L))
    gmst = math.radians(280.46061837 + 360.98564736629 * d) % (2 * math.pi)
    lo = ra - gmst
    return np.array([math.cos(dec) * math.cos(lo), math.cos(dec) * math.sin(lo), math.sin(dec)], np.float32)

N = np.stack([np.cos(LAT) * np.cos(LON), np.cos(LAT) * np.sin(LON), np.sin(LAT)], -1)
S = sun_vector(SLOT)
MU0 = (N @ S).astype(np.float32)                         # cos solar zenith

# ------------------------------------------------------------------ 1. infrared -> BT, blended
cm_ir = load_cmap(f'{SD}/cmap_ir.xml')
bt = {}
def fill_specks(v, holes, max_px=64):
    """Fill small no-data specks with the nearest valid value; large holes stay empty (other satellites cover)."""
    lab, n = ndimage.label(holes)
    if n == 0: return v
    small = np.bincount(lab.ravel()) <= max_px; small[0] = False
    sm = small[lab]
    if not sm.any(): return v
    _, (iy, ix) = ndimage.distance_transform_edt(~np.isfinite(v), return_indices=True)
    out = v.copy(); out[sm] = v[iy[sm], ix[sm]]
    return out
def despike(v, size=5, tol=20.0):
    """The GIBS tiles are downsampled with blending, which creates in-between colours; some of those happen to be
    exact palette entries for a very different temperature (a blended warm grey that reads as -75 C). Such pixels
    disagree with nearly all their neighbours: replace anything more than tol K from its 5x5 median."""
    fin = np.isfinite(v)
    med = ndimage.median_filter(np.where(fin, v, np.nanmedian(v)), size=size)
    bad = fin & (np.abs(v - med) > tol)
    out = v.copy(); out[bad] = med[bad]
    return out, int(bad.sum())
for k in ['goes_e', 'goes_w', 'hima']:
    m = mosaic(k)
    white = (m[..., 3] > 0) & (m[..., 0] == 255) & (m[..., 1] == 255) & (m[..., 2] == 255)
    m[white, 3] = 0                                      # GIBS draws missing pixels pure white (= -92 C on the scale)
    bt[k], ex = invert(m, cm_ir)
    bt[k] = fill_specks(bt[k], white)
    bt[k], nsp = despike(bt[k])
    if nsp: log(f'   {k}: {nsp} colour-blend spikes repaired')
    log(f'{k}: IR, exact colour {ex*100:.0f}%, white no-data px {white.sum()}')
bt['mtg'] = fit_curve(grey('mtg'), bt['goes_e'], (lonA > -45) & (lonA < -15) & (np.abs(latA) < 50))
bt['iodc'] = fit_curve(grey('iodc'), bt['mtg'], (lonA > 20) & (lonA < 35) & (np.abs(latA) < 45))
num = np.zeros((H, W), np.float32); den = np.zeros((H, W), np.float32)
COSZ = {}
for k, v in bt.items():
    w, cz = view_weight(k); COSZ[k] = (w, cz)
    w = np.where(np.isfinite(v), w, 0)
    num += w * np.nan_to_num(v); den += w
BT = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan).astype(np.float32)
log(f'IR coverage {np.isfinite(BT).mean()*100:.0f}%')

# ------------------------------------------------------------------ 2. clear-sky reference
dem, _ = invert(mosaic('dem', SD), load_cmap(f'{SD}/cmap_dem.xml'), nearest=False)
Z = np.nan_to_num(dem, nan=0.0).clip(0, 8000)
sst, _ = invert(mosaic('sst', SD), load_cmap(f'{SD}/cmap_sst.xml'))
SEA_ANY = np.isfinite(sst)
# The SST analysis also covers lakes, reservoirs and rivers, and its 0.25-degree cells spill over coasts. Over
# such pixels the satellite sees land (cold at night, bright by day), so judging them against water made
# speckles of fake cloud all over the continents. Use the sea reference only on large water bodies (the Great
# Lakes and up), away from the shore; everything else is judged as land.
lab, n = ndimage.label(SEA_ANY)
big = np.bincount(lab.ravel()) >= 400; big[0] = False
SEA = ndimage.binary_erosion(big[lab], iterations=2)
del lab
log(f'elevation max {Z.max():.0f} m; sea-surface temperature on {SEA.mean()*100:.0f}% of grid')
LAPSE = 0.0055                                           # K per m (surface skin, mid-day/night average)
BTa = BT + LAPSE * Z                                     # "as if at sea level"
cs = 16
blocks = np.where(np.isfinite(BTa), BTa, -999).reshape(H // cs, cs, W // cs, cs)
warm = np.percentile(blocks, 95, axis=(1, 3))
warm[warm < -200] = np.nan
env = ndimage.maximum_filter(np.nan_to_num(warm, nan=-999), size=(3, 5), mode=('nearest', 'wrap'))
env[env < -200] = np.nanmean(warm)
env = ndimage.uniform_filter(env, size=(3, 5), mode=('nearest', 'wrap'))
ENV = np.array(Image.fromarray(env.astype(np.float32)).resize((W, H), Image.BILINEAR)) - LAPSE * Z
# sea: SST minus the clear atmosphere's water-vapour cooling. Its size depends mostly on humidity, i.e. on
# latitude: take the clearest 2% of sea pixels in each 5-degree latitude band (thousands of clear pixels
# per band even on a cloudy day), then let it vary locally only within +-1.5 K of that.
diff = np.where(SEA & np.isfinite(BT), sst - BT, np.nan)
band_lat = np.arange(-90, 90, 5)
dlat = []
for b0 in band_lat:
    sel = np.isfinite(diff) & (latA >= b0) & (latA < b0 + 5)
    dlat.append(np.percentile(diff[sel], 2) if sel.sum() > 5000 else np.nan)
dlat = np.array(dlat); good = np.isfinite(dlat)
dlat = np.interp(band_lat, band_lat[good], dlat[good]) if good.any() else np.full(len(band_lat), 4.0)
DLAT = np.interp(latA[:, 0], band_lat + 2.5, dlat).astype(np.float32)[:, None] * np.ones((1, W), np.float32)
bs = 128
db = diff.reshape(H // bs, bs, W // bs, bs).transpose(0, 2, 1, 3).reshape(H // bs, W // bs, -1)
cnt = np.isfinite(db).sum(-1)
with np.errstate(all='ignore'):
    loc = np.where(cnt > 500, np.nanpercentile(np.where(cnt[..., None] > 500, db, 0), 2, axis=-1), np.nan)
locf = np.array(Image.fromarray(np.nan_to_num(loc, nan=-99).astype(np.float32)).resize((W, H), Image.NEAREST))
# warm tropical seas carry much more water vapour than their latitude band's average: allow the local
# estimate to go higher there; low cloud decks live over cool water, where it stays tight.
up = np.where(np.nan_to_num(sst, nan=0) > 25.5, 3.5, 1.0)
dl = np.where(locf > -50, np.clip(locf, DLAT - 1.5, DLAT + up), DLAT + np.where(up > 1, 1.0, 0.0))
dl = ndimage.uniform_filter(dl[::16, ::16], size=5, mode='wrap')
fillv = float(np.median(dlat))
VAP = np.array(Image.fromarray(dl.astype(np.float32)).resize((W, H), Image.BILINEAR))
TSEA = sst - VAP
TCLEAR = np.where(SEA, np.fmax(TSEA, ENV), ENV)
TCLEAR = np.fmax(TCLEAR, BT)
deficit = TCLEAR - BT
margin = np.where(SEA, 3.0, 6.0)                        # sea reference is far more precise than the land envelope
ir = np.clip((deficit - margin) / 40.0, 0, 1)
ir = np.maximum(ir, np.clip((-25 - BT) / 35, 0, 1))
ir = ndimage.median_filter(np.nan_to_num(ir), size=3)

log(f'water-vapour offset over sea by latitude: ' + ' '.join(f'{b}:{v:.1f}' for b, v in zip(band_lat[::3], dlat[::3])))

# ------------------------------------------------------------------ 3. visible (daylight only)
def vis_gibs(k):
    a = mosaic(k)
    g = np.full((H, W), np.nan, np.float32); ok = a[..., 3] > 0; g[ok] = a[..., 0][ok] / 255.0
    return g
vis = {'goes_e': vis_gibs('goes_e_vis'), 'goes_w': vis_gibs('goes_w_vis'), 'hima': vis_gibs('hima_vis'),
       'mtg': grey('mtg_vis') / 255.0, 'iodc': grey('iodc_vis') / 255.0}
day = np.array(Image.open(DAYTEX).convert('RGB').resize((W, H), Image.BILINEAR), np.float32) / 255.0
surf_lin = (day[..., 0] ** 2.2) * 0.9 + 0.02             # Blue Marble red, linear, as a surface-reflectance guess
# sunglint: sun reflected toward each satellite off a flat sea
def glint_mask(k):
    sat = np.array([math.cos(math.radians(SUBLON[k])), math.sin(math.radians(SUBLON[k])), 0], np.float32) * 6.6227
    v = sat[None, None, :] - N; v /= np.linalg.norm(v, axis=-1, keepdims=True)
    h = v + S[None, None, :]; h /= np.linalg.norm(h, axis=-1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip((h * N).sum(-1), -1, 1)))
    return np.clip((ang - 10) / 15, 0, 1)                # 0 near the specular point, 1 beyond 25 deg
vnum = np.zeros((H, W), np.float32); vden = np.zeros((H, W), np.float32)
for k, v in vis.items():
    if not np.isfinite(v).any(): log(f'{k}: no visible image (night side)'); continue
    # each sensor's grey scale differs: map to a common scale by matching the clear-sea / bright-cloud levels
    sun_ok = (MU0 > 0.25) & np.isfinite(v)
    w = COSZ[k][0] * sun_ok * (glint_mask(k) if True else 1)
    lin = v ** 2.2                                       # undo display gamma (approximate)
    refl = lin / np.maximum(MU0, 0.25)
    sea_clear = sun_ok & SEA & (ir < 0.02)
    hi_cloud = sun_ok & (ir > 0.85)
    lo_l = np.percentile(refl[sea_clear], 50) if sea_clear.sum() > 2000 else None
    hi_l = np.percentile(refl[hi_cloud], 70) if hi_cloud.sum() > 2000 else None
    if lo_l is None or hi_l is None or hi_l < lo_l * 2 + 0.02:
        log(f'{k}: too little daylight in view, visible skipped'); continue
    if True:
        refl = (refl - lo_l) / max(hi_l - lo_l, 1e-3) * 0.75 + 0.05
        log(f'{k}: visible normalised (clear sea {lo_l:.3f}, thick cloud {hi_l:.3f})')
    vnum += w * np.nan_to_num(refl); vden += w
VIS = np.where(vden > 1e-4, vnum / np.maximum(vden, 1e-9), np.nan)
vcl = np.clip((VIS - np.where(SEA, 0.05, surf_lin * 1.15 + 0.05)) / 0.45, 0, 1)
vcl = ndimage.median_filter(np.nan_to_num(vcl), size=3)
dayw = np.clip((MU0 - 0.25) / 0.2, 0, 1) * np.clip(vden / 1e-3, 0, 1)
# don't let bright snow/ice or deserts masquerade as cloud: require at least slight IR coolness over land
vcl = np.where(SEA, vcl, vcl * np.clip((deficit - 1.0) / 4.0, 0, 1))
vlo = vcl * dayw
log(f'visible used on {(dayw > 0.5).mean()*100:.0f}% of the grid; low-cloud pixels added by visible: {((vlo > 0.3) & (ir < 0.1)).mean()*100:.1f}%')

# ------------------------------------------------------------------ 4. encode, fill the poles
def enc(i):
    """cloud index 0..1 -> stored grey (0.38 clear, 0.40..0.96 cloud); drops the faint residue below 0.08"""
    i = np.clip((i - 0.08) / 0.92, 0, 1)
    return np.where(i > 0, 0.40 + 0.56 * i ** 0.7, 0.38).astype(np.float32)
old = np.array(Image.open(OLDMAP).convert('L').resize((W, H), Image.BILINEAR), np.float32) / 255
wfill = np.clip(den / 1e-2, 0, 1)
def fill(raw):
    raw = np.where(np.isfinite(BT), raw, np.nan)
    return np.maximum(np.where(np.isfinite(raw), raw, old) * wfill + old * (1 - wfill), 0.38)   # never darker than clear sky
def L8(a): return Image.fromarray((np.clip(np.nan_to_num(a), 0, 1) * 255).astype(np.uint8))
img = L8(fill(enc(np.maximum(ir, vlo))))                 # what the satellites see now (live map)
img.resize((4096, 2048), Image.LANCZOS).save(f'{OUT}_4096.jpg', quality=90)
img.resize((2048, 1024), Image.LANCZOS).save(f'{OUT}_2048.jpg', quality=90)
# Layers for the replay, which carries daylight-only low cloud through the night (run.py):
#   _ir: infrared-only map (same at day and night, so it is also what motion is measured on)
#   _lo: low cloud seen only in visible light (valid where _dw, the daylight weight, is high)
iimg = L8(fill(enc(ir)))
iimg.resize((4096, 2048), Image.LANCZOS).save(f'{OUT}_ir_4096.jpg', quality=90)
iimg.resize((2048, 1024), Image.LANCZOS).save(f'{OUT}_ir_2048.jpg', quality=92)
iimg.resize((512, 256), Image.BOX).save(f'{OUT}_512.png')
lo = np.where(vlo > ir, vcl, 0) * wfill
L8(lo).resize((4096, 2048), Image.BILINEAR).save(f'{OUT}_lo_4096.png', optimize=True)
L8(lo).resize((2048, 1024), Image.BILINEAR).save(f'{OUT}_lo_2048.png', optimize=True)
L8(dayw * wfill).resize((512, 256), Image.BOX).save(f'{OUT}_dw_512.png')
import os
if not os.path.exists(f'{SD}/sea_2048.png'):
    L8(SEA.astype(np.float32)).resize((2048, 1024), Image.BOX).save(f'{SD}/sea_2048.png')
# data layers for the other globe views
# infrared: cloud-top / surface temperature, 0 = no data, 1..255 = -90..+50 C
bt8 = np.where(np.isfinite(BT), 1 + np.clip((BT + 90) / 140 * 254, 0, 254), 0).astype(np.uint8)
bimg = Image.fromarray(bt8)
bimg.resize((4096, 2048), Image.BILINEAR).save(f'{OUT}_bt_4096.jpg', quality=90)
bimg.resize((2048, 1024), Image.BILINEAR).save(f'{OUT}_bt_2048.jpg', quality=90)
# sea-surface temperature: 0 = land/ice/no data, 1..255 = -2..+35 C
sst8 = np.where(SEA_ANY, 1 + np.clip((sst + 2) / 37 * 254, 0, 254), 0).astype(np.uint8)
Image.fromarray(sst8).resize((2048, 1024), Image.NEAREST).save(f'{OUT}_sst_2048.png', optimize=True)
# diagnostics
def q(a): return Image.fromarray((np.clip(np.nan_to_num(a), 0, 1) * 255).astype(np.uint8)).resize((1280, 640))
q(ir).save(f'{OUT}_dbg_ir.png'); q(vlo).save(f'{OUT}_dbg_vis.png')
log('saved', f'{OUT}_4096.jpg')
