"""Ventusky-style weather layers from NOAA's free GFS model (0.5 degree) and its wave model, every 3 hours from
24 h ago to 12 h ahead. Fetched only when a new model run is out (4 a day); kept in the work dir between runs.

Published per time t (ms):  wx/<t>_a.png  R temperature 2 m, G feels-like, B relative humidity
                            wx/<t>_b.png  R precipitation rate, G frozen share, B thunderstorm energy (CAPE)
                            wx/<t>_c.png  R sea-level pressure, G snow depth, B significant wave height
                            wx/<t>_w.png  R wind east, G wind north (10 m), B gusts
 all 720 x 360, equirectangular, lon -180 at the left, 8-bit (scales below, mirrored in the app)
and wx.json listing the times.
"""
import os, json, math, glob, urllib.request
from datetime import datetime, timedelta, timezone
import numpy as np
from PIL import Image

UA = {'User-Agent': 'earth-live (https://github.com/kckBytes/earth-live)'}
ATM = ('https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p50.pl?dir=%2Fgfs.{d}%2F{c:02d}%2Fatmos&file=gfs.t{c:02d}z.pgrb2full.0p50.f{f:03d}'
       '&var_TMP=on&var_RH=on&lev_2_m_above_ground=on&var_UGRD=on&var_VGRD=on&lev_10_m_above_ground=on'
       '&var_GUST=on&var_PRATE=on&var_CAPE=on&var_SNOD=on&var_CPOFP=on&lev_surface=on&var_PRMSL=on&lev_mean_sea_level=on')
WAVE = ('https://nomads.ncep.noaa.gov/cgi-bin/filter_gfswave.pl?dir=%2Fgfs.{d}%2F{c:02d}%2Fwave%2Fgridded'
        '&file=gfswave.t{c:02d}z.global.0p25.f{f:03d}.grib2&var_HTSGW=on&lev_surface=on')
SCALE = {  # value range mapped to 0..255 (the app uses the same table)
    'temp': (-60, 50), 'feels': (-60, 50), 'rh': (0, 100), 'precip_log1p': (0, math.log1p(50)), 'frozen': (0, 100),
    'cape_sqrt': (0, math.sqrt(5000)), 'pressure': (950, 1060), 'snow_sqrt': (0, math.sqrt(3)), 'waves': (0, 15),
    'wind': (-50, 50), 'gust': (0, 60)}
H = 3600_000

def ms(t): return int(t.timestamp() * 1000)

def raw(url, timeout=120):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read()

def messages(blob):
    import eccodes
    out, off = {}, 0
    while True:
        i = blob.find(b'GRIB', off)
        if i < 0: break
        n = int.from_bytes(blob[i + 8:i + 16], 'big')                # GRIB2 total length
        g = eccodes.codes_new_from_message(blob[i:i + n])
        try:
            key = (eccodes.codes_get(g, 'shortName'), eccodes.codes_get(g, 'typeOfLevel'))
            nj, ni = eccodes.codes_get(g, 'Nj'), eccodes.codes_get(g, 'Ni')
            v = eccodes.codes_get_values(g).reshape(nj, ni)
            v[v > 9e20] = np.nan
            out[key] = v
        finally: eccodes.codes_release(g)
        off = i + n
    return out

def to_map(v, w=720, h=360):
    """model grid (lat 90..-90, lon 0..360) -> w x h, lon -180 at the left"""
    v = np.roll(v[:-1] if v.shape[0] % 2 else v, v.shape[1] // 2, axis=1)       # drop the extra pole row, centre on 0
    return np.asarray(Image.fromarray(v.astype(np.float32)).resize((w, h), Image.BILINEAR))

def enc(v, key):
    lo, hi = SCALE[key]
    return np.clip(np.round((np.nan_to_num(v, nan=lo) - lo) / (hi - lo) * 255), 0, 255).astype(np.uint8)

def feels_like(tc, rh, wind):
    """NWS heat index above 26.7 C, wind chill below 10 C with wind, else the air temperature."""
    tf = tc * 9 / 5 + 32
    hi = (-42.379 + 2.04901523 * tf + 10.14333127 * rh - .22475541 * tf * rh - .00683783 * tf * tf
          - .05481717 * rh * rh + .00122874 * tf * tf * rh + .00085282 * tf * rh * rh - .00000199 * tf * tf * rh * rh)
    hi_c = (hi - 32) * 5 / 9
    kmh = wind * 3.6
    wc = 13.12 + 0.6215 * tc - 11.37 * np.power(np.maximum(kmh, 0.1), 0.16) + 0.3965 * tc * np.power(np.maximum(kmh, 0.1), 0.16)
    return np.where((tc >= 26.7) & (rh >= 40), np.maximum(hi_c, tc), np.where((tc <= 10) & (kmh > 4.8), np.minimum(wc, tc), tc))

def newest_cycle(now):
    c = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=now.hour % 6)
    for k in range(5):
        cc = c - timedelta(hours=6 * k)
        try:
            raw(ATM.format(d=cc.strftime('%Y%m%d'), c=cc.hour, f=15).split('&var_')[0] + '&var_PRMSL=on&lev_mean_sea_level=on', 60)
            return cc
        except Exception: continue
    return None

def plan(now, newest):
    """time -> (cycle, forecast hour): past times from each cycle's +3/+6 h, the future from the newest run"""
    t0 = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=now.hour % 3)
    out = {}
    for k in range(-9, 5):
        t = t0 + timedelta(hours=3 * k)
        c = t - timedelta(hours=3)
        c = c - timedelta(hours=c.hour % 6)
        if c > newest: c = newest
        f = int((t - c).total_seconds() // 3600)
        if 3 <= f <= 24: out[t] = (c, f)
    return out

def build(work, site, now=None):
    now = now or datetime.now(timezone.utc)
    wd = f'{work}/wx'; os.makedirs(wd, exist_ok=True)
    idx_p = f'{wd}/index.json'
    idx = json.load(open(idx_p)) if os.path.exists(idx_p) else {}
    newest = newest_cycle(now)
    if newest is None:
        print('weather: no GFS run reachable'); return
    todo = {t: cf for t, cf in plan(now, newest).items() if idx.get(str(ms(t))) != f'{ms(cf[0])}+{cf[1]}'}
    for t, (c, f) in sorted(todo.items()):
        try:
            m = messages(raw(ATM.format(d=c.strftime('%Y%m%d'), c=c.hour, f=f)))
            g = lambda n, lev: m.get((n, lev))
            tc = to_map(g('2t', 'heightAboveGround') if g('2t', 'heightAboveGround') is not None else g('t', 'heightAboveGround')) - 273.15
            rh = to_map(g('2r', 'heightAboveGround') if g('2r', 'heightAboveGround') is not None else g('r', 'heightAboveGround'))
            u = to_map(g('10u', 'heightAboveGround')); v = to_map(g('10v', 'heightAboveGround'))
            gust = to_map(g('gust', 'surface')); prate = to_map(g('prate', 'surface')) * 3600
            cape = to_map(g('cape', 'surface')); snod = to_map(g('sde', 'surface') if g('sde', 'surface') is not None else g('snod', 'surface'))
            frz = to_map(g('cpofp', 'surface')); pres = to_map(g('prmsl', 'meanSea')) / 100
            try: waves = to_map(messages(raw(WAVE.format(d=c.strftime('%Y%m%d'), c=c.hour, f=f))).get(('swh', 'surface')))
            except Exception: waves = np.zeros_like(tc)
            fl = feels_like(tc, rh, np.hypot(u, v))
            frz = np.where(prate > 0.05, np.clip(frz, 0, 100), 0)
            n = ms(t)
            Image.fromarray(np.stack([enc(tc, 'temp'), enc(fl, 'feels'), enc(rh, 'rh')], -1)).save(f'{wd}/{n}_a.png', optimize=True)
            Image.fromarray(np.stack([enc(np.log1p(np.maximum(prate, 0)), 'precip_log1p'), enc(frz, 'frozen'),
                                      enc(np.sqrt(np.maximum(cape, 0)), 'cape_sqrt')], -1)).save(f'{wd}/{n}_b.png', optimize=True)
            Image.fromarray(np.stack([enc(pres, 'pressure'), enc(np.sqrt(np.maximum(snod, 0)), 'snow_sqrt'), enc(waves, 'waves')], -1)).save(f'{wd}/{n}_c.png', optimize=True)
            Image.fromarray(np.stack([enc(u, 'wind'), enc(v, 'wind'), enc(gust, 'gust')], -1)).save(f'{wd}/{n}_w.png', optimize=True)
            idx[str(n)] = f'{ms(c)}+{f}'
            print(f'weather {t:%m-%d %H:%M} from {c:%HZ}+{f}h: {np.nanmin(tc):.0f}..{np.nanmax(tc):.0f} C, max wind {np.nanmax(np.hypot(u, v)):.0f} m/s')
        except Exception as e:
            print(f'weather {t:%m-%d %H:%M} failed: {e}')
    keep = {str(ms(t)) for t in plan(now, newest)}
    for k in list(idx):
        if k not in keep:
            idx.pop(k)
            for f in glob.glob(f'{wd}/{k}_*.png'): os.remove(f)
    json.dump(idx, open(idx_p, 'w'))
    os.makedirs(f'{site}/wx', exist_ok=True)
    times = sorted(int(k) for k in idx if os.path.exists(f'{wd}/{k}_w.png'))
    for k in times:
        for s in 'abcw':
            if os.path.exists(f'{wd}/{k}_{s}.png'): os.link(f'{wd}/{k}_{s}.png', f'{site}/wx/{k}_{s}.png') if False else \
                __import__('shutil').copy(f'{wd}/{k}_{s}.png', f'{site}/wx/{k}_{s}.png')
    json.dump({'times': times, 'run': ms(newest), 'size': [720, 360], 'scale': SCALE}, open(f'{site}/wx.json', 'w'))
    print(f'weather: {len(times)} times published, model run {newest:%m-%d %HZ}, {len(todo)} fetched this time')
