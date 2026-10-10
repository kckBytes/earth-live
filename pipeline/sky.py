"""More things on the globe, all free: space stations (CelesTrak orbits + SGP4), the five satellites whose images
make our clouds, NOAA's aurora forecast, NASA FIRMS fire detections and GOES lightning (GLM).

Outputs (site/):
  sky.json      {"cameras": [{name, lon}], "fires": [[ms, lat, lon, power]], "lightning": [[ms, lat, lon, flashes]] (6 h),
                 "aurora_time": ms}
  sats.json     {"sats": [{name, kind, alt_km, track: [[ms, lat, lon]...]}]}  (-24 h .. +12 h, every minute)
  aurora.png    360 x 181, lon -180..180 (left to right), lat 90..-90: chance of aurora overhead, 0..255 = 0..100 %
"""
import os, io, re, json, glob, math, csv, urllib.request, concurrent.futures as cf
from datetime import datetime, timedelta, timezone
import numpy as np
from PIL import Image

UA = {'User-Agent': 'earth-live (https://github.com/kckBytes/earth-live)'}
STATIONS = [(25544, 'ISS', 'Space station'), (48274, 'Tiangong', 'Space station'), (20580, 'Hubble', 'Telescope')]
CAMERAS = [('GOES-East', -75.2), ('GOES-West', -137.2), ('Himawari', 140.7), ('MTG', 0.0), ('Meteosat', 45.5)]
FIRMS = 'https://firms.modaps.eosdis.nasa.gov/data/active_fire/noaa-20-viirs-c2/csv/J1_VIIRS_C2_Global_24h.csv'
AURORA = 'https://services.swpc.noaa.gov/json/ovation_aurora_latest.json'
GLM = 'https://{b}.s3.amazonaws.com/?list-type=2&prefix=GLM-L2-LCFA/{y}/{doy:03d}/{h:02d}/'

def raw(url, timeout=60):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read()

def ms(t): return int(t.timestamp() * 1000)

def stations(now):
    from sgp4.api import Satrec, jday
    out = []
    for cat, name, kind in STATIONS:
        try:
            l = [x.strip() for x in raw(f'https://celestrak.org/NORAD/elements/gp.php?CATNR={cat}&FORMAT=TLE', 30).decode().splitlines() if x.strip()]
            sat = Satrec.twoline2rv(l[1], l[2])
            tr, alt = [], 0
            for k in range(-24 * 60, 12 * 60 + 1):                     # every minute, -24 h .. +12 h
                t = now + timedelta(minutes=k)
                jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond / 1e6)
                e, r, _ = sat.sgp4(jd, fr)
                if e: continue
                # TEME -> Earth-fixed via Greenwich sidereal angle (good to well under a pixel here)
                d = jd + fr - 2451545.0
                gmst = math.radians((280.46061837 + 360.98564736629 * d) % 360)
                x = r[0] * math.cos(gmst) + r[1] * math.sin(gmst); y = -r[0] * math.sin(gmst) + r[1] * math.cos(gmst); z = r[2]
                rr = math.sqrt(x * x + y * y + z * z); alt = rr - 6371
                tr.append([ms(t), round(math.degrees(math.asin(z / rr)), 2), round(math.degrees(math.atan2(y, x)), 2)])
            out.append({'name': name, 'kind': kind, 'alt_km': round(alt), 'track': tr})
        except Exception as e:
            print('sky: orbit for', name, 'failed:', e)
    return out

def aurora(site):
    a = json.loads(raw(AURORA, 60))
    g = np.zeros((181, 360), np.float32)
    for lon, lat, p in a['coordinates']:
        x = int(round(lon)) % 360; x = (x + 180) % 360                  # 0..359 east -> -180 at the left
        g[90 - int(round(lat)), x] = p
    Image.fromarray(np.clip(g * 2.55, 0, 255).astype(np.uint8)).save(f'{site}/aurora.png', optimize=True)
    t = datetime.fromisoformat(a['Forecast Time'].replace('Z', '+00:00'))
    print(f'sky: aurora forecast {t:%H:%M}, strongest {g.max():.0f} %')
    return ms(t)

def fires(now):
    rows = list(csv.DictReader(io.StringIO(raw(FIRMS, 120).decode('utf-8', 'replace'))))
    cells = {}
    for r in rows:
        try:
            lat, lon, frp = float(r['latitude']), float(r['longitude']), float(r['frp'] or 0)
            t = datetime.strptime(r['acq_date'] + r['acq_time'].zfill(4), '%Y-%m-%d%H%M').replace(tzinfo=timezone.utc)
        except Exception: continue
        k = (round(lat * 4) / 4, round(lon * 4) / 4)                    # 0.25-degree cells: one dot per fire area
        c = cells.get(k)
        if c is None: cells[k] = [ms(t), lat, lon, frp]
        else: c[3] += frp; c[0] = min(c[0], ms(t))
    big = sorted(cells.values(), key=lambda c: -c[3])[:600]
    print(f'sky: fires {len(rows)} detections -> {len(cells)} areas, showing the {len(big)} strongest')
    return [[c[0], round(c[1], 2), round(c[2], 2), round(c[3])] for c in big]

def lightning(now, work):
    """GOES-East / West lightning mapper: flashes of the last ~10 min binned to 0.25 degrees; 24 h kept in 10-min bins."""
    import netCDF4
    hist_p = f'{work}/lightning.json'
    hist = json.load(open(hist_p)) if os.path.exists(hist_p) else {}
    bin_t = now.replace(second=0, microsecond=0) - timedelta(minutes=now.minute % 10 + 10)      # the last complete 10 min
    key = str(ms(bin_t))
    if key not in hist:
        urls = []
        for b in ('noaa-goes19', 'noaa-goes18'):
            for hh in sorted({bin_t.hour, (bin_t + timedelta(minutes=10)).hour}):
                day = bin_t if hh == bin_t.hour else bin_t + timedelta(minutes=10)
                try:
                    x = raw(GLM.format(b=b, y=day.year, doy=day.timetuple().tm_yday, h=hh), 30).decode()
                except Exception: continue
                for k in re.findall(r'<Key>([^<]+)</Key>', x):
                    m = re.search(r'_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})', k)
                    if not m: continue
                    t = datetime(int(m.group(1)), 1, 1, tzinfo=timezone.utc) + timedelta(days=int(m.group(2)) - 1, hours=int(m.group(3)), minutes=int(m.group(4)), seconds=int(m.group(5)))
                    if bin_t <= t < bin_t + timedelta(minutes=10): urls.append(f'https://{b}.s3.amazonaws.com/{k}')
        cells = {}
        def get_bytes(u):
            try: return raw(u, 60)
            except Exception: return None
        with cf.ThreadPoolExecutor(12) as ex:                       # download in parallel, decode one by one
            blobs = list(ex.map(get_bytes, urls))                     # (the HDF5 library underneath is not thread-safe)
        for b in blobs:
            if not b: continue
            try:
                ds = netCDF4.Dataset('mem', memory=b)
                la = np.ma.filled(np.ma.asarray(ds['flash_lat'][:], float), np.nan)
                lo = np.ma.filled(np.ma.asarray(ds['flash_lon'][:], float), np.nan)
                ds.close()
            except Exception: continue
            ok = np.isfinite(la) & np.isfinite(lo)
            for a, c in zip(np.round(la[ok] * 4) / 4, np.round(lo[ok] * 4) / 4):
                k = (float(a), float(c)); cells[k] = cells.get(k, 0) + 1
        top = sorted(cells.items(), key=lambda kv: -kv[1])[:150]
        hist[key] = [[k[0], k[1], n] for k, n in top]
        print(f'sky: lightning {bin_t:%H:%M} from {len(urls)} GLM files: {sum(cells.values())} flashes in {len(cells)} cells')
    for k in list(hist):
        if ms(now) - int(k) > 24.5 * 3600_000: hist.pop(k)
    json.dump(hist, open(hist_p, 'w'))
    return [[int(k), a, b, n] for k, v in sorted(hist.items()) if ms(now) - int(k) <= 6.5 * 3600_000 for a, b, n in v]

def build(work, site, now=None):
    """sky.json (small, changes every run): cameras, fires, lightning of the last 6 h, aurora time.
    sats.json (bigger, the phone fetches it a few times a day): station tracks -24 h .. +12 h, one point a minute."""
    now = now or datetime.now(timezone.utc)
    out = {'generated': ms(now), 'cameras': [{'name': n, 'lon': lon} for n, lon in CAMERAS],
           'fires': [], 'lightning': [], 'aurora_time': 0}
    for name, f in (('fires', lambda: fires(now)), ('lightning', lambda: lightning(now, work)), ('aurora_time', lambda: aurora(site))):
        try: out[name] = f()
        except Exception as e: print('sky:', name, 'failed:', e)
    json.dump(out, open(f'{site}/sky.json', 'w'), separators=(',', ':'))
    try:
        st = stations(now)
        json.dump({'generated': ms(now), 'sats': st}, open(f'{site}/sats.json', 'w'), separators=(',', ':'))
    except Exception as e:
        st = []; print('sky: stations failed:', e)
    print(f"sky: {len(st)} orbits, {len(out['fires'])} fire areas, {len(out['lightning'])} lightning cells")

if __name__ == '__main__':
    os.makedirs('skytest', exist_ok=True); build('skytest', 'skytest')
