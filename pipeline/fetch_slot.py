"""Download the newest imagery from all five geostationary satellites (free, no key) + reference layers.

Sources
  NASA GIBS WMTS (0.0703 deg tiles): GOES-East, GOES-West, Himawari -- infrared band 13 + red visible
  EUMETSAT view WMS: MTG-I (0 deg) IR 10.5 + VIS 0.6, MSG Indian Ocean (45.5 E) IR 10.8 + VIS 0.6
  Static/daily: GHRSST MUR25 sea-surface temperature, SRTM elevation (both GIBS), and their colour maps.
"""
import os, re, time, urllib.request, urllib.error, concurrent.futures as cf
from datetime import datetime, timedelta, timezone

UA = {'User-Agent': 'earth-live/1.0 (personal live-wallpaper project; github.com/kckBytes/earth-live)'}
GIBS = 'https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/{layer}/default/{t}/{tms}/3/{r}/{c}.png'
GIBS_DEFAULT = 'https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/{layer}/default/default/{tms}/3/{r}/{c}.png'
GIBS_STATIC = 'https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/{layer}/default/{tms}/3/{r}/{c}.png'
EUM = ('https://view.eumetsat.int/geoserver/ows?SERVICE=WMS&REQUEST=GetMap&VERSION=1.3.0&CRS=EPSG:4326'
       '&BBOX=-90,-180,90,180&WIDTH=5120&HEIGHT=2560&FORMAT=image/png&TRANSPARENT=TRUE&STYLES=&TIME={t}&LAYERS={layer}')
GIBS_LAYERS = {
    'goes_e': ('GOES-East_ABI_Band13_Clean_Infrared', '2km'),
    'goes_w': ('GOES-West_ABI_Band13_Clean_Infrared', '2km'),
    'hima': ('Himawari_AHI_Band13_Clean_Infrared', '2km'),
    'goes_e_vis': ('GOES-East_ABI_Band2_Red_Visible_1km', '1km'),
    'goes_w_vis': ('GOES-West_ABI_Band2_Red_Visible_1km', '1km'),
    'hima_vis': ('Himawari_AHI_Band3_Red_Visible_1km', '1km'),
}
EUM_LAYERS = {'mtg': 'mtg_fd:ir105_hrfi', 'iodc': 'msg_iodc:ir108', 'mtg_vis': 'mtg_fd:vis06_hrfi', 'iodc_vis': 'msg_iodc:vis006'}
STATIC = {'sst': ('GHRSST_L4_MUR25_Sea_Surface_Temperature', '2km', GIBS_DEFAULT),
          'dem': ('SRTM_Color_Index', '31.25m', GIBS_STATIC)}
CMAPS = {'cmap_ir.xml': 'Clean_Longwave_Infrared_Window_Band', 'cmap_sst.xml': 'GHRSST_Sea_Surface_Temperature',
         'cmap_dem.xml': 'SRTM_Color_Index'}

def fetch(url, path=None, timeout=60):
    err = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                data = r.read()
            if path:
                open(path, 'wb').write(data)
            return data
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            err = e
        except Exception as e:
            err = e
        time.sleep(1 + 2 * attempt)
    print('   fetch failed', url[:110], err)
    return None

def iso(t): return t.strftime('%Y-%m-%dT%H:%M:%SZ')

def latest_gibs(layer, tms, now):
    """Newest 10-minute slot that exists (probe one tile in the middle of the grid)."""
    t = now.replace(second=0, microsecond=0)
    t -= timedelta(minutes=t.minute % 10)
    for k in range(1, 13):
        cand = t - timedelta(minutes=10 * k)
        if fetch(GIBS.format(layer=layer, t=iso(cand), tms=tms, r=2, c=2), timeout=20) is not None:
            return cand
    return None

def latest_eum():
    x = fetch('https://view.eumetsat.int/geoserver/ows?service=WMS&request=GetCapabilities&version=1.3.0', timeout=60)
    out = {}
    if not x:
        return out
    x = x.decode('utf-8', 'replace')
    for k, name in EUM_LAYERS.items():
        m = re.search(r'<Name>' + re.escape(name) + r'</Name>.*?<Dimension name="time"[^>]*default="([^"]+)"', x, re.S)
        if m:
            out[k] = datetime.strptime(m.group(1)[:19], '%Y-%m-%dT%H:%M:%S').replace(tzinfo=timezone.utc)
    return out

def find_times(now):
    with cf.ThreadPoolExecutor(6) as ex:
        futs = {k: ex.submit(latest_gibs, layer, tms, now) for k, (layer, tms) in GIBS_LAYERS.items()}
        times = latest_eum()
        for k, f in futs.items():
            times[k] = f.result()
    return times

def download(times, out, static_dir):
    os.makedirs(out, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)
    jobs = []
    for k, (layer, tms) in GIBS_LAYERS.items():
        if not times.get(k):
            continue
        os.makedirs(f'{out}/{k}', exist_ok=True)
        for r in range(5):
            for c in range(10):
                jobs.append((GIBS.format(layer=layer, t=iso(times[k]), tms=tms, r=r, c=c), f'{out}/{k}/{r}_{c}.png'))
    for k, layer in EUM_LAYERS.items():
        if times.get(k):
            jobs.append((EUM.format(layer=layer, t=iso(times[k])), f'{out}/{k}.png'))
    # reference layers: elevation never changes, sea-surface temperature is daily
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    stamp = f'{static_dir}/stamp.txt'
    refresh = not (os.path.exists(stamp) and open(stamp).read().strip() == today)
    if refresh:
        for k, (layer, tms, tpl) in STATIC.items():
            if k == 'dem' and os.path.exists(f'{static_dir}/dem/2_2.png'):
                continue
            os.makedirs(f'{static_dir}/{k}', exist_ok=True)
            for r in range(5):
                for c in range(10):
                    jobs.append((tpl.format(layer=layer, tms=tms, r=r, c=c), f'{static_dir}/{k}/{r}_{c}.png'))
        for f, name in CMAPS.items():
            jobs.append((f'https://gibs.earthdata.nasa.gov/colormaps/v1.3/{name}.xml', f'{static_dir}/{f}'))
    t0 = time.time()
    with cf.ThreadPoolExecutor(16) as ex:
        res = list(ex.map(lambda j: fetch(*j), jobs))
    got = sum(len(r) for r in res if r)
    print(f'downloaded {len(jobs)} files, {got / 1e6:.1f} MB in {time.time() - t0:.1f}s')
    if refresh:
        open(stamp, 'w').write(today)
