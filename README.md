# earth-live

A free, keyless near-real-time global cloud map stitched from five geostationary weather satellites every 10 minutes by GitHub Actions and published on GitHub Pages:
**https://kckbytes.github.io/earth-live/** (`clouds_4096.jpg`, `clouds_2048.jpg`, `flow.png`, `meta.json`).

It feeds the [Earth](https://kckbytes.github.io/earth-web/) globe (web) and the Earth live wallpaper (Android).

## Method (`pipeline/stitch.py`)
1. **Infrared from all five satellites → brightness temperature.**
   - GOES-East, GOES-West and Himawari come from NASA GIBS; their colour maps are inverted to temperature.
   - Meteosat MTG 0° and Meteosat Indian Ocean come from EUMETSAT; their grey scale is calibrated to the GIBS temperatures where the discs overlap.
   - Each pixel is blended across satellites by viewing angle, so there are no seams.
2. **Clear-sky reference.**
   - Land: a local warm envelope computed on terrain-corrected temperature (5.5 K/km, SRTM elevation), so high plateaus aren't mistaken for cloud.
   - Sea: GHRSST sea-surface temperature minus the clear atmosphere's water-vapour offset, estimated per latitude band from the clearest pixels. Low, warm cloud decks show even at night.
3. **Visible light in daylight**, relative to known surface brightness, with sunglint excluded. It adds low clouds the infrared can't separate.
4. **Motion:** block matching against the map from ~1 h earlier gives a 64×32 wind field (`pipeline/flow.py`).

Contains modified EUMETSAT data. NASA GIBS imagery (GOES via NOAA, Himawari via JMA). GHRSST MUR SST. SRTM elevation.
