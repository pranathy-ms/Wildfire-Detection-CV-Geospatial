# Wildfire Detection and Spread Prediction Using Geospatial and Machine Learning Data

This project is a semester-long research initiative focused on detecting wildfires and predicting their spread using a combination of satellite data, geospatial processing, and machine learning. It began as a computer vision-based detection task and gradually evolved into a geospatial ML pipeline after discovering the potential of VIIRS satellite data.

---

## Project Timeline & Evolution

### Phase 1: CV-based Ground Truth Dataset (SCIPY Paper)
- Initial attempt to build a **ground truth wildfire dataset** using image-based computer vision.
- Scripts under `SCIPY Paper based ground truth dataset/` were used to prepare, test, and export visual data.

### Phase 2: Transition to VIIRS + Geospatial Data
- Learned about **NASA VIIRS fire datasets** with latitude, longitude, FRP (Fire Radiative Power), and confidence.
- Shifted focus to **geospatial ML instead of CV**, which allowed use of satellite time series data.

### Phase 3: Feature Engineering and Spatial Modeling
- Processed **elevation data (DEM)** and calculated **slope** using `richdem`.
- Pulled **ERA5 wind data** for `u10` and `v10` at hourly resolution.
- Mapped VIIRS fire points to terrain and wind features to build training data.

### Phase 4: ML Modeling and Interactive Maps
- Trained a **Random Forest Regressor and Classifier** to predict fire confidence levels.
- Created interactive **Folium visualizations**:
  - Layered slope raster
  - Wind vectors
  - Fire points colored by fire intensity

---

## Repository Structure
```
Wildfire-Detection-CV-Geospatial/
├── SCIPY Paper based ground truth dataset/
│   ├── data_export.py
│   ├── env_initialization.py
│   └── test.py
│
├── VIIRS Analysis/
│   ├── combined_analysis.ipynb
│   ├── viirs_fire_detection_data_analysis.ipynb
│   ├── viirs_fire_data_ingestion.py
│   ├── viirs_fire_detection_data_download.py
│   ├── CDS_Wind_data.py
│   ├── Lidar_elevation_data.py
│   ├── slope.tif
│   ├── slope_overlay.png
│   ├── wildfire_analysis.html
│   ├── wildfire_analysis_layers.html
│   ├── wildfire_timeseries.html
│   ├── fire_analysis_map.html
│   ├── daily_detections.png
│   ├── daily_fires.png
│   ├── custom_date_fires.geojson
│   ├── viirs_fires_country.geojson
│   └── wildfire_db.sqlite
│
├── README.md
├── .gitignore
└── requirements.txt
```

---

## Datasets Used

| Dataset                     | Description                             | Source                                  |
|----------------------------|-----------------------------------------|-----------------------------------------|
| VIIRS Active Fires         | Fire points, confidence, FRP            | [NASA FIRMS](https://earthdata.nasa.gov) |
| USGS 3DEP DEM              | Elevation data (30m resolution)         | [USGS](https://www.usgs.gov/)            |
| ERA5 Reanalysis (u10/v10)  | Wind speed and direction data           | [Copernicus](https://cds.climate.copernicus.eu/) |

---

## How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/pranathy-ms/Wildfire-Detection-CV-Geospatial.git
   cd Wildfire-Detection-CV-Geospatial

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt

3. **Open notebooks**:
    Start with combined_analysis.ipynb under VIIRS Analysis/
    Follow the data extraction, processing, and model training flow

4. **Visualize output**:
    Run the Folium map cell to generate wildfire_analysis_layers.html
    Open in browser to explore layers interactively


## Key Learnings
- First hands-on experience with geospatial data, CRS alignment, and DEM manipulation
- Learned to use xarray, rasterio, and GeoPandas for layered data fusion
- Shifted from CV to satellite ML pipelines for wildfire prediction
- Built interpretable visualizations with wind vectors and temporal layers

## Results & Observations
- Predicting VIIRS confidence using Random Forest Regressor yielded low R² (~ -0.15)
- This allowed the pivot to classification model which yielded with better R² (~ 0.86) 
- Confidence labels may not correlate strongly with terrain + wind
- Predicted spread points could still be visualized with color-coded FRP

## Visualization Output (Folium Interactive Map)
### Key Elements in the Visualization:
- Circle Colors and Size: Proportional to FRP (Fire Radiative Power), representing fire intensity in megawatts
- Slope Raster (Background): Dark red shading indicates steeper slopes, which can influence fire spread
- Blue Lines: Represent wind vectors — direction and strength of the wind based on ERA5 u10/v10 values
- Popup Info: Clicking a fire point shows:
    - Actual Confidence (when toggled on)
    - FRP (MW)
    - Slope (degrees) at that point
    - Predicted Confidence (when toggled on)

This map provides a layered understanding of fire behavior based on satellite data and environmental conditions.
![Wildire Layers Analysis](VIIRS%20Analysis/Wildfire_Analysis.png)

## Future Work
- Integrate TimestampedGeoJson for real date slider-based spread visualization
- Generate synthetic fire points based on slope + wind vectors for forward prediction
- Connect to real-time data feeds for automated monitoring

## Acknowledgements
- NASA FIRMS and USGS for open access to geospatial datasets
- Copernicus Climate Data Store for ERA5 reanalysis

