# CharityChain — Geographic Impact Mapping & Location Privacy

## 1. Overview & Purpose
Phase 11 equips CharityChain with interactive geospatial analytics. The Geographic Impact Mapping service (`GeographicAnalyticsService`) aggregates project and milestone execution locations to visualize where charitable aid is active worldwide.

---

## 2. Geospatial Validation & Invariants
All geographic inputs (e.g. from milestone targets or evidence EXIF metadata) are strictly validated:
- **Latitude Bounds**: Must satisfy `-90.0 <= latitude <= 90.0`.
- **Longitude Bounds**: Must satisfy `-180.0 <= longitude <= 180.0`.
- **Null Handling**: Projects or milestones with missing or invalid coordinates are excluded from map point layers and counted in regional fallback tables.

---

## 3. Location Privacy & Obfuscation Protocol
Beneficiary safety is paramount. Certain charitable interventions operate in high-risk contexts:
- Shelters for victims of domestic violence or human trafficking.
- Field clinics in conflict zones.
- Anti-poaching wildlife sanctuaries.

### Safety Invariants
1. Both `Project` and `Milestone` models include an explicit database column:
   ```python
   is_location_sensitive = Column(Boolean, default=False, nullable=False)
   ```
2. When `is_location_sensitive == True`:
   - Exact coordinates are **never** returned to unauthenticated users or public map views.
   - The platform calculates a **regional centroid** by rounding coordinates to 1 decimal place (~11.1 km radius at the equator):
     $$\text{lat}_{\text{coarsened}} = \text{round}(\text{lat}, 1)$$
     $$\text{lon}_{\text{coarsened}} = \text{round}(\text{lon}, 1)$$
   - The response includes `is_coarsened = True` and a standard protective notice:
     *"Location generalized to protect beneficiary security and operational safety."*

---

## 4. Geographic Aggregations & Country Density
In addition to point markers, the service compiles aggregate metrics by country code (`ISO-3166-1 alpha-2` or country name):
- `project_count`: Total public projects operating in the country.
- `milestones_completed`: Count of successfully completed milestones in that territory.
- `total_funding_cents`: Cumulative net charitable funding deployed in the country.

---

## 5. Map Visualization in Streamlit
The **Global Impact Map** page (`page_impact_map()` in `app_ui.py`) uses `folium` and `streamlit-folium`:
- Interactive Leaflet map with global tile rendering.
- Standard projects display blue markers with detailed progress popups.
- Sensitive projects display amber markers with coarse centroid indicators and privacy notices.
- A summary data table displays country-by-country deployment metrics.

