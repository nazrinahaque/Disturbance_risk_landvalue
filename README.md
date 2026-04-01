# Hurricane Timberland — Land Value Analysis
**Author:** Nazrina Haque | PhD Candidate, Oregon State University  
**Funding:** USDA Forest Service  

---

## Overview
This repository contains the econometric analysis behind my job market paper:

> *Estimating the Impact of a Category 5 Hurricane on Forest Land Value in the Southeastern U.S.*

The project estimates the causal effect of **Hurricane Michael (2018)** on
timberland transaction prices across Georgia, Alabama, South Carolina, and
Florida using parcel-level CoreLogic land sales data, forest damage surveys,
NDVI satellite imagery, and panel econometric methods.

---

## Research Question
How does hurricane disturbance — measured by both satellite-derived vegetation
damage (NDVI) and USDA Forest Inventory Analysis (FIA) forest damage severity —
affect per-acre timberland sale prices before and after landfall?

---

## Data Sources
| Data | Source |
|------|--------|
| Parcel-level land transactions | CoreLogic |
| Hurricane windswaths | NOAA National Hurricane Center |
| Forest damage severity | USDA Forest Inventory Analysis (FIA) |
| NDVI vegetation index | Sentinel-2 via Google Earth Engine |
| Climate controls | PRISM Climate Group |
| Road and urban distances | US Census TIGER / National Highway System |

---

## Repository Structure

| File | Description |
|------|-------------|
| `01_data_preparation.R` | Loads raw data, cleans outliers, creates price and time variables |
| `02_ndvi_indicators.R` | Builds NDVI-based vegetation damage indicators post-Michael |
| `03_fia_damage_indicators.R` | Builds FIA forest damage treatment variables |
| `04_regressions.R` | Runs all main OLS regressions with robust standard errors |
| `05_quantile_fe_canay2011.R` | Canay (2011) quantile fixed effects with bootstrapped SEs |
| `disturbance_risk_land_value.do` | Original Stata version of full analysis |

---

## Methods
- **Difference-in-differences (DiD)** comparing hurricane-hit vs. non-hit parcels
- **OLS with county and year fixed effects** and HC1 robust standard errors
- **Canay (2011) quantile fixed effects estimator** at median (tau=0.5) and 20th percentile (tau=0.2)
- **Bootstrapped standard errors** clustered at the county level

---

## Key Variables
| Variable | Definition |
|----------|------------|
| `lperacresale` | Log per-acre sale price |
| `hurricane_hit` | Parcel hit by Hurricane Michael (post-2018) |
| `michael_ndvi` | Michael hit + confirmed NDVI vegetation decline |
| `michaelnot_ndvi` | Michael hit + no NDVI decline |
| `michaeldamaged` | Michael hit + FIA confirmed forest damage |
| `afterpolicy` | Post-2019 dummy |
| `cheap` | Sale price under $100/acre indicator |

---

## Tools
![R](https://img.shields.io/badge/-R-276DC3?style=flat&logo=r&logoColor=white)
![Stata](https://img.shields.io/badge/-Stata-1A5E9A?style=flat)
![Python](https://img.shields.io/badge/-Python-3776AB?style=flat&logo=python&logoColor=white)

---

## Contact
Nazrina Haque | haquen@oregonstate.edu  
Department of Applied Economics, Oregon State University
