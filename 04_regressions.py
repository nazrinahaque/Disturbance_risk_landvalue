# ============================================================
# DISTURBANCE RISK AND LAND VALUE — MAIN REGRESSIONS
# Author: Nazrina Haque
# Description: Four OLS regressions with robust standard
#              errors estimating hurricane disturbance
#              effects on timberland per-acre sale prices
#
# Models:
#   1. NDVI disturbance on log land value
#   2. Cheap sale probability
#   3. Parallel trend test
#   4. FIA forest damage on log land value
# ============================================================

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 1. LOAD DATA
# ============================================================
df     = pd.read_csv('hurricane_timberland_clean.csv')
df_reg = df[df['year'] >= 2014].copy()

print("Regression sample:", df_reg.shape)
print("Years:", sorted(df_reg['year'].unique()))


# ============================================================
# 2. CONTROLS STRING
# ============================================================
controls = (
    'other_hurricane_hit + di + elevation + mtemp + pi + '
    'precip + vpdmax + vpdmin + '
    'near_dist_roads + near_dist_urban + '
    'C(county) + C(year)'
)


# ============================================================
# 3. REGRESSION 1 — NDVI DISTURBANCE ON LAND VALUE
# ============================================================
print("\n" + "="*60)
print("REGRESSION 1: NDVI Disturbance on Log Per-Acre Sale Price")
print("="*60)

model1 = smf.ols(
    f'lperacresale ~ michael_ndvi + michaelnot_ndvi + {controls}',
    data = df_reg
).fit(cov_type='HC1')

print(model1.summary().tables[1])

# Percent effects
coef1 = model1.params
print(f"\n  Michael + NDVI damage:    {(np.exp(coef1.get('michael_ndvi', 0))    - 1)*100:.2f}%")
print(f"  Michael + no NDVI damage: {(np.exp(coef1.get('michaelnot_ndvi', 0)) - 1)*100:.2f}%")


# ============================================================
# 4. REGRESSION 2 — CHEAP SALE PROBABILITY
# ============================================================
print("\n" + "="*60)
print("REGRESSION 2: Cheap Sale Probability (< $100/acre)")
print("="*60)

model2 = smf.ols(
    'cheap ~ michaelndvi_forallparcels + michaelnotndvi_forallparcels + '
    'other_hurricane_hit + di + elevation + mtemp + pi + '
    'precip + vpdmax + vpdmin + C(county) + C(year)',
    data = df_reg
).fit(cov_type='HC1')

print(model2.summary().tables[1])


# ============================================================
# 5. REGRESSION 3 — PARALLEL TREND TEST
# ============================================================
print("\n" + "="*60)
print("REGRESSION 3: Parallel Trend Test (2013 onwards)")
print("="*60)

df_parallel = df[df['year'] >= 2013].copy()

model3 = smf.ols(
    f'lperacresale ~ C(year):michaelndvi_forallparcels + '
    f'michaelnotndvi_forallparcels + '
    f'C(year):michael_ndvi + michaelnot_ndvi + {controls}',
    data = df_parallel
).fit(cov_type='HC1')

print(model3.summary().tables[1])


# ============================================================
# 6. REGRESSION 4 — FIA FOREST DAMAGE ON LAND VALUE
# ============================================================
print("\n" + "="*60)
print("REGRESSION 4: FIA Forest Damage on Log Per-Acre Sale Price")
print("="*60)

model4 = smf.ols(
    f'lperacresale ~ michaeldamaged + michaelnotdamaged + {controls}',
    data = df_reg
).fit(cov_type='HC1')

print(model4.summary().tables[1])

coef4 = model4.params
print(f"\n  Michael + forest damage:    {(np.exp(coef4.get('michaeldamaged', 0))    - 1)*100:.2f}%")
print(f"  Michael + no forest damage: {(np.exp(coef4.get('michaelnotdamaged', 0)) - 1)*100:.2f}%")

print("\n✅ All regressions complete!")
