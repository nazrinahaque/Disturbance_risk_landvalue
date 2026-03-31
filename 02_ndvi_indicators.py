# ============================================================
# DISTURBANCE RISK AND LAND VALUE — NDVI INDICATORS
# Author: Nazrina Haque
# Description: Creates NDVI-based vegetation disturbance
#              indicators for Hurricane Michael (2018)
#              Negative NDVI = confirmed vegetation damage
# ============================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 1. LOAD CLEAN DATA
# ============================================================
df = pd.read_csv('hurricane_timberland_clean.csv')
print("Data loaded:", df.shape)


# ============================================================
# 2. NDVI DECLINE AFTER HURRICANE MICHAEL
# Post-2017 only (captures Michael's impact)
# ============================================================
df['ndvidiff'] = (
    (df['ndvi_diff'] < 0) & (df['year'] > 2017)
).astype(int)


# ============================================================
# 3. MICHAEL HIT WITH/WITHOUT VEGETATION DAMAGE
# ============================================================

# Hit by Michael AND NDVI declined → confirmed damage
df['michael_ndvi'] = (
    (df['hurricane_hit'] == 1) & (df['ndvidiff'] == 1)
).astype(int)

# Hit by Michael BUT no NDVI decline → no vegetation damage
df['michaelnot_ndvi'] = (
    (df['hurricane_hit'] == 1) & (df['ndvidiff'] == 0)
).astype(int)


# ============================================================
# 4. NDVI DECLINE ACROSS ALL PARCELS (all years)
# ============================================================
df['ndvi_diff_forallparcels'] = (df['ndvi_diff'] < 0).astype(int)

# Michael hit with NDVI decline (all years)
df['michaelndvi_forallparcels'] = (
    (df['ndvi_diff_forallparcels'] == 1) & (df['michael_hit'] == 1)
).astype(int)

# Michael hit without NDVI decline (all years)
df['michaelnotndvi_forallparcels'] = (
    (df['michael_hit'] == 1) & (df['ndvi_diff_forallparcels'] == 0)
).astype(int)


# ============================================================
# 5. NDVI QUANTILE AMONG DAMAGED PARCELS
# Quartiles of damage severity
# ============================================================
damaged_mask = (df['ndvi_diff'] < 0) & (df['michael_hit'] == 1)
damaged_ndvi = df.loc[damaged_mask, 'ndvi_diff']

df.loc[damaged_mask, 'ndvi_q'] = pd.qcut(
    damaged_ndvi, q=4, labels=[1, 2, 3, 4]
)


# ============================================================
# 6. SUMMARY
# ============================================================
print("\n✅ NDVI indicators created")
print(f"   Michael + NDVI damage:    {df['michael_ndvi'].sum()}")
print(f"   Michael + no NDVI damage: {df['michaelnot_ndvi'].sum()}")
print(f"   All parcels NDVI decline: {df['ndvi_diff_forallparcels'].sum()}")


# ============================================================
# 7. SAVE
# ============================================================
df.to_csv('hurricane_timberland_clean.csv', index=False)
print("✅ NDVI indicators saved")
