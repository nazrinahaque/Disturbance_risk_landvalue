# ============================================================
# DISTURBANCE RISK AND LAND VALUE — FIA DAMAGE INDICATORS
# Author: Nazrina Haque
# Description: Creates forest damage indicators using
#              USDA Forest Inventory Analysis (FIA) data
#              severity == 1 indicates confirmed forest damage
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
# 2. FOREST DAMAGE AFTER MICHAEL
# severity == 1 AND post-2017
# ============================================================
df['damaged'] = (
    (df['severity'] == 1) & (df['year'] > 2017)
).astype(int)


# ============================================================
# 3. MICHAEL HIT WITH/WITHOUT CONFIRMED FOREST DAMAGE
# ============================================================

# Hit by Michael AND FIA confirms forest damage
df['michaeldamaged'] = (
    (df['hurricane_hit'] == 1) & (df['damaged'] == 1)
).astype(int)

# Hit by Michael BUT no confirmed forest damage
df['michaelnotdamaged'] = (
    (df['hurricane_hit'] == 1) & (df['damaged'] == 0)
).astype(int)


# ============================================================
# 4. SUMMARY
# ============================================================
print("\n✅ FIA damage indicators created")
print(f"   Forest damaged parcels:     {df['damaged'].sum()}")
print(f"   Michael + forest damage:    {df['michaeldamaged'].sum()}")
print(f"   Michael + no forest damage: {df['michaelnotdamaged'].sum()}")


# ============================================================
# 5. SAVE
# ============================================================
df.to_csv('hurricane_timberland_clean.csv', index=False)
print("✅ FIA indicators saved")
