# ============================================================
# DISTURBANCE RISK AND LAND VALUE — DATA PREPARATION
# Author: Nazrina Haque
# Description: Loads raw timberland sales data, cleans
#              outliers, encodes county fixed effects,
#              and creates time variables
# ============================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 1. LOAD DATA
# ============================================================
df = pd.read_csv('hurricane_timberland_data.csv')
print("Raw data loaded:", df.shape)


# ============================================================
# 2. COUNTY ENCODING & PRICE VARIABLES
# ============================================================
df['county']       = pd.Categorical(df['situs_county']).codes
df['peracresale']  = df['sale_amount'] / df['acres']
df['lperacresale'] = np.log(df['peracresale'])

# Extract year from sale date (e.g. 20181205 → 2018)
df['year'] = (df['sale_date'] // 10000).astype(int)


# ============================================================
# 3. DROP OUTLIERS
# ============================================================
df = df[df['lperacresale'] >= 0]
df = df[df['acres']        >= 10]
df = df[df['peracresale']  <= 60000]

print("After cleaning:", df.shape)


# ============================================================
# 4. CHEAP SALE FLAG
# Under $100 per acre
# ============================================================
df['cheap'] = (df['peracresale'] < 100).astype(int)


# ============================================================
# 5. YEAR-QUARTER VARIABLE
# ============================================================
df['sale_date_str'] = df['sale_date'].astype(str)
df['month_of_sale'] = df['sale_date_str'].str[4:6].astype(int)

df['quarter'] = 1
df.loc[df['month_of_sale'].isin([4, 5, 6]),    'quarter'] = 2
df.loc[df['month_of_sale'].isin([7, 8, 9]),    'quarter'] = 3
df.loc[df['month_of_sale'].isin([10, 11, 12]), 'quarter'] = 4

df['year_quarter'] = df['year'] * 10 + df['quarter']


# ============================================================
# 6. AFTER-POLICY DUMMY (2020 and after)
# ============================================================
df['afterpolicy'] = (df['year'] > 2019).astype(int)


# ============================================================
# 7. SAVE CLEANED DATA
# ============================================================
df.to_csv('hurricane_timberland_clean.csv', index=False)
print("✅ Clean data saved:", df.shape)
print("   Years in data:", sorted(df['year'].unique()))
print("   Counties:", df['county'].nunique())
