# ============================================================
# DISTURBANCE RISK AND LAND VALUE ANALYSIS
# Author: Nazrina Haque
# Description: Estimates the effect of hurricane disturbance
#              and forest damage on per-acre timberland sale
#              prices using difference-in-differences and
#              quantile fixed effects regression (Canay 2011)
# ============================================================

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 1. LOAD DATA
# ============================================================
# Replace with your actual file path
df = pd.read_csv('hurricane_timberland_data.csv')


# ============================================================
# 2. DATA PREPARATION
# ============================================================
# County fixed effect encoding
df['county'] = pd.Categorical(df['situs_county']).codes

# Per-acre sale price and log transformation
df['peracresale']  = df['sale_amount'] / df['acres']
df['lperacresale'] = np.log(df['peracresale'])

# Extract year from sale date (e.g., 20181205 → 2018)
df['year'] = (df['sale_date'] // 10000).astype(int)

# Drop outliers and invalid observations
df = df[df['lperacresale'] >= 0]
df = df[df['acres'] >= 10]
df = df[df['peracresale'] <= 60000]

# Cheap sale indicator (under $100/acre)
df['cheap'] = (df['peracresale'] < 100).astype(int)

print("✅ Data prepared:", df.shape)


# ============================================================
# 3. NDVI DISTURBANCE INDICATORS
# ============================================================

# NDVI decline after Hurricane Michael (post-2017)
df['ndvidiff'] = ((df['ndvi_diff'] < 0) & (df['year'] > 2017)).astype(int)

# Hit by Michael WITH negative NDVI (vegetation damage confirmed)
df['michael_ndvi'] = ((df['hurricane_hit'] == 1) & (df['ndvidiff'] == 1)).astype(int)

# Hit by Michael WITHOUT negative NDVI (hit but no vegetation damage)
df['michaelnot_ndvi'] = ((df['hurricane_hit'] == 1) & (df['ndvidiff'] == 0)).astype(int)

# NDVI decline across ALL parcels (not just post-2017)
df['ndvi_diff_forallparcels'] = (df['ndvi_diff'] < 0).astype(int)

# Michael hit with/without NDVI decline (all years)
df['michaelndvi_forallparcels']    = ((df['ndvi_diff_forallparcels'] == 1) & (df['michael_hit'] == 1)).astype(int)
df['michaelnotndvi_forallparcels'] = ((df['michael_hit'] == 1) & (df['ndvi_diff_forallparcels'] == 0)).astype(int)

# NDVI quantile among damaged parcels
damaged_ndvi = df.loc[(df['ndvi_diff'] < 0) & (df['michael_hit'] == 1), 'ndvi_diff']
df.loc[damaged_ndvi.index, 'ndvi_q'] = pd.qcut(damaged_ndvi, q=4, labels=[1, 2, 3, 4])

print("✅ NDVI indicators created")


# ============================================================
# 4. FIA FOREST DAMAGE INDICATORS
# ============================================================

# Forest damage after Michael (severity == 1, post-2017)
df['damaged'] = ((df['severity'] == 1) & (df['year'] > 2017)).astype(int)

# Michael hit WITH confirmed forest damage
df['michaeldamaged']    = ((df['hurricane_hit'] == 1) & (df['damaged'] == 1)).astype(int)

# Michael hit WITHOUT forest damage
df['michaelnotdamaged'] = ((df['hurricane_hit'] == 1) & (df['damaged'] == 0)).astype(int)

print("✅ FIA damage indicators created")


# ============================================================
# 5. YEAR-QUARTER VARIABLE
# ============================================================
df['sale_date_str'] = df['sale_date'].astype(str)
df['month_of_sale'] = df['sale_date_str'].str[4:6].astype(int)

df['quarter'] = 1
df.loc[df['month_of_sale'].isin([4, 5, 6]),   'quarter'] = 2
df.loc[df['month_of_sale'].isin([7, 8, 9]),   'quarter'] = 3
df.loc[df['month_of_sale'].isin([10, 11, 12]),'quarter'] = 4

df['year_quarter'] = df['year'] * 10 + df['quarter']

# After-policy dummy (2020 and after)
df['afterpolicy'] = (df['year'] > 2019).astype(int)

print("✅ Time variables created")


# ============================================================
# 6. HELPER — CONTROLS STRING
# ============================================================
controls = ('other_hurricane_hit + di + elevation + mtemp + pi + '
            'precip + vpdmax + vpdmin + near_dist_roads + near_dist_urban + '
            'C(county) + C(year)')


# ============================================================
# 7. MAIN REGRESSION — NDVI DISTURBANCE ON LAND VALUE
# ============================================================
print("\n" + "="*60)
print("REGRESSION 1: NDVI Disturbance on Log Per-Acre Sale Price")
print("="*60)

df_reg = df[df['year'] >= 2014].copy()

formula = (f'lperacresale ~ michael_ndvi + michaelnot_ndvi + {controls}')

model1 = smf.ols(formula, data=df_reg).fit(
    cov_type='HC1'  # Robust standard errors
)
print(model1.summary().tables[1])


# ============================================================
# 8. CHEAP SALE REGRESSION
# ============================================================
print("\n" + "="*60)
print("REGRESSION 2: Cheap Sale Probability")
print("="*60)

formula2 = (f'cheap ~ michaelndvi_forallparcels + michaelnotndvi_forallparcels + '
            f'other_hurricane_hit + di + elevation + mtemp + pi + '
            f'precip + vpdmax + vpdmin + C(county) + C(year)')

model2 = smf.ols(formula2, data=df_reg).fit(cov_type='HC1')
print(model2.summary().tables[1])


# ============================================================
# 9. PARALLEL TREND TEST (Pre-2014)
# ============================================================
print("\n" + "="*60)
print("REGRESSION 3: Parallel Trend Test")
print("="*60)

df_parallel = df[df['year'] >= 2013].copy()

formula3 = (f'lperacresale ~ C(year):michaelndvi_forallparcels + '
            f'michaelnotndvi_forallparcels + '
            f'C(year):michael_ndvi + michaelnot_ndvi + {controls}')

model3 = smf.ols(formula3, data=df_parallel).fit(cov_type='HC1')
print(model3.summary().tables[1])


# ============================================================
# 10. FIA DAMAGE REGRESSION
# ============================================================
print("\n" + "="*60)
print("REGRESSION 4: FIA Forest Damage on Land Value")
print("="*60)

formula4 = (f'lperacresale ~ michaeldamaged + michaelnotdamaged + {controls}')

model4 = smf.ols(formula4, data=df_reg).fit(cov_type='HC1')
print(model4.summary().tables[1])


# ============================================================
# 11. CANAY (2011) QUANTILE FE — NDVI DISTURBANCE
# ============================================================
print("\n" + "="*60)
print("QUANTILE FE (Canay 2011): NDVI Disturbance")
print("="*60)

from statsmodels.regression.quantile_regression import QuantReg

def canay_quantile_fe(data, treatment_vars, controls_list, quantile=0.5):
    """
    Canay (2011) 2-step quantile fixed effects estimator.
    Step 1: OLS with county FE → extract individual effects
    Step 2: Quantile regression on FE-adjusted outcome
    """
    # Step 1 — Linear FE regression
    fe_formula = ('lperacresale ~ ' +
                  ' + '.join(treatment_vars) +
                  ' + ' + ' + '.join(controls_list) +
                  ' + C(county) + C(year)')

    fe_model  = smf.ols(fe_formula, data=data).fit()
    county_fe = fe_model.fittedvalues - (
        fe_model.fittedvalues - fe_model.resid -
        data.groupby('county')['lperacresale']
        .transform('mean') +
        data['lperacresale'].mean()
    )

    # Step 2 — Remove FE, run quantile regression
    data = data.copy()
    data['logprice_FE'] = data['lperacresale'] - county_fe

    qr_formula = ('logprice_FE ~ ' +
                  ' + '.join(treatment_vars) +
                  ' + ' + ' + '.join(controls_list) +
                  ' + C(year)')

    qr_model = smf.quantreg(qr_formula, data=data).fit(q=quantile)
    return qr_model


treatment_ndvi = [
    'michael_ndvi',
    'michael_ndvi:afterpolicy',
    'michaelnot_ndvi',
    'michaelnot_ndvi:afterpolicy',
    'other_hurricane_hit',
    'other_hurricane_hit:afterpolicy'
]

controls_list = [
    'di', 'elevation', 'mtemp', 'pi',
    'precip', 'vpdmax', 'vpdmin',
    'near_dist_roads', 'near_dist_urban'
]

qfe_ndvi = canay_quantile_fe(
    df_reg, treatment_ndvi, controls_list, quantile=0.5
)
print(qfe_ndvi.summary().tables[1])

# Percent effect on land value
print("\nPercent effects on land value:")
print(f"  Michael + NDVI damage (pre-2020):  {(np.exp(qfe_ndvi.params.get('michael_ndvi', 0)) - 1)*100:.2f}%")
print(f"  Michael no NDVI damage (pre-2020): {(np.exp(qfe_ndvi.params.get('michaelnot_ndvi', 0)) - 1)*100:.2f}%")


# ============================================================
# 12. CANAY (2011) QUANTILE FE — FIA DAMAGE DATA
# ============================================================
print("\n" + "="*60)
print("QUANTILE FE (Canay 2011): FIA Forest Damage")
print("="*60)

treatment_damage = [
    'michaeldamaged',
    'michaeldamaged:afterpolicy',
    'michaelnotdamaged',
    'michaelnotdamaged:afterpolicy',
    'other_hurricane_hit'
]

qfe_damage = canay_quantile_fe(
    df_reg, treatment_damage, controls_list, quantile=0.2
)
print(qfe_damage.summary().tables[1])

print("\nPercent effects on land value:")
print(f"  Michael + forest damage (pre-2020):    {(np.exp(qfe_damage.params.get('michaeldamaged', 0)) - 1)*100:.2f}%")
print(f"  Michael no forest damage (pre-2020):   {(np.exp(qfe_damage.params.get('michaelnotdamaged', 0)) - 1)*100:.2f}%")

print("\n✅ All regressions complete!")# Disturbance_risk_landvalue
