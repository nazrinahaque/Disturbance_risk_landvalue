# ============================================================
# DISTURBANCE RISK AND LAND VALUE — CANAY (2011) QUANTILE FE
# Author: Nazrina Haque
# Description: 2-step quantile fixed effects estimator
#              following Canay (2011)
#
# Step 1: OLS with county FE → extract individual effects
# Step 2: Subtract FE from outcome → run quantile regression
#
# Models:
#   A. NDVI disturbance (median, tau = 0.5)
#   B. FIA forest damage (20th percentile, tau = 0.2)
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

print("Sample loaded:", df_reg.shape)


# ============================================================
# 2. CONTROLS LIST
# ============================================================
controls_list = [
    'di', 'elevation', 'mtemp', 'pi',
    'precip', 'vpdmax', 'vpdmin',
    'near_dist_roads', 'near_dist_urban'
]


# ============================================================
# 3. CANAY (2011) FUNCTION
# ============================================================
def canay_quantile_fe(data, treatment_vars, controls_list, quantile=0.5):
    """
    Canay (2011) 2-step quantile fixed effects estimator.

    Parameters:
        data           : DataFrame
        treatment_vars : list of treatment variable names
        controls_list  : list of control variable names
        quantile       : quantile to estimate (default 0.5)

    Returns:
        Fitted quantile regression model
    """
    # ----------------------------------------------------------
    # Step 1 — OLS with county FE to extract individual effects
    # ----------------------------------------------------------
    fe_formula = (
        'lperacresale ~ ' +
        ' + '.join(treatment_vars) +
        ' + ' + ' + '.join(controls_list) +
        ' + C(county) + C(year)'
    )

    fe_model  = smf.ols(fe_formula, data=data).fit()

    county_fe = fe_model.fittedvalues - (
        fe_model.fittedvalues - fe_model.resid -
        data.groupby('county')['lperacresale'].transform('mean') +
        data['lperacresale'].mean()
    )

    # ----------------------------------------------------------
    # Step 2 — Subtract FE, run quantile regression
    # ----------------------------------------------------------
    data = data.copy()
    data['logprice_FE'] = data['lperacresale'] - county_fe

    qr_formula = (
        'logprice_FE ~ ' +
        ' + '.join(treatment_vars) +
        ' + ' + ' + '.join(controls_list) +
        ' + C(year)'
    )

    return smf.quantreg(qr_formula, data=data).fit(q=quantile)


# ============================================================
# 4. MODEL A — NDVI DISTURBANCE (tau = 0.5, median)
# ============================================================
print("\n" + "="*60)
print("CANAY (2011): NDVI Disturbance — Median (tau = 0.5)")
print("="*60)

treatment_ndvi = [
    'michael_ndvi',
    'michael_ndvi:afterpolicy',
    'michaelnot_ndvi',
    'michaelnot_ndvi:afterpolicy',
    'other_hurricane_hit',
    'other_hurricane_hit:afterpolicy'
]

qfe_ndvi = canay_quantile_fe(
    df_reg, treatment_ndvi, controls_list, quantile=0.5
)
print(qfe_ndvi.summary().tables[1])

print("\nPercent effects on land value:")
print(f"  Michael + NDVI damage (pre-2020):    {(np.exp(qfe_ndvi.params.get('michael_ndvi',    0)) - 1)*100:.2f}%")
print(f"  Michael + NDVI damage (post-2020):   {(np.exp(qfe_ndvi.params.get('michael_ndvi',    0) + qfe_ndvi.params.get('michael_ndvi:afterpolicy',    0)) - 1)*100:.2f}%")
print(f"  Michael no NDVI damage (pre-2020):   {(np.exp(qfe_ndvi.params.get('michaelnot_ndvi', 0)) - 1)*100:.2f}%")
print(f"  Michael no NDVI damage (post-2020):  {(np.exp(qfe_ndvi.params.get('michaelnot_ndvi', 0) + qfe_ndvi.params.get('michaelnot_ndvi:afterpolicy', 0)) - 1)*100:.2f}%")


# ============================================================
# 5. MODEL B — FIA FOREST DAMAGE (tau = 0.2, 20th percentile)
# ============================================================
print("\n" + "="*60)
print("CANAY (2011): FIA Forest Damage — 20th Percentile (tau = 0.2)")
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
print(f"  Michael + forest damage (pre-2020):    {(np.exp(qfe_damage.params.get('michaeldamaged',    0)) - 1)*100:.2f}%")
print(f"  Michael + forest damage (post-2020):   {(np.exp(qfe_damage.params.get('michaeldamaged',    0) + qfe_damage.params.get('michaeldamaged:afterpolicy',    0)) - 1)*100:.2f}%")
print(f"  Michael no forest damage (pre-2020):   {(np.exp(qfe_damage.params.get('michaelnotdamaged', 0)) - 1)*100:.2f}%")
print(f"  Michael no forest damage (post-2020):  {(np.exp(qfe_damage.params.get('michaelnotdamaged', 0) + qfe_damage.params.get('michaelnotdamaged:afterpolicy', 0)) - 1)*100:.2f}%")

print("\n✅ Canay (2011) quantile FE regressions complete!")
