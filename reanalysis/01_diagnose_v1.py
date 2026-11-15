"""v1 diagnosis: reproduces the original thesis pipeline (notebook 01) and isolates the effect of each flaw.

Ablation scenarios (one factor changed at a time, same split and seed):
  A = original pipeline (24h and 48h, exactly as in the notebook)
  B = with / without `total_days_observed` (the leaky feature)
  C = negatives cut at a random point of their stay instead of at discharge

Usage: python reanalysis/01_diagnose_v1.py . reanalysis/results/results_v1_diagnosis.json   (data in ./Data)
Note: variable names and the data preparation intentionally mirror the original notebook cells.
"""
from __future__ import annotations

import sys, os, json, random, warnings
from typing import Any

import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split, cross_val_predict
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
BASE, OUT = sys.argv[1], sys.argv[2]
os.chdir(BASE)
SEED = 77
np.random.seed(SEED); random.seed(SEED)
garantee_dtime = lambda c: pd.to_datetime(c, errors="coerce")
last_not_null = lambda x: x.dropna().iloc[-1] if x.notna().any() else np.nan

# ------------------------------------------------------------------ cells 2-13 (same as the notebook)
def load_base() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild the v1 hourly dataset exactly as notebook cells 2-13 do (including the flaws under study).

    Returns the merged hourly frame (with sepse_24h / sepse_48h labels) and the demographics frame.
    """
    df_label = pd.read_csv('Data/training_data/SepsisLabel_train.csv').drop_duplicates()
    df_labs = pd.read_csv('Data/training_data/measurement_lab_train.csv').drop_duplicates()
    df_meds = pd.read_csv('Data/training_data/measurement_meds_train.csv').drop_duplicates()
    df_obs = pd.read_csv('Data/training_data/measurement_observation_train.csv').drop_duplicates()
    df_demo = pd.read_csv('Data/training_data/person_demographics_episode_train.csv').drop_duplicates()
    for d in (df_label, df_labs, df_meds, df_obs):
        d['measurement_datetime'] = garantee_dtime(d['measurement_datetime'])
        d.rename(columns={'measurement_datetime': 'datetime'}, inplace=True)
    df_demo['visit_start_date'] = garantee_dtime(df_demo['visit_start_date'])
    df_demo.rename(columns={'visit_start_date': 'datetime'}, inplace=True)

    # cell 4: timeline, synthetic D-1/D-2 rows for positives and the leaky total_days_observed
    df = df_label.sort_values(['person_id', 'datetime']).copy()
    df['qtty_days'] = df.groupby('person_id')['datetime'].transform(lambda x: (x - x.min()).dt.days)
    first_sepsis = df[df['SepsisLabel'] == 1].groupby('person_id').first().reset_index()
    first_record = df.groupby('person_id').first().reset_index()
    base = first_sepsis[['person_id', 'datetime']].merge(first_record, on='person_id', suffixes=('_sepse', '_base'))
    rows = []
    for _, row in base.iterrows():
        for lag in [1, 2]:
            nr = row.copy()
            nr['datetime'] = row['datetime_sepse'] - pd.Timedelta(days=lag)
            nr['qtty_days'] = nr['qtty_days'] - lag
            rows.append(nr)
    df_new = pd.DataFrame(rows)[df.columns]
    df_new['_synthetic'] = 1
    df['_synthetic'] = 0
    df = pd.concat([df, df_new], ignore_index=True).sort_values(['person_id', 'datetime']).copy()
    last_date = df.groupby('person_id')['datetime'].max().reset_index().rename(columns={'datetime': 'last_datetime'})
    df = df.merge(last_date, on='person_id', how='left')
    df['first_datetime'] = df.groupby('person_id')['datetime'].transform('min')
    df['total_days_observed'] = (df['last_datetime'] - df['first_datetime']).dt.days
    fs = df[df['SepsisLabel'] == 1].groupby('person_id')['datetime'].min().reset_index().rename(columns={'datetime': 'first_sepsis_datetime'})
    df = df.merge(fs, on='person_id', how='left')
    df = df[(df['first_sepsis_datetime'].isna()) | (df['datetime'] <= df['first_sepsis_datetime'])].copy()

    # cell 6: rename columns
    df_meds.rename(columns={'Systolic blood pressure': 'systolic_pressure', 'Diastolic blood pressure': 'diastolic_pressure',
                            'Body temperature': 'temperature', 'Respiratory rate': 'resp_rate', 'Heart rate': 'heart_rate',
                            'Measurement of oxygen saturation at periphery': 'oxygen_periphery',
                            'Oxygen/Gas total [Pure volume fraction] Inhaled gas': 'oxygen_total'}, inplace=True)
    df_obs = df_obs[['person_id', 'visit_occurrence_id', 'datetime', 'Left pupil Diameter Auto', 'Right pupil Diameter Auto',
                     'Glasgow coma scale', 'Capillary refill [Time]', 'Pulse', 'Arterial pulse pressure']]
    df_obs = df_obs.rename(columns={'Left pupil Diameter Auto': 'l_pupil_diameter', 'Right pupil Diameter Auto': 'r_pupil_diameter',
                                    'Glasgow coma scale': 'glasgow', 'Capillary refill [Time]': 'capillary_refil',
                                    'Pulse': 'pulse', 'Arterial pulse pressure': 'art_pressure'})
    rename_cols = {
        'Bilirubin.total [Moles/volume] in Serum or Plasma': 'bilirubin', 'C reactive protein [Mass/volume] in Serum or Plasma': 'c_reactive_protein',
        'Procalcitonin [Mass/volume] in Serum or Plasma': 'procalcitonin', 'Lactate [Moles/volume] in Blood': 'lactate',
        'Creatinine [Mass/volume] in Blood': 'creatinine', 'Partial thromboplastin time': 'thromboplastin_time',
        'White blood cell count': 'white_cell', 'Hemoglobin [Moles/volume] in Blood': 'hemoglobin',
        'Base excess in Venous blood by calculation': 'base_excess_venous', 'Base excess in Arterial blood by calculation': 'base_excess_arterial',
        'Phosphate [Moles/volume] in Serum or Plasma': 'phosphate', 'Potassium [Moles/volume] in Blood': 'potassium',
        'Neutrophil Ab [Units/volume] in Serum': 'neutrophil_ap', 'Bicarbonate [Moles/volume] in Arterial blood': 'bicarbonate_arterial',
        'Hematocrit [Volume Fraction] of Blood': 'hematocrit', 'Glucose [Moles/volume] in Serum or Plasma': 'glucose',
        'Calcium [Moles/volume] in Serum or Plasma': 'calcium', 'Chloride [Moles/volume] in Blood': 'chloride',
        'Sodium [Moles/volume] in Serum or Plasma': 'sodium', 'Carbon dioxide [Partial pressure] in Venous blood': 'carbon_dioxide_venous',
        'Oxygen [Partial pressure] in Venous blood': 'oxygen_venous', 'Albumin [Mass/volume] in Serum or Plasma': 'albumin_venous',
        'Bicarbonate [Moles/volume] in Venous blood': 'bicarbonate_venous', 'Oxygen [Partial pressure] in Arterial blood': 'oxygen_arterial',
        'Carbon dioxide [Partial pressure] in Arterial blood': 'carbon_dioxide_arterial', 'Interleukin 6 [Mass/volume] in Body fluid': 'interleukin6',
        'Magnesium [Moles/volume] in Blood': 'magnesium', 'Prothrombin time (PT)': 'prothrombin_time', 'Fibrinogen measurement': 'fibrinogen',
        'Bilirubin measurement': 'bilirubin_measurement', ' activated': 'activated', 'Total white blood count': 'white_blood',
        'Platelet count': 'platelet', 'Blood venous pH': 'venous_ph', 'D-dimer level': 'd_dimer_level', 'Blood arterial pH': 'arterial_ph',
        'Ionised calcium measurement': 'ionised_calcium'}
    df_labs.rename(columns=rename_cols, inplace=True)
    # cell 7: one-hot encoding of categorical observations
    cap_map = {'Normal capillary filling': 'normal', 'Increased capillary filling time': 'increased',
               'Decreased capillary filling time': 'decreased', np.nan: 'none'}
    df_obs['capillary_refil'] = df_obs['capillary_refil'].map(cap_map)
    df_obs = df_obs.rename(columns={'capillary_refil': 'capillary'})
    df_obs = (pd.concat([df_obs, pd.get_dummies(df_obs, columns=['capillary', 'art_pressure', 'pulse'])
                         .drop(columns=['l_pupil_diameter', 'r_pupil_diameter', 'glasgow', 'person_id', 'visit_occurrence_id', 'datetime'])], axis=1)
              .drop(columns=['capillary', 'pulse', 'art_pressure']))
    enc = ['capillary_decreased', 'capillary_increased', 'capillary_none', 'capillary_normal', 'art_pressure_Absent',
           'art_pressure_Present', 'art_pressure_Weak', 'pulse_Absent', 'pulse_Present', 'pulse_Weak']
    df_obs[enc] = df_obs[enc].astype('int32').fillna(0)
    # cell 8: merge all tables on (person_id, datetime)
    df_labs = df_labs.sort_values(['person_id', 'datetime']).drop(columns='visit_occurrence_id')
    df_meds = df_meds.sort_values(['person_id', 'datetime']).drop(columns='visit_occurrence_id')
    df_obs = df_obs.sort_values(['person_id', 'datetime']).drop(columns='visit_occurrence_id')
    df_demo = df_demo.sort_values(['person_id', 'datetime']).drop(columns='visit_occurrence_id')
    df_demo = pd.get_dummies(df_demo, columns=['gender'], dtype=int).rename(columns={'gender_FEMALE': 'female', 'gender_MALE': 'male'})
    df_final = (df.merge(df_labs, on=['person_id', 'datetime'], how='left')
                .merge(df_meds, on=['person_id', 'datetime'], how='left')
                .merge(df_obs, on=['person_id', 'datetime'], how='left'))
    # cell 9: physiological plausibility limits (same as the notebook)
    LIMITS = {'temperature': (32.0, 43.0), 'hematocrit': (12.0, 70.0), 'base_excess_venous': (-30.0, 25.0), 'base_excess_arterial': (-30.0, 25.0),
              'phosphate': (0.1, 15.0), 'potassium': (1.5, 9.5), 'bilirubin': (0.0, 50.0), 'neutrophil_ap': (0.0, 80.0),
              'bicarbonate_arterial': (2.0, 45.0), 'glucose': (5.0, 800.0), 'calcium': (4.0, 18.0), 'chloride': (70.0, 130.0),
              'sodium': (105.0, 185.0), 'c_reactive_protein': (0.0, 600.0), 'carbon_dioxide_venous': (10.0, 70.0), 'oxygen_venous': (10.0, 85.0),
              'albumin_venous': (0.5, 7.0), 'bicarbonate_venous': (2.0, 45.0), 'oxygen_arterial': (20.0, 650.0),
              'carbon_dioxide_arterial': (10.0, 120.0), 'interleukin6': (0.0, 50000.0), 'magnesium': (0.2, 8.0),
              'prothrombin_time': (5.0, 120.0), 'procalcitonin': (0.0, 500.0), 'lactate': (0.2, 25.0), 'creatinine': (0.0, 10.0),
              'fibrinogen': (20.0, 1200.0), 'bilirubin_measurement': (0.0, 50.0), 'thromboplastin_time': (5.0, 120.0),
              'activated': (20.0, 200.0), 'white_blood': (0.0, 120.0), 'platelet': (1.0, 2000.0), 'white_cell': (0.0, 120.0),
              'venous_ph': (6.5, 7.65), 'd_dimer_level': (0.0, 50.0), 'arterial_ph': (6.5, 7.65), 'hemoglobin': (3.0, 28.0),
              'ionised_calcium': (0.3, 3.5), 'systolic_pressure': (20.0, 220.0), 'diastolic_pressure': (10.0, 140.0),
              'resp_rate': (0.0, 120.0), 'heart_rate': (20.0, 300.0), 'oxygen_periphery': (50.0, 102.0), 'oxygen_total': (50.0, 102.0),
              'l_pupil_diameter': (0.5, 10.0), 'r_pupil_diameter': (0.5, 10.0), 'glasgow': (3.0, 15.0)}
    for col, (lo, hi) in LIMITS.items():
        if col in df_final.columns:
            df_final.loc[(df_final[col] < lo) | (df_final[col] > hi), col] = np.nan
    # cell 11: drop columns that are entirely empty
    notok = ['person_id', 'datetime', 'SepsisLabel', 'qtty_days', 'last_datetime', 'first_sepsis_datetime', 'day', 'week',
             'first_datetime', '_synthetic']
    ok = [c for c in df_final.columns if c not in notok + enc]
    c1 = df_final[ok].describe()
    df_final = df_final.drop(columns=[i for i in c1.columns if np.isnan(c1[i]['max'])])
    # cell 13: early labels (1 on the synthetic row 1 or 2 days before onset)
    df_final['sepse_24h'] = (df_final['qtty_days'] == -1).astype(int)
    df_final['sepse_48h'] = (df_final['qtty_days'] == -2).astype(int)
    df_final = df_final[~df_final['datetime'].isna()].copy()
    return df_final, df_demo


CLIN = None


def build(df_final: pd.DataFrame, df_demo: pd.DataFrame, h: int, neg_cutoff: str = "last",
          time_feature: str = "leaky") -> tuple[pd.DataFrame, pd.Series, int]:
    """Build the one-row-per-patient table for horizon h (24 or 48), as in the notebook.

    Args:
        neg_cutoff: 'last' (original: negatives summarized up to discharge) or 'random'
            (cut-off drawn uniformly within the negative patient's stay).
        time_feature: 'leaky' (original total_days_observed), 'none', or 'until_cutoff'
            (hours observed up to the cut-off).

    Returns X, y and the number of positives whose only rows before the cut-off are synthetic.
    """
    d = df_final.copy()
    cut = d['first_sepsis_datetime'] - pd.Timedelta(hours=h)
    if neg_cutoff == "last":
        neg = d['last_datetime']
    else:
        rng = np.random.default_rng(SEED + h)
        pid = d['person_id'].unique()
        u = pd.Series(rng.uniform(0, 1, len(pid)), index=pid)
        span = d['last_datetime'] - d['first_datetime']
        neg = d['first_datetime'] + span * d['person_id'].map(u).values
    d['cutoff'] = np.where(d['first_sepsis_datetime'].notna(), cut, neg)
    d['cutoff'] = pd.to_datetime(d['cutoff'])
    d = d[d['datetime'] <= d['cutoff']].copy()
    y = f'sepse_{h}h'
    d['hours_until_cutoff'] = (d['cutoff'] - d['first_datetime']).dt.total_seconds() / 3600
    feat_cols = [c for c in d.columns if c not in ['person_id', 'datetime', 'SepsisLabel', 'visit_occurrence_id', 'qtty_days',
                                                   'last_datetime', 'first_sepsis_datetime', 'first_datetime', 'day', 'week',
                                                   'sepse_24h', 'sepse_48h', 'cutoff', 'total_days_observed',
                                                   'hours_until_cutoff', '_synthetic']]
    agg = {y: (y, 'max'), 'total_days_observed': ('total_days_observed', 'max'),
           'hours_until_cutoff': ('hours_until_cutoff', 'max'), '_only_synth': ('_synthetic', 'min')}
    for c in feat_cols:
        agg[c.lower() if c.startswith(('art_pressure', 'pulse_')) else c] = (c, last_not_null)
    g = d.groupby('person_id').agg(**agg).reset_index()
    # cell 22: _last/_mean/_max/_min of a single row per patient are all the same value
    # (std would be NaN and is removed by the 70% missingness rule)
    clin = [c for c in g.columns if c not in ['person_id', y, 'total_days_observed', 'hours_until_cutoff', '_only_synth']
            and not c.startswith(('capillary_', 'art_pressure_', 'pulse_'))]
    for c in clin:
        for s in ('last', 'mean', 'max', 'min'):
            g[f'{c}_{s}'] = g[c]
    g = g.drop(columns=clin)
    demo = df_demo.drop_duplicates(subset=['person_id'], keep='last')
    g = g.merge(demo, on='person_id', how='left').drop(columns=['datetime', 'birth_datetime'], errors='ignore')
    miss = g.isna().mean() * 100
    g = g.drop(columns=list(miss[miss >= 70].index))
    only_synth_pos = int(((g['_only_synth'] == 1) & (g[y] == 1)).sum())
    X = g.drop(columns=['person_id', y, '_only_synth', 'total_days_observed', 'hours_until_cutoff'])
    if time_feature == "leaky":
        X['total_days_observed'] = g['total_days_observed']
    elif time_feature == "until_cutoff":
        X['hours_until_cutoff'] = g['hours_until_cutoff']
    return X, g[y], only_synth_pos


# ------------------------------------------------------------------ models and search space (cells 53-55)
lr = [0.01, 0.03]; md = [2, 3, 4]; msl = [3, 7, 10]; cw = [{0: 1, 1: 30}, {0: 1, 1: 50}, {0: 1, 1: 100}]
PARAMS = {
    'LGBM': dict(boosting_type=["gbdt", "dart", "goss"], n_estimators=[300, 600, 1000], num_leaves=[20, 40], learning_rate=lr,
                 max_bin=[100, 200], subsample=[0.6, 0.9], colsample_bytree=[0.6, 0.9], min_child_samples=[4, 8, 12],
                 reg_lambda=[1, 3], min_split_gain=[0, 0.05, 0.1], class_weight=cw, metric=['average_precision', 'recall']),
    'HIST': dict(max_iter=[300, 600, 1000], max_leaf_nodes=[3, 30, 50], learning_rate=lr, max_depth=md, min_samples_leaf=msl,
                 class_weight=cw, scoring=['average_precision', 'recall']),
    'XGB': dict(n_estimators=[300, 600, 1000], max_depth=md, learning_rate=lr, subsample=[0.6, 0.9], colsample_bytree=[0.6, 0.9],
                eval_metric=['aucpr', 'logloss', 'recall'], scale_pos_weight=[30, 50, 100]),
}


def models() -> dict[str, Any]:
    """The three v1 candidate models with the notebook settings."""
    return {'LGBM': LGBMClassifier(objective='binary', random_state=SEED, n_jobs=6, verbose=-1),
            'HIST': HistGradientBoostingClassifier(random_state=SEED, early_stopping=True),
            'XGB': XGBClassifier(objective='binary:logistic', random_state=SEED, n_jobs=6)}


def thr_rule(y: np.ndarray | pd.Series, p: np.ndarray, min_prec: float = 0.25) -> float | None:
    """Grid-search threshold with the highest recall subject to precision >= min_prec (notebook rule)."""
    best_t, best_r = None, -1
    for t in np.linspace(0, 1, 2000):
        yp = (p >= t).astype(int)
        if yp.sum() == 0:
            continue
        pr = precision_score(y, yp, zero_division=0); rc = recall_score(y, yp)
        if pr >= min_prec and rc > best_r:
            best_r, best_t = rc, t
    return best_t


def boot_ci(y: np.ndarray | pd.Series, p: np.ndarray, fn: Any, n: int = 1000) -> list[float]:
    """95% percentile bootstrap CI of metric `fn`."""
    rng = np.random.default_rng(SEED); y = np.asarray(y); p = np.asarray(p); vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0:
            continue
        vals.append(fn(y[idx], p[idx]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def run(X: pd.DataFrame, y: pd.Series, name: str) -> dict[str, Any]:
    """70/30 split, randomized search for each model, then test metrics and both threshold strategies."""
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    res = {}
    best = (-1, None, None)
    for nm, m in models().items():
        s = RandomizedSearchCV(m, PARAMS[nm], n_iter=5, cv=cv, scoring='average_precision', n_jobs=6, random_state=SEED)
        s.fit(Xtr, ytr)
        folds = [s.cv_results_[f'split{i}_test_score'][s.best_index_] for i in range(5)]
        res[nm] = dict(cv_mean=float(s.best_score_), cv_median=float(np.median(folds)),
                       test_ap=float(average_precision_score(yte, s.best_estimator_.predict_proba(Xte)[:, 1])))
        if s.best_score_ > best[0]:
            best = (s.best_score_, nm, s.best_estimator_)
    _, bn, be = best
    p = be.predict_proba(Xte)[:, 1]
    out = dict(name=name, n_patients=int(len(y)), n_pos=int(y.sum()), n_test_pos=int(yte.sum()), n_features=int(X.shape[1]),
               per_model=res, best_model=bn, test_ap=float(average_precision_score(yte, p)),
               test_ap_ci=boot_ci(yte, p, average_precision_score), test_roc=float(roc_auc_score(yte, p)))
    # (a) threshold chosen on the TEST set (as in the notebook)
    t_test = thr_rule(yte, p)
    yp = (p >= t_test).astype(int)
    out['thr_on_test'] = dict(t=float(t_test), recall=float(recall_score(yte, yp)), precision=float(precision_score(yte, yp)),
                              cm=confusion_matrix(yte, yp).ravel().tolist())
    # (b) threshold chosen on out-of-fold TRAINING predictions (correct), evaluated on test
    oof = cross_val_predict(clone(be), Xtr, ytr, cv=cv, method='predict_proba')[:, 1]
    t_oof = thr_rule(ytr, oof)
    if t_oof is not None:
        yp = (p >= t_oof).astype(int)
        out['thr_oof'] = dict(t=float(t_oof), recall=float(recall_score(yte, yp)),
                              precision=float(precision_score(yte, yp, zero_division=0)),
                              cm=confusion_matrix(yte, yp).ravel().tolist())
    else:
        out['thr_oof'] = None
    # feature importance (split/gain), top 8
    imp = getattr(be, 'feature_importances_', None)
    if imp is not None:
        s = pd.Series(imp, index=X.columns).sort_values(ascending=False)
        out['top_features'] = [[k, float(v)] for k, v in s.head(8).items()]
        if 'total_days_observed' in X.columns:
            out['rank_total_days_observed'] = int(list(s.index).index('total_days_observed')) + 1
    print(json.dumps({k: out[k] for k in ('name', 'best_model', 'test_ap', 'test_ap_ci', 'thr_on_test', 'thr_oof')}, default=str), flush=True)
    return out


if __name__ == "__main__":
    df_final, df_demo = load_base()
    results = []
    plan = [
        ("A_original_24h", 24, "last", "none"),
        ("A_original_48h", 48, "last", "leaky"),
        ("B_24h_com_total_days", 24, "last", "leaky"),
        ("B_48h_sem_total_days", 48, "last", "none"),
        ("C_24h_corte_aleatorio_neg", 24, "random", "none"),
        ("C_48h_corte_aleatorio_neg", 48, "random", "none"),
    ]
    for name, h, nc, tf in plan:
        X, y, osp = build(df_final, df_demo, h, nc, tf)
        r = run(X, y, name)
        r['positives_only_synthetic_rows'] = osp
        results.append(r)
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1, default=str)
    print("DONE")
