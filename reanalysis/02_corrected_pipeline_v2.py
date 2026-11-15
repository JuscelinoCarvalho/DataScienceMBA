"""
Early sepsis prediction in pediatric ICU — CORRECTED PIPELINE (v2)
==================================================================
Standalone script. The original thesis notebook
(notebooks/01_pipeline_agregacao_24h_48h_PT-BR.ipynb) is left untouched.

Fixes relative to v1:
  1. Drops `total_days_observed`, which used the patient's last timestamp (including after sepsis onset).
     It is replaced by `hours_since_admission`, computed only up to the cut-off.
  2. Drops the synthetic D-1/D-2 rows. The label comes directly from the first hour with SepsisLabel=1.
  3. Same inclusion rule for both classes: at least MIN_OBS_H hours of stay and at least one real
     clinical measurement before the cut-off. This removes positives with no data at all.
  4. Matched negative cut-offs: sampled from the positives' "hours since admission" distribution, and
     the negative must remain observed for another h hours (a true negative).
  5. Features use history up to the cut-off only: last, mean, min, max, number of measurements,
     hours since the last measurement, mean over the last 24h and 24h trend.
  6. Evaluation: threshold chosen out-of-fold on the training set, 95% bootstrap CIs, repeated
     stratified 5x5 CV and temporal validation (train on older patients, test on newer ones).

Usage:   python reanalysis/02_corrected_pipeline_v2.py   (Kaggle data in ./Data/training_data)
Outputs: results/results_v2.json, figures/pr_*.png, bases/base_{24,48}h_v2.csv (not versioned)
"""
from __future__ import annotations

import os, json, warnings
from typing import Any, Callable

import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import (StratifiedKFold, RepeatedStratifiedKFold, RandomizedSearchCV,
                                     train_test_split, cross_val_predict, cross_val_score)
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_score, recall_score,
                             confusion_matrix, precision_recall_curve)
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import HistGradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "Data", "training_data")
SEED = 77
MIN_OBS_H = 6          # minimum hours of stay before the cut-off
MIN_PRECISION = 0.25   # clinical constraint used to pick the threshold (same as v1)
N_ITER = 10            # RandomizedSearchCV iterations (light version; was 20)
MISS_DROP = 0.95       # drop VALUE features with >= 95% missing (count features are kept)
rng_global = np.random.default_rng(SEED)

# Physiological plausibility limits; values outside the range become NaN (same limits as v1).
LIMITS: dict[str, tuple[float, float]] = {
          'temperature': (32.0, 43.0), 'hematocrit': (12.0, 70.0), 'base_excess_venous': (-30.0, 25.0),
          'base_excess_arterial': (-30.0, 25.0), 'phosphate': (0.1, 15.0), 'potassium': (1.5, 9.5), 'bilirubin': (0.0, 50.0),
          'neutrophil_ap': (0.0, 80.0), 'bicarbonate_arterial': (2.0, 45.0), 'glucose': (5.0, 800.0), 'calcium': (4.0, 18.0),
          'chloride': (70.0, 130.0), 'sodium': (105.0, 185.0), 'c_reactive_protein': (0.0, 600.0),
          'carbon_dioxide_venous': (10.0, 70.0), 'oxygen_venous': (10.0, 85.0), 'albumin_venous': (0.5, 7.0),
          'bicarbonate_venous': (2.0, 45.0), 'oxygen_arterial': (20.0, 650.0), 'carbon_dioxide_arterial': (10.0, 120.0),
          'interleukin6': (0.0, 50000.0), 'magnesium': (0.2, 8.0), 'prothrombin_time': (5.0, 120.0),
          'procalcitonin': (0.0, 500.0), 'lactate': (0.2, 25.0), 'creatinine': (0.0, 10.0), 'fibrinogen': (20.0, 1200.0),
          'bilirubin_measurement': (0.0, 50.0), 'thromboplastin_time': (5.0, 120.0), 'activated': (20.0, 200.0),
          'white_blood': (0.0, 120.0), 'platelet': (1.0, 2000.0), 'white_cell': (0.0, 120.0), 'venous_ph': (6.5, 7.65),
          'd_dimer_level': (0.0, 50.0), 'arterial_ph': (6.5, 7.65), 'hemoglobin': (3.0, 28.0), 'ionised_calcium': (0.3, 3.5),
          'systolic_pressure': (20.0, 220.0), 'diastolic_pressure': (10.0, 140.0), 'resp_rate': (0.0, 120.0),
          'heart_rate': (20.0, 300.0), 'oxygen_periphery': (50.0, 102.0), 'oxygen_total': (50.0, 102.0),
          'l_pupil_diameter': (0.5, 10.0), 'r_pupil_diameter': (0.5, 10.0), 'glasgow': (3.0, 15.0)}

# Kaggle column names -> short snake_case names
LAB_RENAME: dict[str, str] = {
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
MEDS_RENAME: dict[str, str] = {
               'Systolic blood pressure': 'systolic_pressure', 'Diastolic blood pressure': 'diastolic_pressure',
               'Body temperature': 'temperature', 'Respiratory rate': 'resp_rate', 'Heart rate': 'heart_rate',
               'Measurement of oxygen saturation at periphery': 'oxygen_periphery',
               'Oxygen/Gas total [Pure volume fraction] Inhaled gas': 'oxygen_total'}


# ---------------------------------------------------------------- 1. hourly data
def load_hourly() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Load and clean the hourly Kaggle tables.

    Returns:
        clin: one row per (person_id, datetime) with all clinical values (labs, vitals, observations).
        demo: one row per patient with age in months and sex.
        tl: per-patient timeline with admission, last observation and sepsis onset (NaT if none).
        value_cols: names of the clinical value columns in `clin`.
    """
    rd: Callable[[str], pd.DataFrame] = lambda f: pd.read_csv(os.path.join(DATA, f)).drop_duplicates()
    lab = rd('SepsisLabel_train.csv')
    labs = rd('measurement_lab_train.csv').rename(columns=LAB_RENAME)
    meds = rd('measurement_meds_train.csv').rename(columns=MEDS_RENAME)
    obs = rd('measurement_observation_train.csv')
    demo = rd('person_demographics_episode_train.csv')
    for d in (lab, labs, meds, obs):
        d['datetime'] = pd.to_datetime(d.pop('measurement_datetime'), errors='coerce')

    obs = obs[['person_id', 'datetime', 'Left pupil Diameter Auto', 'Right pupil Diameter Auto', 'Glasgow coma scale',
               'Capillary refill [Time]', 'Pulse', 'Arterial pulse pressure']].rename(columns={
        'Left pupil Diameter Auto': 'l_pupil_diameter', 'Right pupil Diameter Auto': 'r_pupil_diameter',
        'Glasgow coma scale': 'glasgow', 'Capillary refill [Time]': 'capillary', 'Pulse': 'pulse',
        'Arterial pulse pressure': 'art_pressure'})
    obs['capillary'] = obs['capillary'].map({'Normal capillary filling': 'normal', 'Increased capillary filling time': 'increased',
                                             'Decreased capillary filling time': 'decreased'})
    # categorical -> 0/1 indicators; NaN when the observation was not taken in that hour
    for col in ['capillary', 'pulse', 'art_pressure']:
        dummies = pd.get_dummies(obs[col], prefix=col, dtype=float)
        dummies[obs[col].isna()] = np.nan
        obs = pd.concat([obs.drop(columns=col), dummies], axis=1)

    keep: Callable[[pd.DataFrame], pd.DataFrame] = lambda d: d.drop(columns=[c for c in ['visit_occurrence_id'] if c in d.columns])
    clin = (keep(labs).merge(keep(meds), on=['person_id', 'datetime'], how='outer')
            .merge(keep(obs), on=['person_id', 'datetime'], how='outer'))
    clin = clin.groupby(['person_id', 'datetime'], as_index=False).mean(numeric_only=True)
    for col, (lo, hi) in LIMITS.items():
        if col in clin.columns:
            clin.loc[(clin[col] < lo) | (clin[col] > hi), col] = np.nan
    value_cols = [c for c in clin.columns if c not in ['person_id', 'datetime']]
    clin = clin.dropna(subset=value_cols, how='all')

    demo['visit_start_date'] = pd.to_datetime(demo['visit_start_date'], errors='coerce')
    demo = demo.sort_values('visit_start_date').drop_duplicates('person_id', keep='last')
    demo = pd.DataFrame({'person_id': demo['person_id'], 'age_in_months': demo['age_in_months'],
                         'male': (demo['gender'] == 'MALE').astype(int)})

    # per-patient timeline (built from the label table, which is hourly)
    lab = lab.sort_values(['person_id', 'datetime'])
    tl = lab.groupby('person_id').agg(admission=('datetime', 'min'), last_obs=('datetime', 'max')).reset_index()
    onset = lab[lab['SepsisLabel'] == 1].groupby('person_id')['datetime'].min().rename('onset').reset_index()
    tl = tl.merge(onset, on='person_id', how='left')
    return clin, demo, tl, value_cols


# ---------------------------------------------------------------- 2. cut-off points
def make_cutoffs(tl: pd.DataFrame, h: int, seed: int) -> pd.DataFrame:
    """Choose one prediction cut-off per patient for an h-hour horizon.

    Positives: cut-off = onset - h (kept only if it falls at least MIN_OBS_H hours after admission).
    Negatives: cut-off sampled from the positives' hours-since-admission distribution, constrained so the
    patient is still observed for another h hours afterwards (falls back to uniform if no match).

    Returns a frame with person_id, admission, cutoff and label.
    """
    rng = np.random.default_rng(seed)
    pos = tl[tl['onset'].notna()].copy()
    pos['cutoff'] = pos['onset'] - pd.Timedelta(hours=h)
    pos['label'] = 1
    pos = pos[pos['cutoff'] >= pos['admission'] + pd.Timedelta(hours=MIN_OBS_H)]
    ref_h = ((pos['cutoff'] - pos['admission']).dt.total_seconds() / 3600).values  # reference distribution

    neg = tl[tl['onset'].isna()].copy()
    max_h = (neg['last_obs'] - neg['admission']).dt.total_seconds() / 3600 - h   # must stay observed h more hours
    cut_h: list[float] = []
    for mh in max_h.values:
        if not np.isfinite(mh) or mh < MIN_OBS_H:   # invalid dates or stay too short
            cut_h.append(np.nan); continue
        draw = np.nan
        for _ in range(30):                       # match the positives' distribution
            c = rng.choice(ref_h)
            if MIN_OBS_H <= c <= mh:
                draw = c; break
        if np.isnan(draw):                        # fallback: uniform over the feasible interval
            draw = rng.uniform(MIN_OBS_H, mh)
        cut_h.append(draw)
    neg['cutoff'] = neg['admission'] + pd.to_timedelta(cut_h, unit='h')
    neg['label'] = 0
    neg = neg[neg['cutoff'].notna()]
    return pd.concat([pos, neg], ignore_index=True)[['person_id', 'admission', 'cutoff', 'label']]


# ---------------------------------------------------------------- 3. aggregation up to the cut-off
def build_features(clin: pd.DataFrame, demo: pd.DataFrame, cuts: pd.DataFrame,
                   value_cols: list[str]) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Summarize each patient's history between admission and cut-off into one row.

    Only measurements with datetime <= cutoff are used, so no feature can see the future.
    Patients without any clinical measurement before the cut-off are dropped (same rule for both classes).

    Returns:
        X: feature matrix indexed by person_id.
        y: binary label (1 = sepsis h hours after the cut-off).
        meta: the cut-off timestamp per patient (used for the temporal split).
    """
    d = clin.merge(cuts, on='person_id', how='inner')
    d = d[(d['datetime'] >= d['admission']) & (d['datetime'] <= d['cutoff'])].sort_values(['person_id', 'datetime'])
    g = d.groupby('person_id')
    feats: dict[str, pd.Series] = {}
    last = g[value_cols].last()                 # last() skips NaN -> last valid value (implicit forward fill)
    feats.update({f'{c}_last': last[c] for c in value_cols})
    for fn in ('mean', 'min', 'max'):
        a = g[value_cols].agg(fn)
        feats.update({f'{c}_{fn}': a[c] for c in value_cols})
    cnt = g[value_cols].count()
    feats.update({f'{c}_n': cnt[c] for c in value_cols})
    # hours since the last measurement of each variable
    cutoff_by_pid = d.groupby('person_id')['cutoff'].first()
    for c in value_cols:
        lt = d.loc[d[c].notna()].groupby('person_id')['datetime'].max()
        feats[f'{c}_h_since'] = ((cutoff_by_pid - lt).dt.total_seconds() / 3600).reindex(cutoff_by_pid.index)
    # mean over the last 24h and trend (24h mean - overall mean)
    d24 = d[d['datetime'] > d['cutoff'] - pd.Timedelta(hours=24)]
    m24 = d24.groupby('person_id')[value_cols].mean()
    for c in value_cols:
        feats[f'{c}_mean24h'] = m24[c].reindex(cutoff_by_pid.index)
        feats[f'{c}_trend24h'] = feats[f'{c}_mean24h'] - feats[f'{c}_mean']
    X = pd.DataFrame(feats)
    X['n_rows_observed'] = g.size()
    base = cuts.set_index('person_id')
    X['hours_since_admission'] = ((base.loc[X.index, 'cutoff'] - base.loc[X.index, 'admission']).dt.total_seconds() / 3600).values
    X = X.join(demo.set_index('person_id'), how='left')
    # patients with no clinical measurement before the cut-off never enter X (same rule for both classes)
    y = base.loc[X.index, 'label']
    meta = base.loc[X.index, ['cutoff']]
    # drop value features that are almost always missing (count features `_n` are kept)
    miss = X.isna().mean()
    X = X.drop(columns=[c for c in X.columns if miss[c] >= MISS_DROP and not c.endswith('_n')])
    return X, y.astype(int), meta


# ---------------------------------------------------------------- 4. models
cw = [{0: 1, 1: 10}, {0: 1, 1: 30}, {0: 1, 1: 50}, {0: 1, 1: 100}]
PARAMS: dict[str, dict[str, list[Any]]] = {
    'LGBM': dict(n_estimators=[200, 400, 800], num_leaves=[7, 15, 31], learning_rate=[0.01, 0.03, 0.05],
                 subsample=[0.6, 0.8, 1.0], subsample_freq=[1], colsample_bytree=[0.3, 0.5, 0.8],
                 min_child_samples=[5, 10, 20, 40], reg_lambda=[0, 1, 5, 10], class_weight=cw),
    'XGB': dict(n_estimators=[200, 400, 800], max_depth=[2, 3, 4], learning_rate=[0.01, 0.03, 0.05],
                subsample=[0.6, 0.8, 1.0], colsample_bytree=[0.3, 0.5, 0.8], min_child_weight=[1, 3, 5],
                reg_lambda=[1, 5, 10], scale_pos_weight=[10, 30, 50, 100]),
    'HGB': dict(max_iter=[200, 400, 800], max_leaf_nodes=[7, 15, 31], learning_rate=[0.01, 0.03, 0.05],
                max_depth=[3, 4, None], min_samples_leaf=[5, 10, 20, 40], l2_regularization=[0, 1, 5], class_weight=cw),
}


def models() -> dict[str, BaseEstimator]:
    """Candidate models.

    Light version: HGB removed (worst model in every v1 scenario) and one thread per model,
    leaving parallelism to cross-validation only (avoids thread contention).
    """
    return {'LGBM': LGBMClassifier(objective='binary', random_state=SEED, n_jobs=1, verbose=-1),
            'XGB': XGBClassifier(objective='binary:logistic', eval_metric='aucpr', random_state=SEED, n_jobs=1)}


def thr_rule(y: np.ndarray | pd.Series, p: np.ndarray, min_prec: float = MIN_PRECISION) -> float | None:
    """Return the threshold with the highest recall among those with precision >= min_prec (same rule as v1).

    Returns None if no threshold reaches the required precision.
    """
    prec, rec, thr = precision_recall_curve(y, p)
    ok = np.where(prec[:-1] >= min_prec)[0]
    if len(ok) == 0:
        return None
    best = ok[np.argmax(rec[:-1][ok])]
    return float(thr[best])


def boot_ci(y: np.ndarray | pd.Series, p: np.ndarray, fn: Callable[[np.ndarray, np.ndarray], float],
            n: int = 2000) -> list[float]:
    """95% percentile bootstrap CI of metric `fn` (resamples patients; skips single-class resamples)."""
    rng = np.random.default_rng(SEED); y = np.asarray(y); p = np.asarray(p); v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            v.append(fn(y[i], p[i]))
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]


def metrics_at(y: np.ndarray | pd.Series, p: np.ndarray, t: float) -> dict[str, Any]:
    """Confusion matrix and clinical metrics at threshold t."""
    yp = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    return dict(threshold=round(t, 5), recall=round(tp / (tp + fn), 3) if tp + fn else None,
                precision=round(tp / (tp + fp), 3) if tp + fp else None,
                specificity=round(tn / (tn + fp), 3), alerts_per_true_case=round((tp + fp) / tp, 2) if tp else None,
                cm=dict(TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp)))


def tune(Xtr: pd.DataFrame, ytr: pd.Series) -> tuple[dict[str, Any], str | None, BaseEstimator | None]:
    """Randomized search (5-fold stratified CV, average precision) for each model.

    Returns per-model CV scores and params, plus the name and fitted estimator of the best model.
    """
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    out, best = {}, (-1, None, None)
    for nm, m in models().items():
        s = RandomizedSearchCV(m, PARAMS[nm], n_iter=N_ITER, cv=cv, scoring='average_precision', n_jobs=6,
                               random_state=SEED, error_score=np.nan)
        try:
            s.fit(Xtr, ytr)
        except ValueError as e:              # one model failing must not break the comparison
            print(f'  [warning] {nm} failed: {str(e).splitlines()[0][:120]}', flush=True)
            continue
        out[nm] = dict(cv_ap=round(float(s.best_score_), 3), params=s.best_params_)
        if s.best_score_ > best[0]:
            best = (s.best_score_, nm, s.best_estimator_)
    return out, best[1], best[2]


def evaluate(Xtr: pd.DataFrame, ytr: pd.Series, Xte: pd.DataFrame, yte: pd.Series, tag: str,
             figdir: str) -> tuple[dict[str, Any], BaseEstimator, str, dict[str, Any]]:
    """Tune on train, pick the threshold out-of-fold on train, then evaluate once on test.

    Saves the test precision-recall curve to `figdir` and returns the results dict, the best fitted
    estimator, its name and the per-model summary.
    """
    # columns that are fully empty or constant on train (happens in the temporal split) are dropped
    keep = [c for c in Xtr.columns if Xtr[c].nunique(dropna=True) > 1]
    Xtr, Xte = Xtr[keep], Xte[keep]
    per_model, bn, be = tune(Xtr, ytr)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof = cross_val_predict(clone(be), Xtr, ytr, cv=cv, method='predict_proba')[:, 1]
    t = thr_rule(ytr, oof)
    p = be.predict_proba(Xte)[:, 1]
    for nm in per_model:   # test PR-AUC of every model (informative only; selection was done on CV)
        m = clone(models()[nm]).set_params(**per_model[nm]['params']).fit(Xtr, ytr)
        per_model[nm]['test_ap'] = round(float(average_precision_score(yte, m.predict_proba(Xte)[:, 1])), 3)
    res = dict(tag=tag, n_train=int(len(ytr)), pos_train=int(ytr.sum()), n_test=int(len(yte)), pos_test=int(yte.sum()),
               prevalence_test=round(float(yte.mean()), 4), best_model=bn, per_model=per_model,
               test_ap=round(float(average_precision_score(yte, p)), 3), test_ap_ci=boot_ci(yte, p, average_precision_score),
               test_roc=round(float(roc_auc_score(yte, p)), 3), test_roc_ci=boot_ci(yte, p, roc_auc_score),
               operating_point_oof=metrics_at(yte, p, t) if t is not None else None)
    imp = getattr(be, 'feature_importances_', None)
    if imp is not None:
        s = pd.Series(imp, index=Xtr.columns).sort_values(ascending=False)
        res['top_features'] = [[k, round(float(v), 4)] for k, v in s.head(15).items()]
    # figure: PR curve (axis labels kept in Portuguese to match the published figures)
    pr, rc, _ = precision_recall_curve(yte, p)
    plt.figure(figsize=(5.5, 4.2))
    plt.plot(rc, pr, label=f'{bn}  PR-AUC={res["test_ap"]:.3f}')
    plt.axhline(yte.mean(), ls='--', c='red', lw=1, label=f'baseline (prevalência)={yte.mean():.3f}')
    plt.xlabel('Recall'); plt.ylabel('Precisão'); plt.title(f'Curva PR — {tag}'); plt.legend(fontsize=8)
    plt.ylim(0, 1.02); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(os.path.join(figdir, f'pr_{tag}.png'), dpi=150); plt.close()
    return res, be, bn, per_model


def repeated_cv(X: pd.DataFrame, y: pd.Series, model: BaseEstimator) -> dict[str, float]:
    """Repeated stratified 5x5 CV of average precision on the full dataset (fixed hyperparameters)."""
    rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    sc = cross_val_score(clone(model), X, y, cv=rcv, scoring='average_precision', n_jobs=6)
    return dict(mean=round(float(sc.mean()), 3), sd=round(float(sc.std()), 3),
                p05=round(float(np.percentile(sc, 5)), 3), p95=round(float(np.percentile(sc, 95)), 3))


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    figdir = os.path.join(HERE, 'figures'); os.makedirs(figdir, exist_ok=True)
    os.makedirs(os.path.join(HERE, 'bases'), exist_ok=True); os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    clin, demo, tl, value_cols = load_hourly()
    print(f'patients: {len(tl)} | with sepsis: {tl.onset.notna().sum()} | clinical variables: {len(value_cols)}', flush=True)
    results: dict[str, Any] = {'config': dict(SEED=SEED, MIN_OBS_H=MIN_OBS_H, MIN_PRECISION=MIN_PRECISION, N_ITER=N_ITER, MISS_DROP=MISS_DROP)}
    for h in (24, 48):
        cuts = make_cutoffs(tl, h, SEED + h)
        X, y, meta = build_features(clin, demo, cuts, value_cols)
        pos_total = int(tl.onset.notna().sum())
        info = dict(patients=int(len(y)), positives=int(y.sum()), prevalence=round(float(y.mean()), 4),
                    positives_excluded=pos_total - int(y.sum()), n_features=int(X.shape[1]))
        print(f'\n=== {h}h: {info}', flush=True)
        pd.concat([X, y.rename('label'), meta], axis=1).to_csv(os.path.join(HERE, 'bases', f'base_{h}h_v2.csv'))

        # (a) stratified 70/30 split (same design as v1)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
        r_split, best_est, bn, per_model = evaluate(Xtr, ytr, Xte, yte, f'{h}h_split', figdir)
        print(json.dumps({k: r_split[k] for k in ('best_model', 'test_ap', 'test_ap_ci', 'test_roc', 'operating_point_oof')}), flush=True)

        # (b) repeated 5x5 CV on the whole dataset, with the selected hyperparameters
        rcv = repeated_cv(X, y, clone(models()[bn]).set_params(**per_model[bn]['params']))
        print('repeated CV:', rcv, flush=True)

        # (c) temporal validation: oldest 70% (by cut-off date) -> train; newest 30% -> test
        order = meta['cutoff'].sort_values()
        split_at = order.iloc[int(len(order) * 0.7)]
        tr_idx = meta.index[meta['cutoff'] < split_at]; te_idx = meta.index[meta['cutoff'] >= split_at]
        r_time, _, _, _ = evaluate(X.loc[tr_idx], y.loc[tr_idx], X.loc[te_idx], y.loc[te_idx], f'{h}h_temporal', figdir)
        r_time['split_date'] = str(split_at)
        print(json.dumps({k: r_time[k] for k in ('best_model', 'test_ap', 'test_ap_ci', 'test_roc', 'operating_point_oof')}), flush=True)

        results[f'{h}h'] = dict(dataset=info, split=r_split, repeated_cv=rcv, temporal=r_time)
        with open(os.path.join(HERE, 'results', 'results_v2.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=1, ensure_ascii=False, default=str)
    print('\nDONE')
