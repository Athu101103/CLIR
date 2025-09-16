#!/usr/bin/env python3
"""
Run all available experiments (QT, DR, Zero-shot), aggregate results, and save reports.
Usage:
  python experiments/run_all_experiments.py [--quick]
--quick: run on small subsets to finish fast
"""
import sys
import os
import argparse
import json
from datetime import datetime
import logging

# Ensure src and project root are importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(ROOT)
if ROOT not in sys.path:
    sys.path.append(ROOT)          # so 'src.*' absolute imports work inside modules
src_path = os.path.join(ROOT, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)      # so 'qt_framework.*' etc. can be imported here

import pandas as pd

from qt_framework.qt_runner import QTRunner
from dr_framework.dr_runner import DRRunner
from zero_shot_framework.zs_runner import ZSRunner
# DT runner may not have all models fully wired yet; import guarded
try:
    from dt_framework.dt_runner import DTRunner
    HAS_DT = True
except Exception:
    HAS_DT = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("run_all")


def to_rows(tag: str, result: dict) -> list:
    rows = []
    eval_results = result.get('evaluation_results')
    if not eval_results:
        rows.append({
            'experiment': tag,
            'model_name': result.get('model_name', tag),
            'note': result.get('error', 'no evaluation')
        })
        return rows
    model_name = result.get('model_name', tag)
    for metric, values in eval_results.items():
        for k, v in values.items():
            rows.append({
                'experiment': tag,
                'model_name': model_name,
                'metric': metric,
                'k': k,
                'value': v,
            })
    return rows


def save_reports(all_results: dict, out_dir: str = 'results') -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_path = os.path.join(out_dir, f'all_results_{ts}.json')
    csv_path = os.path.join(out_dir, f'all_results_{ts}.csv')

    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    rows = []
    for exp_name, exp_results in all_results.items():
        for key, res in exp_results.items():
            rows.extend(to_rows(key, res))
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)

    logger.info(f"Saved reports: {json_path}, {csv_path}")
    return csv_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Use small subsets for a fast run')
    args = parser.parse_args()

    all_results = {}

    # QT
    logger.info('Running QT experiments...')
    qt = QTRunner()
    qt.load_data()
    if args.quick:
        qt.train_data = qt.train_data.select(range(min(200, len(qt.train_data))))
        qt.validation_data = qt.validation_data.select(range(min(100, len(qt.validation_data))))
        qt.test_data = qt.test_data.select(range(min(50, len(qt.test_data))))
    qt.run_bm25_experiment()
    qt.run_xlm_roberta_experiment()
    all_results['qt'] = qt.results

    # DR
    logger.info('Running DR experiments...')
    dr = DRRunner()
    dr.load_data()
    if args.quick:
        dr.train_data = dr.train_data.select(range(min(200, len(dr.train_data))))
        dr.validation_data = dr.validation_data.select(range(min(100, len(dr.validation_data))))
        dr.test_data = dr.test_data.select(range(min(50, len(dr.test_data))))
    dr.run_mdpr_experiment()
    dr.run_xlm_roberta_experiment()
    try:
        from dr_framework.multilingual_e5_dr import MultilingualE5DRModel  # noqa: F401
        dr.run_e5_experiment()
    except Exception as e:
        logger.warning(f"Skipping DR E5 due to error: {e}")
    all_results['dr'] = dr.results

    # Zero-shot
    logger.info('Running Zero-shot experiments...')
    zs = ZSRunner()
    zs.load_data()
    if args.quick:
        zs.train_data = zs.train_data.select(range(min(200, len(zs.train_data))))
        zs.validation_data = zs.validation_data.select(range(min(100, len(zs.validation_data))))
        zs.test_data = zs.test_data.select(range(min(50, len(zs.test_data))))
    zs.run_all_experiments()
    all_results['zero_shot'] = zs.results

    # DT (optional subset)
    if HAS_DT:
        try:
            logger.info('Running DT experiments (available subset)...')
            dt = DTRunner()
            dt.load_data()
            if args.quick:
                dt.train_data = dt.train_data.select(range(min(200, len(dt.train_data))))
                dt.validation_data = dt.validation_data.select(range(min(100, len(dt.validation_data))))
                dt.test_data = dt.test_data.select(range(min(50, len(dt.test_data))))
            dt.run_bm25_experiment()
            dt.run_contriever_experiment()
            all_results['dt'] = dt.results
        except Exception as e:
            logger.warning(f"DT experiments skipped due to error: {e}")

    # Save
    save_reports(all_results)

    logger.info('All experiments completed.')


if __name__ == '__main__':
    main()
