"""
worker_classify.py - Standalone classification worker
Reads CSV paths written by app.py, outputs JSON result to stdout.
Usage: python worker_classify.py <temp_csv_path> <vib_csv_path>
"""

import sys
import json
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd

if __name__ == '__main__':
    temp_path = sys.argv[1]
    vib_path  = sys.argv[2]

    try:
        df_temp = pd.read_csv(temp_path)
        df_vib  = pd.read_csv(vib_path)

        from classify import classify_bearing
        result = classify_bearing(df_temp, df_vib)

        # IMPORTANT: print JSON as the very last line to stdout
        print(json.dumps({'success': True, 'classifications': result}))
    except Exception as e:
        import traceback
        print(json.dumps({'success': False, 'error': str(e), 'traceback': traceback.format_exc()}))
        sys.exit(1)