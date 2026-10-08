# app.py

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd
import os
import traceback
import time
import json
import subprocess
import sys
import tempfile
from pathlib import Path

app = Flask(__name__)
CORS(app)

# ==================== CONFIGURATION ====================

class Config:
    ORIGINAL_FREQ   = 25600
    TARGET_FREQ     = 1000
    WINDOW_SIZE     = 2000
    OVERLAP         = 0.5
    SEQUENCE_LENGTH = 10

config = Config()

# ==================== HEALTH STATE ====================

def classify_health_state(rul):
    if rul > 60:
        return 'Healthy', 'Low'
    elif rul > 40:
        return 'Warning', 'Medium'
    elif rul > 20:
        return 'Severe', 'High'
    else:
        return 'Critical', 'Critical'

# ==================== SUBPROCESS HELPERS ====================

def _launch(script_name, *args):
    """
    Launch a worker script as an independent OS process.
    Returns Popen immediately — caller starts both before waiting on either.
    """
    return subprocess.Popen(
        [sys.executable, script_name, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=os.getcwd(),
    )


def run_parallel(temp_bytes, vib_bytes, model_type):
    """
    Write CSVs to named temp files, then spawn RUL + classification workers
    as two separate OS processes simultaneously.

    Wall-clock time = max(rul_time, classify_time) instead of their sum.

    Returns:
        y_pred                 : np.ndarray
        classification_results : dict
        elapsed                : float (seconds)
    """
    tmp_temp_file = tempfile.NamedTemporaryFile(
        mode='wb', suffix='_temp.csv', delete=False
    )
    tmp_vib_file = tempfile.NamedTemporaryFile(
        mode='wb', suffix='_vib.csv', delete=False
    )
    try:
        tmp_temp_file.write(temp_bytes)
        tmp_vib_file.write(vib_bytes)
        tmp_temp_file.close()
        tmp_vib_file.close()

        temp_path = tmp_temp_file.name
        vib_path  = tmp_vib_file.name

        t0 = time.perf_counter()

        print("\n" + "="*60)
        print("LAUNCHING PARALLEL WORKERS")
        print("="*60)

        # ── Start BOTH processes before waiting on either ──────────────────
        proc_rul = _launch('worker_rul.py',      temp_path, vib_path, model_type)
        proc_clf = _launch('worker_classify.py', temp_path, vib_path)
        # ───────────────────────────────────────────────────────────────────

        print(f"  ✓ RUL worker       started  (PID {proc_rul.pid})")
        print(f"  ✓ Classify worker  started  (PID {proc_clf.pid})")
        print("  ⏳ Both running simultaneously...")

        # Block until BOTH finish — they are already running in parallel
        rul_stdout, rul_stderr = proc_rul.communicate()
        clf_stdout, clf_stderr = proc_clf.communicate()

        elapsed = time.perf_counter() - t0
        print(f"  ✅ Both workers done in {elapsed:.2f}s")

        # ── Parse RUL result (fatal on failure) ───────────────────────────
        try:
            last_line = rul_stdout.strip().split('\n')[-1]
            rul_data  = json.loads(last_line)
            if not rul_data.get('success'):
                raise RuntimeError(rul_data.get('error', 'Unknown RUL error'))
            y_pred = np.array(rul_data['predictions'])
        except Exception as e:
            print(f"\n  ✗ RUL worker stderr:\n{rul_stderr}")
            raise RuntimeError(f"RUL worker failed: {e}") from e

        # ── Parse classification result (non-fatal) ───────────────────────
        classification_results = {}
        try:
            last_line = clf_stdout.strip().split('\n')[-1]
            clf_data  = json.loads(last_line)
            if clf_data.get('success'):
                classification_results = clf_data.get('classifications', {})
            else:
                print(f"  ✗ Classification worker error: {clf_data.get('error')}")
                if clf_stderr:
                    print(clf_stderr)
        except Exception as e:
            print(f"  ✗ Could not parse classification output: {e}")
            if clf_stderr:
                print(clf_stderr)

        return y_pred, classification_results, elapsed

    finally:
        # Always clean up temp files even if something crashes
        try:
            os.unlink(tmp_temp_file.name)
        except Exception:
            pass
        try:
            os.unlink(tmp_vib_file.name)
        except Exception:
            pass

# ==================== ROUTES ====================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/trymodel')
def trymodel():
    return render_template('trymodel.html')

@app.route('/faq')
def faq():
    return render_template('faq.html')

@app.route('/about-us')
def about_us():
    return render_template('about-us.html')

@app.route('/about-modal')
def about_modal():
    return render_template('about-modal.html')

@app.route('/modal-evaluation')
def modal_evaluation():
    return render_template('modal-evaluation.html')

@app.route('/data-collection')
def data_collection():
    return render_template('data-collection.html')

@app.route('/maintenance-evolution')
def maintenance_evolution():
    return render_template('maintenance-evolution.html')

@app.route('/future-scope')
def future_scope():
    return render_template('future-scope.html')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status' : 'healthy',
        'message': 'System ready for predictions'
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request — RUL + classification run truly in parallel."""
    try:
        model_type = request.form.get('model_type', 'misalign')

        if 'temp_file' not in request.files or 'vib_file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'Both temperature and vibration files are required'
            }), 400

        temp_file = request.files['temp_file']
        vib_file  = request.files['vib_file']

        if temp_file.filename == '' or vib_file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400

        # Read file bytes once — workers read from temp files on disk
        temp_bytes = temp_file.read()
        vib_bytes  = vib_file.read()

        print(f"\n{'='*60}")
        print(f"NEW PREDICTION REQUEST  (model: {model_type})")
        print(f"{'='*60}")

        # ── TRUE parallel execution via OS subprocesses ────────────────────
        y_pred, classification_results, elapsed = run_parallel(
            temp_bytes, vib_bytes, model_type
        )
        # ───────────────────────────────────────────────────────────────────

        # Build response
        sequence_indices    = np.arange(len(y_pred))
        rul_minutes         = y_pred.tolist()
        rul_hours           = (y_pred / 60).tolist()
        health_states       = []
        maintenance_urgency = []
        for rul in y_pred:
            hs, urg = classify_health_state(rul)
            health_states.append(hs)
            maintenance_urgency.append(urg)

        healthy_count  = int(sum(1 for s in health_states if s == 'Healthy'))
        warning_count  = int(sum(1 for s in health_states if s == 'Warning'))
        severe_count   = int(sum(1 for s in health_states if s == 'Severe'))
        critical_count = int(sum(1 for s in health_states if s == 'Critical'))

        stats_data = {
            'total_sequences'     : int(len(y_pred)),
            'mean_rul_minutes'    : float(np.mean(y_pred)),
            'median_rul_minutes'  : float(np.median(y_pred)),
            'mean_rul_hours'      : float(np.mean(y_pred) / 60),
            'min_rul_minutes'     : float(np.min(y_pred)),
            'max_rul_minutes'     : float(np.max(y_pred)),
            'std_rul_minutes'     : float(np.std(y_pred)),
            'variance_rul_minutes': float(np.var(y_pred)),
            'range_rul_minutes'   : float(np.ptp(y_pred)),
            'q1_rul_minutes'      : float(np.percentile(y_pred, 25)),
            'q3_rul_minutes'      : float(np.percentile(y_pred, 75)),
            'iqr_rul_minutes'     : float(np.percentile(y_pred, 75) - np.percentile(y_pred, 25)),
            'healthy_count'       : healthy_count,
            'warning_count'       : warning_count,
            'severe_count'        : severe_count,
            'critical_count'      : critical_count,
            'model_type'          : model_type,
            'inference_time_s'    : round(elapsed, 2),
        }

        print(f"\n{'='*60}")
        print(f"REQUEST COMPLETED  —  {elapsed:.2f}s total wall-clock time")
        print(f"{'='*60}\n")

        return jsonify({
            'success'        : True,
            'predictions'    : {
                'sequence_indices'   : sequence_indices.tolist(),
                'rul_minutes'        : rul_minutes,
                'rul_hours'          : rul_hours,
                'health_states'      : health_states,
                'maintenance_urgency': maintenance_urgency,
            },
            'statistics'     : stats_data,
            'classifications': classification_results,
            'message'        : f'Predictions generated successfully using {model_type} model',
        })

    except Exception as e:
        print(f"\n{'='*60}")
        print("ERROR DURING PREDICTION")
        print(f"{'='*60}")
        print(traceback.format_exc())
        print(f"{'='*60}\n")
        return jsonify({
            'success': False,
            'error'  : str(e),
            'message': 'An error occurred during prediction'
        }), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)