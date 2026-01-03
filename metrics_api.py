"""
Metrics API Server
Provides REST API endpoints for performance metrics
"""

from flask import Flask, jsonify, render_template_string
from performance_metrics import PerformanceMetrics
import threading
import time

app = Flask(__name__, static_folder='web', static_url_path='')

# Global metrics instance (shared with main.py)
global_metrics = None

def set_global_metrics(metrics):
    """Set the global metrics instance from main.py"""
    global global_metrics
    global_metrics = metrics

@app.route('/')
def index():
    """Serve the main web interface"""
    return app.send_static_file('index.html')

@app.route('/performance_dashboard.html')
def performance_dashboard():
    """Serve the performance dashboard"""
    return app.send_static_file('performance_dashboard.html')

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """
    Get current performance metrics as JSON
    
    Returns:
        JSON with all performance metrics
    """
    if global_metrics is None:
        return jsonify({
            'error': 'Metrics not initialized',
            'fps': {'current': 0, 'min': 0, 'max': 0, 'target': 30},
            'latency_ms': {'current': 0, 'min': 0, 'max': 0, 'target': 33},
            'confidence': {'current': 0, 'min': 0, 'max': 0, 'target': 0.8},
            'accuracy': {'overall': 0, 'correct_predictions': 0, 'total_predictions': 0},
            'resources': {'memory_mb': 0, 'cpu_percent': 0},
            'session': {'duration_seconds': 0, 'total_frames': 0, 'total_gestures': 0},
            'gesture_stats': {}
        }), 200
    
    try:
        summary = global_metrics.get_summary()
        
        # Add gesture stats
        gesture_stats = {}
        for gesture, stats in global_metrics.gesture_stats.items():
            gesture_stats[gesture] = {
                'attempts': stats['attempts'],
                'correct': stats['correct'],
                'accuracy': stats['accuracy'],
                'confidences': stats['confidences']
            }
        
        response = {
            'fps': summary['fps'],
            'latency_ms': summary['latency_ms'],
            'confidence': summary['confidence'],
            'accuracy': summary['accuracy'],
            'resources': summary['resources'],
            'session': summary['session'],
            'gesture_stats': gesture_stats
        }
        
        return jsonify(response), 200
    except Exception as e:
        print(f"Error getting metrics: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/metrics/summary', methods=['GET'])
def get_metrics_summary():
    """Get a text summary of metrics"""
    if global_metrics is None:
        return jsonify({'error': 'Metrics not initialized'}), 500
    
    try:
        report = global_metrics.get_detailed_report()
        return jsonify({'report': report}), 200
    except Exception as e:
        print(f"Error getting metrics summary: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/metrics/reset', methods=['POST'])
def reset_metrics():
    """Reset all metrics"""
    if global_metrics is None:
        return jsonify({'error': 'Metrics not initialized'}), 500
    
    try:
        global_metrics.reset()
        return jsonify({'message': 'Metrics reset successfully'}), 200
    except Exception as e:
        print(f"Error resetting metrics: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/metrics/gestures', methods=['GET'])
def get_gesture_metrics():
    """Get gesture-specific metrics"""
    if global_metrics is None:
        return jsonify({'error': 'Metrics not initialized'}), 500
    
    try:
        top_gestures = global_metrics.get_top_gestures(10)
        worst_gestures = global_metrics.get_worst_gestures(5)
        
        return jsonify({
            'top_gestures': [{'name': g, 'accuracy': a} for g, a in top_gestures],
            'worst_gestures': [{'name': g, 'accuracy': a} for g, a in worst_gestures],
            'total_gestures': len(global_metrics.gesture_stats)
        }), 200
    except Exception as e:
        print(f"Error getting gesture metrics: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'metrics_available': global_metrics is not None
    }), 200

if __name__ == '__main__':
    # Run Flask server
    app.run(host='0.0.0.0', port=5001, debug=False)
