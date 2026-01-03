"""
WebSocket Server for Real-Time Sign ↔ Voice Translation
- Receives recognized speech from Python (sign language → text)
- Sends animation updates to web clients
- Receives text from web clients and forwards to Python for TTS playback

Run: python websocket_server.py
Then connect web client to ws://localhost:5000 and Python recognizer to ws://localhost:5000/sign
"""

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room, disconnect
import logging

app = Flask(__name__, static_folder='web', static_url_path='', template_folder='web')
app.config['SECRET_KEY'] = 'sign-voice-secret-key-change-in-production'
socketio = SocketIO(app, cors_allowed_origins="*")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Track connected clients
connected_clients = {
    'web': [],      # web browser clients
    'python': None  # python recognizer connection
}

# Global metrics instance (shared with main.py)
global_metrics = None

def set_global_metrics(metrics):
    """Set the global metrics instance from main.py"""
    global global_metrics
    global_metrics = metrics

@app.route('/')
def index():
    """Serve the web demo HTML"""
    return app.send_static_file('index.html')

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

@socketio.on('register_client')
def handle_register_client(data=None):
    """Client registration (replaces handle_connect for custom logic)"""
    client_type = data.get('type', 'web') if isinstance(data, dict) else 'web'
    logger.info(f"Client registered: {client_type}")
    sid = request.sid
    
    if client_type == 'python':
        connected_clients['python'] = sid
        emit('status', {'message': 'Python recognizer connected'})
        # Broadcast to all web clients that Python is online
        socketio.emit('status', {'message': 'Python recognizer online', 'python_online': True}, broadcast=True, skip_sid=sid)
    else:
        connected_clients['web'].append(sid)
        logger.info(f"Web client registered: sid={sid}, total_web_clients={len(connected_clients['web'])}")
        emit('status', {'message': 'Web client registered', 'python_online': connected_clients['python'] is not None})

@socketio.on('connect')
def handle_connect(data=None):
    """Client connects - just log it"""
    logger.info(f"Client connected from {request.remote_addr}")
    emit('status', {'message': 'Connected to WebSocket server'})
    if connected_clients['python']:
        socketio.emit('web_client_joined', {}, to=connected_clients['python'])

@socketio.on('disconnect')
def handle_disconnect():
    """Client disconnects"""
    # Use Flask's request proxy to get the socket id
    sid = request.sid
    
    if sid == connected_clients.get('python'):
        logger.info("Python recognizer disconnected")
        connected_clients['python'] = None
        # Broadcast to web clients that python went offline
        socketio.emit('status', {'message': 'Python recognizer offline', 'python_online': False}, broadcast=True)
    elif sid in connected_clients['web']:
        logger.info("Web client disconnected")
        connected_clients['web'].remove(sid)
        if connected_clients['python']:
            socketio.emit('web_client_count', {'count': len(connected_clients['web'])}, to=connected_clients['python'])
        logger.info(f"Web client removed: sid={sid}, remaining_web_clients={len(connected_clients['web'])}")

@socketio.on('sign_recognized')
def handle_sign_recognized(data):
    """
    Python sends recognized sign language text.
    Broadcast to all web clients to trigger animation.
    """
    text = data.get('text', '')
    logger.info(f"[sign_recognized] Received from Python: '{text}'")
    logger.info(f"[sign_recognized] Broadcasting to web clients: '{text}'")
    
    # Forward to all web clients using correct Flask-SocketIO syntax
    # broadcast=True means send to all connected clients
    socketio.emit('animate_text', {'text': text}, broadcast=True)
    socketio.emit('sign_recognized', {'text': text}, broadcast=True)

@socketio.on('web_animate_request')
def handle_web_animate_request(data):
    """
    Web client requests animation (user typed text).
    Optionally send to Python for TTS.
    """
    text = data.get('text', '')
    logger.info(f"Animation request from web: {text}")
    
    # Notify Python to play TTS (optional)
    if connected_clients['python']:
        socketio.emit('speak', {'text': text}, to=connected_clients['python'])

@socketio.on('python_ready')
def handle_python_ready():
    """Python recognizer signals it's ready"""
    logger.info("Python recognizer ready")
    connected_clients['python'] = request.sid
    emit('status', {'message': 'Python ready, web clients: ' + str(len(connected_clients['web']))})

if __name__ == '__main__':
    logger.info("Starting WebSocket server on ws://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
