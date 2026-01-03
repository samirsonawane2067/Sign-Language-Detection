# Performance Metrics Web Dashboard - Complete Guide

## Overview

A beautiful, interactive web dashboard has been created to display performance metrics in real-time. Users can click a "Performance" button to view detailed metrics, charts, and gesture statistics.

## Features

### 🎨 Beautiful UI
- Modern gradient design with purple theme
- Responsive layout (works on desktop, tablet, mobile)
- Color-coded status indicators (Green/Yellow/Red)
- Smooth animations and transitions
- Professional card-based layout

### 📊 Real-Time Metrics Display
- **FPS (Frames Per Second)** - Video processing speed
- **Latency** - Prediction processing time
- **Confidence** - Model confidence scores
- **Accuracy** - Overall recognition accuracy
- **Memory Usage** - RAM consumption
- **CPU Usage** - Processor utilization

### 📈 Interactive Charts
- FPS trend chart
- Latency trend chart
- Confidence trend chart
- Accuracy trend chart
- Auto-updating every 2 seconds
- Last 20 data points displayed

### 🎯 Gesture Performance
- Per-gesture accuracy display
- Confidence scores per gesture
- Visual progress bars
- Sorted by accuracy
- Attempt counts

### 📋 Session Information
- Session duration
- Total frames processed
- Total gestures recognized
- Memory usage
- CPU usage

## Files Created

### 1. `performance_dashboard.html` (400+ lines)
- Main dashboard interface
- Beautiful responsive design
- Real-time metrics display
- Interactive charts using Chart.js
- Auto-refresh functionality

### 2. `metrics_api.py` (150+ lines)
- Flask REST API server
- Endpoints for metrics data
- Health check endpoint
- Gesture metrics endpoint
- Reset functionality

### 3. Modified Files
- `web/index.html` - Added Performance button
- `web/app.js` - Added button click handler

## How to Use

### Step 1: Start the Main Application
```bash
python main.py run --camera 0
```

### Step 2: Start the Metrics API Server
In a new terminal:
```bash
python metrics_api.py
```

The API server will run on `http://localhost:5001`

### Step 3: Open Web Interface
Open your browser and go to:
```
http://localhost:5000
```

You should see the main web interface with a new "📊 Performance" button.

### Step 4: Click Performance Button
Click the "📊 Performance" button to open the dashboard.

The dashboard will:
- Display real-time metrics
- Show interactive charts
- Display gesture performance
- Auto-refresh every 2 seconds

### Step 5: Monitor Performance
- Watch metrics update in real-time
- Check gesture accuracy
- Identify weak gestures
- Monitor resource usage

## API Endpoints

### GET `/api/metrics`
Get current performance metrics

**Response:**
```json
{
  "fps": {
    "current": 29.8,
    "min": 29.5,
    "max": 30.1,
    "target": 30.0
  },
  "latency_ms": {
    "current": 10.4,
    "min": 10.1,
    "max": 10.9,
    "target": 33.0
  },
  "confidence": {
    "current": 0.8369,
    "min": 0.7027,
    "max": 0.9486,
    "target": 0.8
  },
  "accuracy": {
    "overall": 88.0,
    "correct_predictions": 88,
    "total_predictions": 100
  },
  "resources": {
    "memory_mb": 245.3,
    "cpu_percent": 12.5
  },
  "session": {
    "duration_seconds": 45.2,
    "total_frames": 1350,
    "total_gestures": 26
  },
  "gesture_stats": {
    "A": {
      "attempts": 20,
      "correct": 19,
      "accuracy": 95.0,
      "confidences": [0.95, 0.94, ...]
    }
  }
}
```

### GET `/api/metrics/summary`
Get detailed text summary

**Response:**
```json
{
  "report": "======================================================================\nPERFORMANCE METRICS REPORT\n..."
}
```

### GET `/api/metrics/gestures`
Get gesture-specific metrics

**Response:**
```json
{
  "top_gestures": [
    {"name": "A", "accuracy": 95.0},
    {"name": "B", "accuracy": 93.5}
  ],
  "worst_gestures": [
    {"name": "Z", "accuracy": 72.3}
  ],
  "total_gestures": 26
}
```

### POST `/api/metrics/reset`
Reset all metrics

**Response:**
```json
{
  "message": "Metrics reset successfully"
}
```

### GET `/api/health`
Health check

**Response:**
```json
{
  "status": "ok",
  "metrics_available": true
}
```

## Dashboard Features

### Metrics Cards
- **FPS Card** - Shows current FPS with color indicator
- **Latency Card** - Shows processing time in milliseconds
- **Confidence Card** - Shows model confidence percentage
- **Accuracy Card** - Shows overall accuracy percentage

Each card displays:
- Current value
- Min/Max range
- Target value
- Status indicator (OK/WARNING/POOR)

### Charts
- **FPS Trend** - Line chart showing FPS over time
- **Latency Trend** - Line chart showing latency over time
- **Confidence Trend** - Line chart showing confidence over time
- **Accuracy Trend** - Line chart showing accuracy over time

Charts:
- Update every 2 seconds
- Show last 20 data points
- Use color-coded lines
- Have smooth animations

### Gesture Performance
- Shows all recognized gestures
- Sorted by accuracy (highest first)
- Visual progress bars
- Displays:
  - Gesture name
  - Accuracy percentage
  - Number of attempts
  - Average confidence

### Session Information
- **Duration** - How long the session has been running
- **Total Frames** - Number of frames processed
- **Total Gestures** - Number of unique gestures recognized
- **Memory Usage** - RAM consumption in MB
- **CPU Usage** - Processor utilization percentage

## Color Coding

### Status Indicators
- 🟢 **Green** - Good performance
  - FPS ≥ 28
  - Latency ≤ 33ms
  - Confidence ≥ 80%
  
- 🟡 **Yellow** - Acceptable performance
  - FPS ≥ 20
  - Latency ≤ 50ms
  - Confidence ≥ 60%
  
- 🔴 **Red** - Poor performance
  - FPS < 20
  - Latency > 50ms
  - Confidence < 60%

## Responsive Design

### Desktop (1200px+)
- 4-column metrics grid
- 2-column charts grid
- Full-width gesture grid
- Optimal spacing

### Tablet (768px - 1199px)
- 2-column metrics grid
- 1-column charts grid
- Responsive gesture grid

### Mobile (< 768px)
- 1-column metrics grid
- 1-column charts grid
- Single-column layout
- Touch-friendly buttons

## Browser Compatibility

- Chrome/Chromium (recommended)
- Firefox
- Safari
- Edge
- Mobile browsers (iOS Safari, Chrome Mobile)

## Performance

- Lightweight dashboard (~50KB)
- Efficient chart updates
- Auto-refresh every 2 seconds
- Smooth animations
- Low CPU usage

## Troubleshooting

### Dashboard Not Loading
**Problem**: Page shows "Cannot GET /performance_dashboard.html"
**Solution**: Make sure metrics_api.py is running on port 5001

### No Data Showing
**Problem**: Dashboard shows empty charts
**Solution**: 
1. Make sure main.py is running with camera
2. Make sure metrics_api.py is running
3. Wait 2 seconds for first data point

### Metrics Not Updating
**Problem**: Charts not updating
**Solution**:
1. Check browser console for errors
2. Verify API is responding: http://localhost:5001/api/health
3. Restart metrics_api.py

### Memory Shows 0 MB
**Problem**: Memory usage always shows 0
**Solution**: Install psutil
```bash
pip install psutil
```

## Installation

### Requirements
```bash
pip install flask
pip install psutil  # Optional, for CPU/memory monitoring
```

### Setup
1. Make sure `performance_metrics.py` is in the same directory
2. Make sure `web/` directory exists with HTML files
3. Run `metrics_api.py` on port 5001
4. Access dashboard through web interface

## Integration with main.py

The metrics are automatically shared with the web dashboard:

1. **main.py** creates a `PerformanceMetrics` instance
2. **metrics_api.py** accesses the same instance
3. **performance_dashboard.html** fetches data via REST API
4. Dashboard displays real-time metrics

## Advanced Usage

### Custom Refresh Rate
Edit `performance_dashboard.html` line ~450:
```javascript
// Change from 2000ms to desired interval
setInterval(fetchMetrics, 2000);
```

### Custom Colors
Edit CSS in `performance_dashboard.html`:
```css
/* Change gradient colors */
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
```

### Custom Thresholds
Edit `getStatusClass()` function in `performance_dashboard.html`:
```javascript
const thresholds = {
    fps: { ok: 28, warning: 20 },  // Adjust these values
    latency: { ok: 33, warning: 50 },
    confidence: { ok: 80, warning: 60 },
    accuracy: { ok: 90, warning: 80 }
};
```

## Deployment

### Local Development
```bash
python metrics_api.py
# Access at http://localhost:5001
```

### Production
Use a production WSGI server:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5001 metrics_api.py
```

## Summary

The web dashboard provides a professional, real-time view of your system's performance metrics. It's perfect for:

✅ **Live Monitoring** - Watch metrics update in real-time
✅ **Performance Analysis** - Identify bottlenecks and weak gestures
✅ **Demonstrations** - Show judges a professional dashboard
✅ **Optimization** - Track improvements over time
✅ **Debugging** - Identify issues quickly

## Next Steps

1. ✅ Start main.py with camera
2. ✅ Start metrics_api.py
3. ✅ Open web interface
4. ✅ Click Performance button
5. ✅ Monitor metrics in real-time

---

**Version**: 1.0
**Last Updated**: December 6, 2025
**Status**: Production Ready
