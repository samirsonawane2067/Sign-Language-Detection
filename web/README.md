Three.js Text → Sign Avatar Demo

Overview
- Minimal WebGL demo that loads a GLTF/GLB avatar with animations and maps text to animation clips.
- Built with Three.js (module) and GLTFLoader. No build step required.

Files
- `index.html` — UI + canvas container.
- `app.js` — Three.js scene, GLTF loader, animation mixer, and simple text→animation mapping.
- `models/avatar.glb` — Place your exported GLB here (not included).

How to use
1. Export your avatar and animations from Blender as a single GLB. Name the actions in Blender (e.g. `Wave`, `ThankYou`, `Nod`, `ShakeHead`, `Idle`) and ensure they are included in the GLB.
   - In Blender: select armature, open Action Editor, name actions, then File → Export → glTF 2.0 → Format: GLB, Check "Animations".
2. Put the resulting `avatar.glb` in `web/models/avatar.glb`.
3. Run a local static server from the `web/` folder (browsers block module imports from file://):

```powershell
cd web
python -m http.server 8000
```

4. Open `http://localhost:8000` in your browser.
5. Type a phrase like `hello` or `thank you` and press Animate. The demo maps those tokens to clip names defined in `app.js`.

Mapping
- Customize the `textToAnimMap` in `app.js` to map phrases/words to your animation clip names.
- If no direct mapping is found, the demo plays the first animation clip in the GLB as a fallback.

Unity / Blender / Production
- For higher quality and more natural signing sequences, author animation clips in Blender or use motion-captured sign datasets.
- Unity allows sophisticated avatar systems, IK, blend trees, and runtime retargeting; but requires building a web player or WebGL export.

Next steps / Integration
- Hook this demo into your Python app by embedding the web UI (e.g., `main_web.py`) or serving it alongside your Flask app.
- For full-duplex realtime integration: send recognized speech (from Python) to the web client via WebSocket; the web client triggers the animations on incoming text. Conversely, send signed text from the web client to Python for TTS playback.
