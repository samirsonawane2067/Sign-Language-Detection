import * as THREE from 'https://unpkg.com/three@0.154.0/build/three.module.js';
import { GLTFLoader } from 'https://unpkg.com/three@0.154.0/examples/jsm/loaders/GLTFLoader.js';
import { FBXLoader } from 'https://unpkg.com/three@0.154.0/examples/jsm/loaders/FBXLoader.js';
import { OrbitControls } from 'https://unpkg.com/three@0.154.0/examples/jsm/controls/OrbitControls.js';

// ===== WebSocket Setup =====
const socket = window.io ? io('http://localhost:5000', { reconnection: true }) : null;
const wsIndicator = document.getElementById('wsIndicator');
const wsStatus = document.getElementById('wsStatus');
const incomingText = document.getElementById('incomingText');

if(socket){
    socket.on('connect', ()=>{
        console.log('✓ Connected to WebSocket server');
        wsIndicator.classList.remove('ws-disconnected');
        wsIndicator.classList.add('ws-connected');
        wsStatus.textContent = 'Connected';
        // Identify as web client
        socket.emit('register_client', { type: 'web' });
        console.log(`✓ Sent register_client (socket ID: ${socket.id})`);
    });
    
    socket.on('disconnect', ()=>{
        console.log('✗ Disconnected from WebSocket server');
        wsIndicator.classList.remove('ws-connected');
        wsIndicator.classList.add('ws-disconnected');
        wsStatus.textContent = 'Disconnected';
    });
    
    socket.on('status', (data)=>{
        console.log('Server status:', data);
        wsStatus.textContent = data.message || 'Connected';
    });
    
    // Receive recognized sign language text from Python
    socket.on('sign_recognized', (data)=>{
        const text = data.text || '';
        console.log('🎯 [sign_recognized event received]', { text, availableClips: clips.length, availableActions: Object.keys(actions).length });
        incomingText.textContent = 'Python: ' + text;
        incomingText.style.display = 'block';
        setTimeout(()=>{ incomingText.style.display = 'none'; }, 6000);
        // Auto-animate
        let clipName = mapTextToClip(text);
        console.log(`🎯 [sign_recognized] Mapped "${text}" to clip: "${clipName}"`);
        if(clipName) {
            playAnimationByName(clipName);
        } else if(clips.length > 0) {
            // Pick random animation if no mapping found
            const randomClip = clips[Math.floor(Math.random() * clips.length)];
            console.log(`🎯 [sign_recognized] No mapping, picking random: "${randomClip.name}"`);
            playAnimationByName(randomClip.name);
        } else {
            console.log(`🎯 [sign_recognized] No clips, using dummy animation`);
            playDummyAnimation(text || 'fallback');
        }
    });
    
    // Also listen for animate_text for backward compatibility
    socket.on('animate_text', (data)=>{
        const text = data.text || '';
        console.log('🎯 [animate_text event received]:', text);
        let clipName = mapTextToClip(text);
        if(clipName) playAnimationByName(clipName);
    });
    
    socket.on('speak', (data)=>{
        console.log('Python requesting speak:', data);
    });
} else {
    wsStatus.textContent = 'WebSocket unavailable (Socket.IO not loaded)';
}
// ===== End WebSocket Setup =====

const container = document.getElementById('canvas-container');
const statusEl = document.getElementById('status');

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x222233);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 1.4, 3);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
container.appendChild(renderer.domElement);

const light = new THREE.HemisphereLight(0xffffee, 0x222233, 1.0);
scene.add(light);

const dir = new THREE.DirectionalLight(0xffffff, 0.6);
dir.position.set(5, 10, 7);
scene.add(dir);

const grid = new THREE.GridHelper(6, 12, 0x444444, 0x222222);
scene.add(grid);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0,1,0);
controls.update();

let mixer = null;
let avatar = null;
let actions = {};
let clips = [];
// The mesh/visual node we should transform for visible motion
let avatarVisual = null;
// Optional bone references (if model contains a skeleton)
let leftArmBone = null;
let rightArmBone = null;
let headBone = null;
// Map filename to actual clip names inside that file
let fileClipMap = {}; // e.g., {'Fast Run.fbx': ['Take 001', 'mixamo.com'], ...}
// Fallback programmatic animation state when GLB has no clips
let programmaticAnim = null; // { startTime, duration, fromRotY, toRotY }

const loader = new GLTFLoader();
// Place your model at: web/models/human.glb
const MODEL_PATH = './models/human.glb';

function setStatus(msg){ statusEl.textContent = msg; }

loader.load(MODEL_PATH, (gltf) => {
    avatar = gltf.scene;
    avatar.position.set(0,0,0);
    scene.add(avatar);

    clips = gltf.animations || [];
    console.log('[Model Load] human.glb animations:', clips.length);
    if(clips.length > 0) {
        clips.forEach((c, i) => {
            console.log(`  [${i}] ${c.name || c.uuid}: ${c.duration.toFixed(2)}s, tracks: ${c.tracks.length}`);
        });
    }
    
    if(clips.length === 0){ 
        setStatus('⚠ Model loaded but has NO animations. FBX files will be used.'); 
    }
    else {
        // Log clip names for debugging and show them in the status briefly
        const names = clips.map(c => c.name || c.uuid);
        console.log('GLTF clips:', names);
        setStatus(`Loaded model with ${clips.length} animations: ${names.join(', ')}`);
        // After 3s revert to shorter status
        setTimeout(()=> setStatus(`Loaded model with ${clips.length} animations.`), 3000);
    }

    mixer = new THREE.AnimationMixer(avatar);
    clips.forEach((c)=>{
        const name = c.name || c.uuid;
        actions[name] = mixer.clipAction(c);
    });

    // Find a visible mesh inside the model to apply transforms to (some GLTFs nest meshes)
    avatar.traverse((o)=>{
        if(!avatarVisual && (o.isMesh || o.isSkinnedMesh)){
            avatarVisual = o;
        }
    });
    if(avatarVisual){
        console.log('[Model Load] avatarVisual node:', avatarVisual.name || avatarVisual.type);
    }

    // Try to find bones in the loaded model (useful for programmatic arm/head motion)
    avatar.traverse((o)=>{
        // Mesh with skeleton
        if(o.isSkinnedMesh && o.skeleton){
            o.skeleton.bones.forEach(b=>{
                const n = (b.name || '').toLowerCase();
                if(!leftArmBone && (n.includes('left') && n.includes('arm') || n.includes('l_arm') || n.includes('leftarm'))) leftArmBone = b;
                if(!rightArmBone && (n.includes('right') && n.includes('arm') || n.includes('r_arm') || n.includes('rightarm'))) rightArmBone = b;
                if(!headBone && (n.includes('head') || n.includes('neck') || n.includes('neck1'))) headBone = b;
            });
        }
        // Some exports mark bones as objects
        if(o.type === 'Bone' || o.isBone){
            const n = (o.name || '').toLowerCase();
            if(!leftArmBone && (n.includes('left') && n.includes('arm'))) leftArmBone = o;
            if(!rightArmBone && (n.includes('right') && n.includes('arm'))) rightArmBone = o;
            if(!headBone && (n.includes('head') || n.includes('neck'))) headBone = o;
        }
    });
    if(leftArmBone || rightArmBone || headBone){
        console.log('[Model Load] Detected bones:', {
            left: leftArmBone ? leftArmBone.name : null,
            right: rightArmBone ? rightArmBone.name : null,
            head: headBone ? headBone.name : null
        });
    }
}, (xhr)=>{
    setStatus(`Loading model: ${Math.round((xhr.loaded/xhr.total||0)*100)}%`);
}, (err)=>{
    console.error('Model load error:', err);
    setStatus('Failed to load model. Check web/models/human.glb');
});

// Load animations from manifest
const fbxLoader = new FBXLoader();
const gltfLoader = new GLTFLoader();

async function loadAnimationsFromManifest() {
    try {
        console.log('[Loader] Fetching manifest.json...');
        const response = await fetch('./models/animations/manifest.json');
        if (!response.ok) {
            console.error('[Loader] Failed to fetch manifest:', response.status, response.statusText);
            return;
        }
        const animationFiles = await response.json();
        console.log('[Loader] Animation files from manifest:', animationFiles);
        
        // Count animations to load (skip manifest.json and human.glb)
        const toLoad = animationFiles.filter(f => f !== 'manifest.json' && f !== 'human.glb');
        console.log(`[Loader] Will load ${toLoad.length} animation files`);
        
        let loadedCount = 0;
        
        for (const filename of animationFiles) {
            // Skip manifest.json and human.glb (main model)
            if (filename === 'manifest.json' || filename === 'human.glb') continue;
            
            const filepath = `./models/animations/${filename}`;
            const cleanName = filename.replace(/\.(fbx|glb|gltf)$/i, '').trim();
            
            if (filename.endsWith('.fbx')) {
                fbxLoader.load(filepath, (obj) => {
                    const newClips = obj.animations || [];
                    console.log(`[FBX] Loaded ${filename}: ${newClips.length} clips`);
                    
                    if (newClips.length > 0) {
                        // Store mapping of this file to its clip names
                        fileClipMap[filename] = newClips.map(c => c.name);
                        console.log(`[FBX] Clip names in ${filename}:`, fileClipMap[filename]);
                        
                        newClips.forEach((clip) => {
                            clips.push(clip);
                            if (mixer) {
                                const actionName = clip.name || cleanName;
                                actions[actionName] = mixer.clipAction(clip);
                                console.log(`✓ Action added: "${actionName}" from ${filename} (total clips: ${clips.length})`);
                            }
                        });
                    } else {
                        // FBX loaded but no animations - this is OK for now
                        console.log(`⚠ ${filename} loaded but has no animation clips`);
                    }
                    loadedCount++;
                    console.log(`[Loader] Progress: ${loadedCount}/${toLoad.length}`);
                    updateAnimationStatus();
                }, undefined, (err) => {
                    console.error(`Failed to load FBX ${filename}:`, err);
                    loadedCount++;
                });
            } else if (filename.endsWith('.glb') || filename.endsWith('.gltf')) {
                gltfLoader.load(filepath, (gltf) => {
                    const newClips = gltf.animations || [];
                    console.log(`[GLB/GLTF] Loaded ${filename}: ${newClips.length} clips`);
                    newClips.forEach((clip) => {
                        clips.push(clip);
                        if (mixer) {
                            const actionName = clip.name || cleanName;
                            actions[actionName] = mixer.clipAction(clip);
                            console.log(`✓ Action added: "${actionName}" from ${filename} (total clips: ${clips.length})`);
                        }
                    });
                    loadedCount++;
                    console.log(`[Loader] Progress: ${loadedCount}/${toLoad.length}`);
                    updateAnimationStatus();
                }, undefined, (err) => {
                    console.error(`Failed to load GLB ${filename}:`, err);
                    loadedCount++;
                });
            }
        }
    } catch (err) {
        console.error('Failed to load animation manifest:', err);
    }
}

function updateAnimationStatus() {
    if (clips.length > 0) {
        const names = clips.map(c => c.name || c.uuid).slice(0, 5);
        const more = clips.length > 5 ? ` +${clips.length - 5} more` : '';
        setStatus(`Ready with ${clips.length} animations: ${names.join(', ')}${more}`);
    }
}

// Wait for mixer to be created, then load animations
// Check multiple times with increasing delays
function waitForMixerAndLoad(attempt = 0) {
    console.log(`[Loader] Attempt ${attempt + 1} to check mixer...`);
    if (mixer) {
        console.log('[Loader] ✓ Mixer is ready, starting animation load');
        loadAnimationsFromManifest();
    } else if (attempt < 20) {
        // Try again after a short delay (up to 20 attempts = ~10 seconds)
        setTimeout(() => waitForMixerAndLoad(attempt + 1), 500);
    } else {
        console.warn('[Loader] ✗ Mixer not ready after 10 seconds, loading animations anyway');
        loadAnimationsFromManifest();
    }
}
waitForMixerAndLoad();

// Example mapping: map text tokens to animation clip names
// Automatically matches against loaded animation file names
let textToAnimMap = {
    'hello': 'Waving Gesture',
    'hi': 'Waving Gesture',
    'wave': 'Waving Gesture',
    'waving': 'Waving Gesture',
    'thank you': 'Thankful',
    'thanks': 'Thankful',
    'thankful': 'Thankful',
    'thumbs up': 'Sitting Thumbs Up',
    'thumbs': 'Sitting Thumbs Up',
    'yes': 'Old Man Idle',
    'no': 'Angry',
    'angry': 'Angry',
    'run': 'Fast Run',
    'fast run': 'Fast Run',
    'jump': 'Jump',
    'pray': 'Praying',
    'praying': 'Praying',
    'sit': 'Sitting Clap',
    'clap': 'Sitting Clap',
    'laugh': 'Sitting Laughing',
    'talk': 'Talking',
    'talking': 'Talking',
    'dancing': 'Hip Hop Dancing',
    'dance': 'Hip Hop Dancing',
    'kick': 'Flying Kick',
    'fight': 'Fist Fight A',
    'fall': 'Falling',
    'pain': 'Pain Gesture',
    'point': 'Pain Gesture',
};

function mapTextToClip(text){
    if(!text) return null;
    const cleaned = text.trim().toLowerCase();
    
    // DEBUG: Log all available clips on first call
    if(!window._debugClipsLogged) {
        window._debugClipsLogged = true;
        console.log('=== AVAILABLE CLIPS (by actual clip name) ===');
        clips.forEach((c, i) => {
            console.log(`[${i}] "${c.name}" (from: ${getSourceFile(c.name)})`);
        });
        console.log('=== END CLIPS ===');
    }
    
    // If no clips available, return a special marker to trigger fallback
    if(clips.length === 0) {
        console.warn('[mapTextToClip] No clips available!');
        return 'FALLBACK';
    }
    
    // Try exact phrases first using textToAnimMap
    if(textToAnimMap[cleaned]) {
        const target = textToAnimMap[cleaned];
        console.log(`[mapTextToClip] "${text}" -> mapped to "${target}"`);
        // Now find if we have a clip with that name OR find the first clip from that file
        if(actions[target]) {
            return target;
        }
        // If exact action name not found, try to find a clip from the intended file
        const clipsFromFile = fileClipMap[target + '.fbx'] || [];
        if(clipsFromFile.length > 0) {
            const fallback = clipsFromFile[0];
            console.log(`[mapTextToClip] Action "${target}" not found, using first clip from file: "${fallback}"`);
            return fallback;
        }
        // Last resort: find by file basename match
        for(const [filename, clipNames] of Object.entries(fileClipMap)) {
            if(filename.toLowerCase().includes(target.toLowerCase()) && clipNames.length > 0) {
                console.log(`[mapTextToClip] Found matching file "${filename}", using clip: "${clipNames[0]}"`);
                return clipNames[0];
            }
        }
    }
    
    // Try single words in the phrase
    const tokens = cleaned.split(/\s+/);
    for(const t of tokens){
        if(textToAnimMap[t]) {
            const target = textToAnimMap[t];
            console.log(`[mapTextToClip] token "${t}" -> mapped to "${target}"`);
            // Try to find clip from that file
            const clipsFromFile = fileClipMap[target + '.fbx'] || [];
            if(clipsFromFile.length > 0) {
                return clipsFromFile[0];
            }
            // Try direct action name match
            if(actions[target]) {
                return target;
            }
        }
    }

    // Try matching against actual clip names (case-insensitive substring match)
    const lowerNames = clips.map(c => (c.name||'').toLowerCase());
    for(const t of [cleaned, ...tokens]){
        for(let i=0;i<lowerNames.length;i++){
            if(lowerNames[i].includes(t)) {
                const match = clips[i].name;
                console.log(`[mapTextToClip] substring match: "${t}" found in clip "${match}"`);
                return match;
            }
        }
    }

    // If no match found, try to pick by file basename
    for(const [filename, clipNames] of Object.entries(fileClipMap)) {
        const baseName = filename.replace(/\.(fbx|glb|gltf)$/i, '').toLowerCase();
        for(const token of [cleaned, ...tokens]) {
            if(baseName.includes(token) && clipNames.length > 0) {
                console.log(`[mapTextToClip] Matched token "${token}" to file "${filename}", using clip: "${clipNames[0]}"`);
                return clipNames[0];
            }
        }
    }

    // If no match found, cycle through available animations
    // This ensures something always happens
    if(clips.length > 0) {
        // Use a simple hash of the text to pick a clip
        let hash = 0;
        for(let i = 0; i < cleaned.length; i++) {
            hash = ((hash << 5) - hash) + cleaned.charCodeAt(i);
            hash = hash & hash;
        }
        const idx = Math.abs(hash) % clips.length;
        const fallbackClip = clips[idx].name;
        console.log(`[mapTextToClip] no match for "${text}", using hash fallback: "${fallbackClip}" (index ${idx}/${clips.length})`);
        return fallbackClip;
    }
    
    // Final fallback - trigger programmatic animation
    console.log(`[mapTextToClip] final fallback for "${text}" - NO CLIPS AVAILABLE, returning FALLBACK`);
    return 'FALLBACK';
}

// Helper function to find which file a clip comes from
function getSourceFile(clipName) {
    for(const [filename, clipNames] of Object.entries(fileClipMap)) {
        if(clipNames.includes(clipName)) {
            return filename;
        }
    }
    return 'unknown';
}

function playAnimationByName(name){
    console.log(`[playAnimationByName] Attempting to play: "${name}", total actions: ${Object.keys(actions).length}`);
    
    // Handle fallback animation for models without clips
    if(name === 'FALLBACK') {
        console.log('[playAnimationByName] Using FALLBACK animation');
        playDummyAnimation('text-to-sign');
        return;
    }
    
    if(!mixer){
        // If no mixer (no animations), try a programmatic fallback
        if(avatar){
            console.warn('[playAnimationByName] No mixer available, using dummy animation');
            playDummyAnimation(name);
            return;
        }
        setStatus('No animation mixer available yet.');
        return;
    }
    if(!actions[name]){
        // If there are no actions at all, fallback to a dummy animation
        if(Object.keys(actions).length === 0 && avatar){
            console.warn('[playAnimationByName] No actions available, using dummy animation');
            playDummyAnimation(name);
            return;
        }
        // Show available actions to help debugging
        const avail = Object.keys(actions).length? Object.keys(actions).slice(0, 5).join(', ') : '(none)';
        const more = Object.keys(actions).length > 5 ? ` +${Object.keys(actions).length - 5} more` : '';
        const msg = `Animation not found: "${name}" — available: ${avail}${more}`;
        setStatus(msg);
        console.error('[playAnimationByName]', msg);
        return;
    }
    // fade out other actions
    Object.keys(actions).forEach(k=>{
        if(k===name) return;
        const a = actions[k];
        a.fadeOut(0.15);
    });
    
    // Check if animation is valid (has actual tracks)
    const clip = clips.find(c => c.name === name);
    if(clip && clip.tracks.length === 0) {
        console.warn(`[playAnimationByName] ⚠ Animation "${name}" has NO tracks - using fallback`);
        playDummyAnimation(name);
        return;
    }
    
    const action = actions[name];
    action.reset();
    action.setLoop(THREE.LoopOnce);
    action.clampWhenFinished = true;
    action.fadeIn(0.15);
    action.play();
    
    // Log animation details
    if(clip) {
        console.log(`[playAnimationByName] ✓ Playing: "${name}"`);
        console.log(`  - Duration: ${clip.duration.toFixed(2)}s`);
        console.log(`  - Tracks: ${clip.tracks.length}`);
        console.log(`  - Tracks:`, clip.tracks.map(t => t.name));
    } else {
        console.log(`[playAnimationByName] ✓ Now playing: ${name}`);
    }
    setStatus('Playing: ' + name);
}

function playDummyAnimation(name){
    // Enhanced procedural fallback: rotation + oscillating bob + stronger arm/head motion
    if(!avatar) {
        console.error('[playDummyAnimation] No avatar available');
        return;
    }
    console.log('[playDummyAnimation] Starting fallback animation for:', name);
    const now = performance.now() / 1000;

    // Choose preset by keyword
    const key = ('' + name).toLowerCase();
    let preset = {
        duration: 1.8,
        yaw: Math.PI * 1.25,
        scale: 1.15,
        bob: 0.15,
        freq: 2.0,
        cycles: 1,
        armSwing: 0.9,
        headNod: 0.25
    };
    if(key.includes('run')){
        // More exaggerated "run" preset: quicker cycles, larger bob, stronger arm swing and lateral sway
        preset = { duration: 1.0, yaw: Math.PI * 1.4, scale: 1.14, bob: 0.6, freq: 6.5, cycles: 8, armSwing: 2.0, headNod: 0.36, lateral: 0.35, tilt: 0.18 };
    } else if(key.includes('clap') || key.includes('appl')){
        preset = { duration: 1.0, yaw: Math.PI * 0.35, scale: 1.22, bob: 0.12, freq: 5.0, cycles: 3, armSwing: 1.4, headNod: 0.08 };
    } else if(key.includes('wave') || key.includes('hand')){
        preset = { duration: 2.0, yaw: Math.PI * 0.5, scale: 1.05, bob: 0.08, freq: 2.2, cycles: 4, armSwing: 1.6, headNod: 0.12 };
    } else if(key.includes('dance')){
        preset = { duration: 2.2, yaw: Math.PI * 1.5, scale: 1.28, bob: 0.32, freq: 2.5, cycles: 4, armSwing: 1.6, headNod: 0.32 };
    }

    // store animation state
    programmaticAnim = {
        startTime: now,
        duration: preset.duration,
        cycles: preset.cycles,
        freq: preset.freq,
        yawFrom: avatar.rotation.y,
        yawAmp: preset.yaw,
        fromScaleY: avatar.scale.y,
        visualFromPos: avatarVisual ? avatarVisual.position.clone() : null,
        visualFromScale: avatarVisual ? avatarVisual.scale.clone() : null,
        scaleAmp: preset.scale,
        fromPosY: avatar.position.y,
        bobAmp: preset.bob,
        lateralAmp: preset.lateral || 0,
        tiltAmp: preset.tilt || 0,
        leftArmFrom: leftArmBone ? leftArmBone.rotation.clone() : null,
        rightArmFrom: rightArmBone ? rightArmBone.rotation.clone() : null,
        headFrom: headBone ? headBone.rotation.clone() : null,
        leftArmAmp: preset.armSwing,
        rightArmAmp: preset.armSwing * 0.6,
        headAmp: preset.headNod,
        pattern: key
    };
    console.log('[playDummyAnimation] ✓ Fallback animation scheduled (preset):', programmaticAnim);
    setStatus(`Fallback animation: ${name}`);
}

// Animation loop
const clock = new THREE.Clock();
function animate(){
    requestAnimationFrame(animate);
    const dt = clock.getDelta();
    if(mixer) mixer.update(dt);
    // Update programmatic fallback animation if present
    if(programmaticAnim && avatar){
        const now = performance.now() / 1000;
        const elapsed = now - programmaticAnim.startTime;

        // repeat/cycle handling
        const totalDuration = programmaticAnim.duration * (programmaticAnim.cycles || 1);
        const tRaw = Math.min(1, elapsed / totalDuration);

        // progress within current cycle (0..1)
        const cycleIndex = Math.floor((elapsed / programmaticAnim.duration));
        const cycleProgress = Math.min(1, (elapsed - cycleIndex * programmaticAnim.duration) / programmaticAnim.duration);

        // use a sinusoidal oscillation for lively motion
        const phase = cycleProgress * Math.PI * 2 * (programmaticAnim.freq || 1);
        const osc = Math.sin(phase);

        // easing envelope (fade in/out per cycle)
        const env = 0.5 * (1 - Math.cos(Math.min(1, cycleProgress) * Math.PI));

        // yaw: base + oscillatory component + overall yaw amplitude scaled by envelope
        const yawVal = (programmaticAnim.yawFrom || 0) + (programmaticAnim.yawAmp || 0) * (0.2 * osc + 0.8 * env);
        // Apply to visual mesh when available for clearer effect
        if(avatarVisual){
            avatarVisual.rotation.y = yawVal;
        } else {
            avatar.rotation.y = yawVal;
        }

        // tilt forward/back (rotate on X) for running feeling
        if(programmaticAnim.tiltAmp){
            avatar.rotation.x = (programmaticAnim.tiltAmp || 0) * Math.abs(osc) * env * -1.0; // lean forward
        }

        // bob using sine (apply to visual mesh if present)
        const bobVal = (programmaticAnim.bobAmp || 0) * Math.abs(osc) * env;
        if(avatarVisual && programmaticAnim.visualFromPos){
            avatarVisual.position.y = programmaticAnim.visualFromPos.y + bobVal;
        } else if(typeof programmaticAnim.fromPosY !== 'undefined'){
            avatar.position.y = programmaticAnim.fromPosY + bobVal;
        }

        // scale pulse (apply to visual mesh if present)
        const scalePulse = programmaticAnim.fromScaleY ? (programmaticAnim.fromScaleY + ((programmaticAnim.scaleAmp || 1) - programmaticAnim.fromScaleY) * env * Math.abs(osc)) : null;
        if(avatarVisual && programmaticAnim.visualFromScale){
            avatarVisual.scale.copy(programmaticAnim.visualFromScale).multiplyScalar(scalePulse || 1);
        } else if(scalePulse){
            avatar.scale.y = scalePulse;
            avatar.scale.x = avatar.scale.z = scalePulse;
        }

        // bone motions (best-effort)
        try{
            if(leftArmBone && programmaticAnim.leftArmFrom){
                leftArmBone.rotation.x = programmaticAnim.leftArmFrom.x + (programmaticAnim.leftArmAmp || 0) * Math.sin(phase) * env;
            }
            if(rightArmBone && programmaticAnim.rightArmFrom){
                rightArmBone.rotation.x = programmaticAnim.rightArmFrom.x + (programmaticAnim.rightArmAmp || 0) * Math.sin(phase * 0.9) * env;
            }
            if(headBone && programmaticAnim.headFrom){
                headBone.rotation.x = programmaticAnim.headFrom.x + (programmaticAnim.headAmp || 0) * Math.sin(phase * 0.6) * env;
            }
        } catch(e){ /* ignore bone update errors */ }

        // lateral sway (position X) to simulate steps (prefer visual node)
        const lateralVal = (programmaticAnim.lateralAmp || 0) * Math.sin(phase) * env;
        if(avatarVisual && programmaticAnim.visualFromPos){
            avatarVisual.position.x = programmaticAnim.visualFromPos.x + lateralVal;
        } else {
            avatar.position.x = lateralVal;
        }

        // one-frame debug log when programmatic animation is active
        if(!programmaticAnim._debugLogged){
            console.log('[progAnim] active preset:', programmaticAnim.pattern, 'amps:', {bob: programmaticAnim.bobAmp, lateral: programmaticAnim.lateralAmp, tilt: programmaticAnim.tiltAmp});
            programmaticAnim._debugLogged = true;
        }

        // end condition
        if(elapsed >= totalDuration){
            // restore base values
            if(programmaticAnim.fromPosY!==undefined) avatar.position.y = programmaticAnim.fromPosY;
            if(programmaticAnim.fromScaleY!==undefined){ avatar.scale.y = programmaticAnim.fromScaleY; avatar.scale.x = avatar.scale.z = programmaticAnim.fromScaleY; }
            // restore bones to original rotations if we stored them
            try{
                if(leftArmBone && programmaticAnim.leftArmFrom) leftArmBone.rotation.copy(programmaticAnim.leftArmFrom);
                if(rightArmBone && programmaticAnim.rightArmFrom) rightArmBone.rotation.copy(programmaticAnim.rightArmFrom);
                if(headBone && programmaticAnim.headFrom) headBone.rotation.copy(programmaticAnim.headFrom);
            }catch(e){}
            programmaticAnim = null;
        }
    }
    renderer.render(scene, camera);
}
animate();

// Resize
window.addEventListener('resize', ()=>{
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

// Expose for debugging
window._threeApp = { scene, camera, renderer, mixer, actions, clips };

// Log available animations for debugging
if(clips.length > 0) {
    console.log('Available animations:', clips.map(c => c.name || c.uuid));
    setStatus(`Ready! Available animations: ${clips.map(c => c.name || c.uuid).join(', ')}`);
} else {
    setStatus('Ready. No animations found in model. Using fallback mode.');
}