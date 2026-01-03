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
        console.log('Connected to WebSocket server');
        wsIndicator.classList.remove('ws-disconnected');
        wsIndicator.classList.add('ws-connected');
        wsStatus.textContent = 'Connected';
        // Identify as web client
        socket.emit('register_client', { type: 'web' });
    });
    
    socket.on('disconnect', ()=>{
        console.log('Disconnected from WebSocket server');
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
        console.log('Received from Python (sign_recognized):', text);
        incomingText.textContent = 'Python: ' + text;
        incomingText.style.display = 'block';
        setTimeout(()=>{ incomingText.style.display = 'none'; }, 6000);
        // Auto-animate
        let clipName = mapTextToClip(text);
        console.log('Mapped text "' + text + '" to clip:', clipName, 'Available clips:', clips.length);
        if(clipName) {
            playAnimationByName(clipName);
        } else if(clips.length > 0) {
            // Pick random animation if no mapping found
            const randomClip = clips[Math.floor(Math.random() * clips.length)];
            playAnimationByName(randomClip.name);
        } else {
            playDummyAnimation(text || 'fallback');
        }
    });
    
    // Also listen for animate_text for backward compatibility
    socket.on('animate_text', (data)=>{
        const text = data.text || '';
        console.log('Received animate_text:', text);
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
let animationsLoaded = 0;
let animationsToLoad = 0;

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
    if(clips.length === 0){ 
        setStatus('Main model has no animations. Loading external animations...'); 
    }
    else {
        // Log clip names for debugging and show them in the status briefly
        const names = clips.map(c => c.name || c.uuid);
        console.log('GLTF clips:', names);
        setStatus(`Loaded model with ${clips.length} animations: ${names.join(', ')}`);
    }

    mixer = new THREE.AnimationMixer(avatar);
    clips.forEach((c)=>{
        const name = c.name || c.uuid;
        actions[name] = mixer.clipAction(c);
    });
    
    // NOW start loading animations from manifest after mixer is ready
    console.log('Mixer created, loading animations from manifest...');
    loadAnimationsFromManifest();
}, (xhr)=>{
    setStatus(`Loading model: ${Math.round((xhr.loaded/xhr.total||0)*100)}%`);
}, (err)=>{
    console.error('Model load error:', err);
    setStatus('Failed to load model. Check web/models/human.glb');
});

// Load animations from manifest
const fbxLoader = new FBXLoader();
const gltfLoader = new GLTFLoader();

function updateAnimationStatus() {
    if (clips.length > 0) {
        const names = clips.map(c => c.name || c.uuid).slice(0, 5);
        const more = clips.length > 5 ? ` +${clips.length - 5} more` : '';
        console.log(`Status update: ${clips.length} animations loaded`);
        setStatus(`Ready with ${clips.length} animations: ${names.join(', ')}${more}`);
    }
}

async function loadAnimationsFromManifest() {
    try {
        const response = await fetch('./models/animations/manifest.json');
        const animationFiles = await response.json();
        console.log('Animation files from manifest:', animationFiles);
        
        // Filter to only FBX/GLB files (skip manifest.json and human.glb)
        const filesToLoad = animationFiles.filter(f => 
            !f.includes('manifest') && !f.includes('human.glb') && 
            (f.endsWith('.fbx') || f.endsWith('.glb') || f.endsWith('.gltf'))
        );
        
        animationsToLoad = filesToLoad.length;
        console.log(`Will load ${animationsToLoad} animation files`);
        
        for (const filename of filesToLoad) {
            const filepath = `./models/animations/${filename}`;
            const cleanName = filename.replace(/\.(fbx|glb|gltf)$/i, '').trim();
            
            if (filename.endsWith('.fbx')) {
                fbxLoader.load(filepath, (obj) => {
                    const newClips = obj.animations || [];
                    console.log(`[FBX] ✓ Loaded ${filename}: ${newClips.length} animation clips`);
                    
                    if (newClips.length > 0) {
                        newClips.forEach((clip) => {
                            clips.push(clip);
                            if (mixer) {
                                const actionName = clip.name || cleanName;
                                actions[actionName] = mixer.clipAction(clip);
                                console.log(`  ✓ Action added: "${actionName}"`);
                            }
                        });
                    }
                    animationsLoaded++;
                    updateAnimationStatus();
                    console.log(`Progress: ${animationsLoaded}/${animationsToLoad}`);
                }, undefined, (err) => {
                    console.error(`[FBX ERROR] Failed to load ${filename}:`, err);
                    animationsLoaded++;
                    updateAnimationStatus();
                });
            } else if (filename.endsWith('.glb') || filename.endsWith('.gltf')) {
                gltfLoader.load(filepath, (gltf) => {
                    const newClips = gltf.animations || [];
                    console.log(`[GLB/GLTF] ✓ Loaded ${filename}: ${newClips.length} animation clips`);
                    newClips.forEach((clip) => {
                        clips.push(clip);
                        if (mixer) {
                            const actionName = clip.name || cleanName;
                            actions[actionName] = mixer.clipAction(clip);
                            console.log(`  ✓ Action added: "${actionName}"`);
                        }
                    });
                    animationsLoaded++;
                    updateAnimationStatus();
                    console.log(`Progress: ${animationsLoaded}/${animationsToLoad}`);
                }, undefined, (err) => {
                    console.error(`[GLB ERROR] Failed to load ${filename}:`, err);
                    animationsLoaded++;
                    updateAnimationStatus();
                });
            }
        }
    } catch (err) {
        console.error('Failed to load animation manifest:', err);
    }
}

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
    
    console.log(`mapTextToClip("${cleaned}") - clips available: ${clips.length}`);
    
    // If no clips available, return a special marker to trigger fallback
    if(clips.length === 0) {
        console.warn('No clips available, using fallback');
        return 'FALLBACK';
    }
    
    // Try exact phrases first
    if(textToAnimMap[cleaned]) {
        console.log(`✓ Exact match found: ${textToAnimMap[cleaned]}`);
        return textToAnimMap[cleaned];
    }
    
    // Try single words in the phrase
    const tokens = cleaned.split(/\s+/);
    for(const t of tokens){ 
        if(textToAnimMap[t]) {
            console.log(`✓ Token match found: ${t} -> ${textToAnimMap[t]}`);
            return textToAnimMap[t]; 
        }
    }

    // Try matching clip names (case-insensitive substring match)
    const lowerNames = clips.map(c => (c.name||'').toLowerCase());
    for(const t of [cleaned, ...tokens]){
        for(let i=0;i<lowerNames.length;i++){
            if(lowerNames[i].includes(t)) {
                console.log(`✓ Name match found: "${t}" matches clip "${clips[i].name}"`);
                return clips[i].name;
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
        const selectedClip = clips[Math.abs(hash) % clips.length];
        console.log(`✓ Hash fallback: "${cleaned}" -> "${selectedClip.name}"`);
        return selectedClip.name;
    }
    
    // Final fallback - trigger programmatic animation
    console.log('Using programmatic fallback animation');
    return 'FALLBACK';
}

function playAnimationByName(name){
    console.log(`playAnimationByName("${name}")`);
    
    // Handle fallback animation for models without clips
    if(name === 'FALLBACK') {
        console.log('Playing dummy/fallback animation');
        playDummyAnimation('text-to-sign');
        return;
    }
    
    if(!mixer){
        console.warn('No mixer available');
        if(avatar){
            playDummyAnimation(name);
            return;
        }
        setStatus('No animation mixer available yet.');
        return;
    }
    if(!actions[name]){
        console.warn(`Action "${name}" not found in actions`);
        // If there are no actions at all, fallback to a dummy animation
        if(Object.keys(actions).length === 0 && avatar){
            console.log('No actions available, using dummy animation');
            playDummyAnimation(name);
            return;
        }
        // Show available actions to help debugging
        const avail = Object.keys(actions).length? Object.keys(actions).join(', ') : '(none)';
        setStatus('Animation not found: ' + name + ' — available: ' + avail);
        console.warn('Available actions:', avail);
        return;
    }
    
    console.log(`✓ Playing animation: "${name}"`);
    // fade out other actions
    Object.keys(actions).forEach(k=>{
        if(k===name) return;
        const a = actions[k];
        a.fadeOut(0.15);
    });
    const action = actions[name];
    action.reset();
    action.setLoop(THREE.LoopOnce);
    action.clampWhenFinished = true;
    action.fadeIn(0.15);
    action.play();
    setStatus('Playing: ' + name);
}

function playDummyAnimation(name){
    // Enhanced rotation and scale animation as a visual fallback when there are no GLB clips
    if(!avatar) return;
    const now = performance.now() / 1000;
    const duration = 1.5; // seconds - longer for more visible effect
    programmaticAnim = {
        startTime: now,
        duration: duration,
        fromRotY: avatar.rotation.y,
        toRotY: avatar.rotation.y + Math.PI * 1.5, // rotate ~270 degrees for more dramatic effect
        fromScaleY: avatar.scale.y,
        toScaleY: avatar.scale.y * 1.2    // slight scale up for emphasis
    };
    setStatus(`Animating: ${name}`);
}

// UI hookup
const textInput = document.getElementById('textInput');
const playBtn = document.getElementById('playBtn');
const sendBtn = document.getElementById('sendBtn');

playBtn.addEventListener('click', ()=>{
    const text = textInput.value;
    const clipName = mapTextToClip(text);
    if(clipName) playAnimationByName(clipName);
    else setStatus('No suitable animation found for: ' + text);
});

// Send text to Python for TTS
if(sendBtn){
    sendBtn.addEventListener('click', ()=>{
        const text = textInput.value;
        if(text && socket){
            socket.emit('web_animate_request', { text: text });
            setStatus('Sent to Python: ' + text);
        } else if(!socket){
            setStatus('WebSocket not connected');
        }
    });
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
        const t = Math.min(1, (now - programmaticAnim.startTime) / programmaticAnim.duration);
        // ease-out cubic
        const ease = 1 - Math.pow(1 - t, 3);
        avatar.rotation.y = programmaticAnim.fromRotY + (programmaticAnim.toRotY - programmaticAnim.fromRotY) * ease;
        
        // Also animate scale if available
        if(programmaticAnim.fromScaleY && programmaticAnim.toScaleY) {
            avatar.scale.y = programmaticAnim.fromScaleY + (programmaticAnim.toScaleY - programmaticAnim.fromScaleY) * ease;
        }
        
        if(t >= 1){ programmaticAnim = null; }
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
console.log('App initialized. Waiting for animations to load...');
setStatus('Loading animations...');
