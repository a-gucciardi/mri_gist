import * as THREE from 'three';
import { NRRDLoader } from 'three/examples/jsm/loaders/NRRDLoader.js';
import { VolumeRenderShader1 } from 'three/examples/jsm/shaders/VolumeShader.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GUI } from 'three/addons/libs/lil-gui.module.min.js';
import Stats from 'three/examples/jsm/libs/stats.module.js';

let stats;
let scene, renderer;
let windowWidth, windowHeight;
let meshBrain, brainBox;
let material, volconfig, cmtextures;
let orbitControls = [];
let currentVolume = null;

const views = [
    {
        left: 0,
        bottom: 0,
        width: 0.5,
        height: 1.0,
        background: new THREE.Color().setRGB(0.5, 0.5, 0.7),
        eye: [0, 2000, 0],
        up: [0, 0, 1],
        fov: 45,
        name: "Coronal",
        zoom: true,
        rotate: true
    },
    {
        left: 0.5,
        bottom: 0,
        width: 0.5,
        height: 0.5,
        background: new THREE.Color().setRGB(0.7, 0.5, 0.5),
        eye: [2000, 0, 0],
        up: [0, 0, 1],
        fov: 30,
        name: "Axial",
        zoom: false,
        rotate: false
    },
    {
        left: 0.5,
        bottom: 0.5,
        width: 0.5,
        height: 0.5,
        background: new THREE.Color().setRGB(0.5, 0.7, 0.7),
        eye: [0, 0, 2000],
        up: [0, -1, 0],
        fov: 60,
        name: "Sagittal",
        zoom: false,
        rotate: false
    }
];

const gui = new GUI();
const fileParams = {
    currentFile: '',
    segment: function () { triggerSegmentation(); }
};

async function init() {
    const container = document.getElementById('container');
    scene = new THREE.Scene();

    // Fetch files from backend
    try {
        const response = await fetch('/api/files');
        const files = await response.json();

        if (files.length > 0) {
            const fileNames = files.map(f => f.name);
            fileParams.currentFile = fileNames[0];

            const fileFolder = gui.addFolder('File Selection');
            fileFolder.add(fileParams, 'currentFile', fileNames).name('Select File').onChange(loadVolume);
            fileFolder.add(fileParams, 'segment').name('Run Segmentation');

            // Load first file
            loadVolume(fileParams.currentFile);
        }
    } catch (e) {
        console.error("Failed to fetch files:", e);
    }

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);
    container.appendChild(renderer.domElement);

    stats = new Stats();
    container.appendChild(stats.dom);

    setupCamerasAndControls();

    window.addEventListener('resize', updateSize);
    animate();
}

function loadVolume(fileName) {
    // Clear existing volume
    if (meshBrain) scene.remove(meshBrain);
    if (brainBox) scene.remove(brainBox);

    // Find file path (in a real app, we'd use the full path or ID)
    // For now, assuming files are served statically or via API
    // Since we are serving static files, we might need to adjust how we access them
    // If the file is in 'nrrd/' folder in static, we can access it directly
    // But the API returns absolute paths.
    // Let's assume for this demo that we can access them via a static route if they are in the static folder
    // OR we need an endpoint to serve the file content.
    // For simplicity, let's assume the files are in 'static/nrrd/' and we just use the filename.

    const filePath = `static/nrrd/${fileName}`; // Simplified for demo

    new NRRDLoader().load(filePath, function (volume) {
        currentVolume = volume;

        // Texture setup
        volconfig = { renderstyle: 'iso', isothreshold: 0.15, colormap: 'gray' };

        let texture = new THREE.Data3DTexture(volume.data, volume.xLength, volume.yLength, volume.zLength);
        texture.format = THREE.RedFormat;
        texture.type = THREE.FloatType;
        texture.minFilter = texture.magFilter = THREE.LinearFilter;
        texture.needsUpdate = true;

        cmtextures = {
            viridis: new THREE.TextureLoader().load('static/textures/cm_viridis.png', render),
            gray: new THREE.TextureLoader().load('static/textures/cm_gray.png', render),
        };

        let shader = VolumeRenderShader1;
        let uniforms = THREE.UniformsUtils.clone(shader.uniforms);

        uniforms['u_data'].value = texture;
        uniforms['u_size'].value.set(volume.xLength, volume.yLength, volume.zLength);
        uniforms['u_renderstyle'].value = volconfig.renderstyle == 'mip' ? 0 : 1;
        uniforms['u_renderthreshold'].value = volconfig.isothreshold;
        uniforms['u_cmdata'].value = cmtextures[volconfig.colormap];

        material = new THREE.ShaderMaterial({
            uniforms: uniforms,
            vertexShader: shader.vertexShader,
            fragmentShader: shader.fragmentShader,
            side: THREE.BackSide,
            clipping: true
        });

        const geometry = new THREE.BoxGeometry(volume.xLength, volume.yLength, volume.zLength);
        geometry.translate(volume.xLength / 2 - 0.5, volume.yLength / 2 - 0.5, volume.zLength / 2 - 0.5);

        meshBrain = new THREE.Mesh(geometry, material);
        meshBrain.scale.set(0.5, 0.5, 0.5);
        meshBrain.rotateZ(Math.PI);
        meshBrain.position.set(volume.xLength / 4, volume.yLength / 4, -(volume.zLength / 4));
        scene.add(meshBrain);

        brainBox = new THREE.Mesh(geometry, material);
        brainBox.scale.set(0.5, 0.5, 0.5);
        brainBox.position.set(-(volume.xLength / 4), - (volume.yLength / 4), - (volume.zLength / 4));
        brainBox.material = new THREE.MeshBasicMaterial({ color: 0x00ff00, wireframe: true });
        brainBox.visible = false;
        scene.add(brainBox);

        setupGUI();
    });
}

function setupCamerasAndControls() {
    const cams = gui.addFolder('Cameras');
    views.forEach((view) => {
        const camera = new THREE.OrthographicCamera(
            -view.width, view.width,
            view.height / 2, -view.height / 2,
            400, 3000
        );
        camera.position.fromArray(view.eye);
        camera.up.fromArray(view.up);
        camera.zoom = 0.002;
        view.camera = camera;

        const controls = new OrbitControls(view.camera, renderer.domElement);
        controls.target.set(0, 0, 0);
        controls.minDistance = 1000;
        controls.enableZoom = view.zoom;
        controls.enableRotate = view.rotate;
        orbitControls.push(controls);
    });
}

let volumeChart = null;

function showStats() {
    const statsContainer = document.getElementById('stats-container');
    if (statsContainer.style.display === 'none') {
        statsContainer.style.display = 'block';

        const ctx = document.getElementById('volumeChart').getContext('2d');

        if (volumeChart) {
            volumeChart.destroy();
        }

        // Dummy data for now - in real app, fetch from backend analysis
        volumeChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['GM', 'WM', 'CSF'],
                datasets: [{
                    label: 'Tissue Volume (mm³)',
                    data: [12000, 15000, 5000],
                    backgroundColor: [
                        'rgba(255, 99, 132, 0.2)',
                        'rgba(54, 162, 235, 0.2)',
                        'rgba(255, 206, 86, 0.2)'
                    ],
                    borderColor: [
                        'rgba(255, 99, 132, 1)',
                        'rgba(54, 162, 235, 1)',
                        'rgba(255, 206, 86, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    } else {
        statsContainer.style.display = 'none';
    }
}

function setupGUI() {
    // Simplified GUI setup for demo
    // In a real app, we'd check if folders exist to avoid duplicates
    const statsFolder = gui.addFolder('Statistics');
    statsFolder.add({ show: showStats }, 'show').name('Toggle Volume Stats');
}

async function triggerSegmentation() {
    if (!fileParams.currentFile) return;

    // Assuming file is in static/nrrd/
    // In reality, we need the absolute path which the backend knows
    // For this demo, we'll send the filename and let backend resolve it relative to DATA_DIR

    // We need to fetch the full path from the file list we got earlier, 
    // or just send the filename if backend handles it.
    // Let's assume backend handles relative paths from DATA_DIR

    const response = await fetch('/api/process/segment', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            input_file: `static/nrrd/${fileParams.currentFile}`, // Hack for demo
            output_file: `static/nrrd/${fileParams.currentFile.replace('.nrrd', '_seg.nrrd')}`
        })
    });

    const result = await response.json();
    console.log("Segmentation started:", result);
    alert("Segmentation started! Check console for details.");
}

function updateSize() {
    if (windowWidth !== window.innerWidth || windowHeight !== window.innerHeight) {
        windowWidth = window.innerWidth;
        windowHeight = window.innerHeight;
        renderer.setSize(windowWidth, windowHeight);
    }
}

function updateUniforms() {
    if (material) {
        material.uniforms['u_renderstyle'].value = volconfig.renderstyle == 'mip' ? 0 : 1;
        material.uniforms['u_renderthreshold'].value = volconfig.isothreshold;
        material.uniforms['u_cmdata'].value = cmtextures[volconfig.colormap];
        render();
    }
}

function animate() {
    render();
    stats.update();
    requestAnimationFrame(animate);
}

function render() {
    updateSize();

    for (let ii = 0; ii < views.length; ++ii) {
        const view = views[ii];
        const camera = view.camera;

        const left = Math.floor(windowWidth * view.left);
        const bottom = Math.floor(windowHeight * view.bottom);
        const width = Math.floor(windowWidth * view.width);
        const height = Math.floor(windowHeight * view.height);

        renderer.setViewport(left, bottom, width, height);
        renderer.setScissor(left, bottom, width, height);
        renderer.setScissorTest(true);
        renderer.setClearColor(view.background);

        camera.aspect = width / height;
        camera.updateProjectionMatrix();

        renderer.render(scene, camera);
    }
}

init();
