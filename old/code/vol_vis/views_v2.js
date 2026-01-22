import * as THREE from 'three';
import { NRRDLoader } from 'three/examples/jsm/loaders/NRRDLoader.js';
import { VolumeRenderShader1 } from 'three/examples/jsm/shaders/VolumeShader.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GUI } from 'three/addons/libs/lil-gui.module.min.js';
import { SimplifyModifier } from 'three/addons/modifiers/SimplifyModifier.js';
import Stats from 'three/examples/jsm/libs/stats.module.js';

let stats;
let scene, renderer;
let windowWidth, windowHeight;
let meshBrain, brainBox;
let material, volconfig, cmtextures;
let orbitControls = [];

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

function init() {
    const container = document.getElementById('container'); //html ref
    scene = new THREE.Scene();

    new NRRDLoader().load('nrrd/right.nrrd', function (volume) {
        // #1 : 3D volume
        volconfig = {renderstyle: 'iso', isothreshold: 0.15, colormap: 'gray' }

        let texture = new THREE.Data3DTexture(volume.data, volume.xLength, volume.yLength, volume.zLength);
        texture.format = THREE.RedFormat;
        texture.type = THREE.FloatType;
        texture.minFilter = texture.magFilter = THREE.LinearFilter;
        texture.needsUpdate = true;

        cmtextures = {
            viridis: new THREE.TextureLoader().load( 'textures/cm_viridis.png', render),
            gray: new THREE.TextureLoader().load( 'textures/cm_gray.png', render ),
            test: new THREE.TextureLoader().load( 'textures/floors/FloorsCheckerboard_S_Diffuse.jpg', render),
        };
        let shader = VolumeRenderShader1;
        let uniforms = THREE.UniformsUtils.clone(shader.uniforms);

        uniforms[ 'u_data' ].value = texture
        uniforms[ 'u_size' ].value.set( volume.xLength, volume.yLength, volume.zLength )
        uniforms[ 'u_renderstyle' ].value = volconfig.renderstyle == 'mip' ? 0 : 1 // 0: MIP, 1: ISO
        uniforms[ 'u_renderthreshold' ].value = volconfig.isothreshold // For ISO renderstyle
        uniforms[ 'u_cmdata' ].value = cmtextures[ volconfig.colormap ]

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
        // meshBrain.visible = false
        scene.add(meshBrain);

        // Add wireframe materials to visualize object mesh structure
        brainBox = new THREE.Mesh(geometry, material);
        brainBox.scale.set(0.5, 0.5, 0.5);
        brainBox.position.set(-(volume.xLength / 4),  - (volume.yLength / 4),  - (volume.zLength / 4))
        const wireframeMaterial = new THREE.MeshBasicMaterial({ color: 0x00ff00, wireframe: true });
        brainBox.material = wireframeMaterial;
        brainBox.visible = false
        scene.add(brainBox);

        // Optimization attempts
        // 1 Low level of details
        // meshBrain.geometry = simplifiedGeometry
        // Assuming you have a geometry named 'originalGeometry'
        // 2 Minimize draw calls ?
        // const mergedGeometry = new THREE.Geometry().fromBufferGeometry(geometry);
        // meshBrain.geometry = mergedGeometry;

        // End of NRRDLoader #1
        // #2: 2D slices viewer

        // x, y, z  planes
        const sliceX = volume.extractSlice( 'x', Math.floor( volume.RASDimensions[ 0 ] / 100 ) ) //
        const sliceY = volume.extractSlice( 'y', Math.floor( volume.RASDimensions[ 1 ] / 2 ) )
        const sliceZ = volume.extractSlice( 'z', Math.floor( volume.RASDimensions[ 2 ] / 100 ) ) //
        const slices = []; slices.push(sliceX, sliceY, sliceZ);
        for (let i = 0; i < slices.length; i++) {
            slices[i].mesh.material.opacity = 0.7
            slices[i].index = 0
            scene.add(slices[i].mesh)
            // slices[i].mesh.visible = true
        }
        slices[1].index=72
        // GUI 
        // 3D
        const model = gui.addFolder('3Dmodel');
        const visibilityBrain = { visible: meshBrain.visible };
        const visibilityBox = { visible: brainBox.visible };
        model.add(visibilityBrain, 'visible').name('3D Volume Visible').onChange(function () {
            meshBrain.visible = visibilityBrain.visible;
        });
        model.add(volconfig, 'colormap', { gray: 'gray', viridis: 'viridis', brain:'test'}).name('Volume colormap').onChange( updateUniforms )
        model.add(visibilityBox, 'visible').name('BBox Visible').onChange(function () {
            brainBox.visible = visibilityBox.visible;
            renderer.render(scene, camera);
        });
        
        // 2D 
        const planes = gui.addFolder( '2D Planes visibility' )
        const visControlY = { visible: true };
        planes.add( visControlY, 'visible' ).name( 'Coronal plane' ).onChange( function () {
            sliceY.mesh.visible = visControlY.visible;
        } );
        planes.add( sliceY, 'index', 0, volume.RASDimensions[ 1 ], 1 ).name( 'Coronal' ).onChange( function () {
            sliceY.repaint.call( sliceY );
        } );
         const visControlZ = { visible: true };
        planes.add( visControlZ, 'visible' ).name( 'Axial plane' ).onChange( function () {
            sliceZ.mesh.visible = visControlZ.visible;
        } );
        planes.add( sliceZ, 'index', 0, volume.RASDimensions[ 2 ], 1 ).name( 'Axial' ).onChange( function () {
            sliceZ.repaint.call( sliceZ );
        } );
        const visControlX = { visible: true };
        planes.add( visControlX, 'visible' ).name( 'Sagittal plane' ).onChange( function () {
            sliceX.mesh.visible = visControlX.visible;
        } );
        planes.add( sliceX, 'index', 0, volume.RASDimensions[ 0 ], 1 ).name( 'Sagittal' ).onChange( function () {
            sliceX.repaint.call( sliceX );
        } );

        planes.add( volume, 'lowerThreshold', volume.min, 70, 1 ).name( 'Color Low' ).onChange( function () {
            volume.repaintAllSlices();
        } );
        volume.upperThreshold = 70
        planes.add( volume, 'upperThreshold', volume.min, 70, 1 ).name( 'Color High' ).onChange( function () {
            volume.repaintAllSlices();
        } );
        // gui.add( volume, 'windowLow', volume.min, volume.max, 1 ).name( 'Window Low' ).onChange( function () {
        //     volume.repaintAllSlices();
        // } );
        // gui.add( volume, 'windowHigh', volume.min, volume.max, 1 ).name( 'Window High' ).onChange( function () {
        //     volume.repaintAllSlices();
        // } );

        // End of NRRDLoader #2 
    } );

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);
    container.appendChild(renderer.domElement);

    stats = new Stats();
    container.appendChild(stats.dom);

    // Cameras & controls
    const cams = gui.addFolder('Cameras');
    views.forEach((view, index) => {
        // cameras
        const camera = new THREE.OrthographicCamera(
            -view.width , view.width,
            view.height /2, -view.height /2,
            400, 3000
        );
        // old perspective Camera
        // const camera = new THREE.PerspectiveCamera(view.fov, window.innerWidth / window.innerHeight, 1, 10000);
        camera.position.fromArray(view.eye);
        camera.up.fromArray(view.up);
        camera.zoom = 0.002;
        view.camera = camera;

        // controls
        const controls = new OrbitControls(view.camera, renderer.domElement);
        controls.target.set(0, 0, 0); 
        controls.minDistance = 1000;
        controls.enableZoom = view.zoom;
        controls.enableRotate = view.rotate;
        orbitControls.push(controls); // Store the controls for each view

        const zoom = { zoomable: controls.enableZoom };
        const rotate = { rotable: controls.enableRotate };
        cams.add(zoom, 'zoomable').name(`Zoom ${view.name}`).onChange(function () {
            controls.enableZoom = zoom.zoomable;
        });
        cams.add(rotate, 'rotable').name(`Rotate ${view.name}`).onChange(function () {
            controls.enableRotate = rotate.rotable;
        });

        // helper for camera positions
        const cameraHelper = new THREE.CameraHelper(view.camera);
        scene.add(cameraHelper);
        cameraHelper.visible = false

        // Toggle controls attempts
        // const toggleControlsBtn = { toggle: () => toggleControls(index) };
        // cams.add(toggleControlsBtn, 'toggle').name(`Toggle Controls ${view.name}`); 

        // camera check
        // controls.addEventListener('change', function() {
        //     console.log('Camera position:', view.camera.zoom);
        // });
    });

    // Annotations
    // create annotations as volumes on depths of interest
    const createRectangularAnnotation = (width, height, depth) => {
        let geometry = new THREE.BoxGeometry(width, height, depth);
        let material = new THREE.MeshBasicMaterial({
            color: 0x966919,
            transparent: true,
            opacity: 0.6
        });
        return new THREE.Mesh(geometry, material);
    }
    
    let plane3D = new THREE.Object3D()
    scene.add(plane3D)
    // read json
    let readAnnot = (json) => {
        const vertices1 = []
        let annotationsArray = json.annotations;
        for (let i = 0; i < annotationsArray.length; i++) {
            let z1 = annotationsArray[i].points.z1 
            let z2 = annotationsArray[i].points.z2
            vertices1.push(annotationsArray[i].points.x1, annotationsArray[i].points.y1, z1*101.5/203 - 50.75)
            vertices1.push(annotationsArray[i].points.x2, annotationsArray[i].points.y2, z2*101.5/203 - 50.75)
        }
        return vertices1
    }
        
    fetch("nrrd/annot_sample.json")
        .then(response => response.json())
        .then(json => {
            let vertices = readAnnot(json);
            // console.log(vertices)
            for (let i = 0; i < vertices.length; i += 6) {
                let annotation = createRectangularAnnotation(2, 140, 2);
                annotation.position.set((vertices[i] + vertices[i + 3]) / 2, 0, (vertices[i + 2] + vertices[i + 5]) / 2);
                plane3D.add(annotation);
            }
        })

    const annots = gui.addFolder( 'Annotations' )
    const visibilityControlA = { visible: true }
    annots.add( visibilityControlA, 'visible' ).name( 'Annotations Visible' ).onChange( function () {
        plane3D.visible = visibilityControlA.visible
        renderer.render( scene, camera )
    } );

}

function updateSize() {
    if (windowWidth !== window.innerWidth || windowHeight !== window.innerHeight) {
        windowWidth = window.innerWidth;
        windowHeight = window.innerHeight;
        renderer.setSize(windowWidth, windowHeight);
    }
}

function updateUniforms() {
    material.uniforms[ 'u_renderstyle' ].value = volconfig.renderstyle == 'mip' ? 0 : 1; // 0: MIP, 1: ISO
    material.uniforms[ 'u_renderthreshold' ].value = volconfig.isothreshold; // For ISO renderstyle
    material.uniforms[ 'u_cmdata' ].value = cmtextures[ volconfig.colormap ];
    render();
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

const resetButton = { resetCamera: function() {
    views.forEach(view => {
        view.camera.zoom = 0.002;
        view.camera.position.fromArray(view.eye);
        view.camera.up.fromArray(view.up);
        view.camera.updateProjectionMatrix();
    });
}};
gui.add(resetButton, 'resetCamera').name('Reset Cameras');

init();
animate();
