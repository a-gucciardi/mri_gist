import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { NRRDLoader } from 'three/examples/jsm/loaders/NRRDLoader.js';
import { VolumeRenderShader1 } from 'three/examples/jsm/shaders/VolumeShader.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import Stats from 'three/examples/jsm/libs/stats.module.js';
import * as nifti from 'nifti-reader-js';
import * as fflate from 'fflate';

const VolumeViewer = ({ file, url, colormap = 'gray', renderStyle = 'iso', isoThreshold = 0.15 }) => {
    const containerRef = useRef(null);
    const rendererRef = useRef(null);
    const sceneRef = useRef(null);
    const requestRef = useRef(null);
    const controlsRef = useRef([]);

    useEffect(() => {
        if (!containerRef.current) return;

        // Initialize Stats
        const stats = new Stats();
        containerRef.current.appendChild(stats.dom);

        // Initialize Scene
        const scene = new THREE.Scene();
        sceneRef.current = scene;

        // Initialize Renderer
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
        containerRef.current.appendChild(renderer.domElement);
        rendererRef.current = renderer;

        // Initialize Cameras and Views
        // Using the same 3-view setup as original app.js
        const views = [
            {
                left: 0, bottom: 0, width: 0.5, height: 1.0,
                background: new THREE.Color().setRGB(0.5, 0.5, 0.7),
                eye: [0, 2000, 0], up: [0, 0, 1], fov: 45,
                zoom: true, rotate: true
            },
            {
                left: 0.5, bottom: 0, width: 0.5, height: 0.5,
                background: new THREE.Color().setRGB(0.7, 0.5, 0.5),
                eye: [2000, 0, 0], up: [0, 0, 1], fov: 30,
                zoom: false, rotate: false
            },
            {
                left: 0.5, bottom: 0.5, width: 0.5, height: 0.5,
                background: new THREE.Color().setRGB(0.5, 0.7, 0.7),
                eye: [0, 0, 2000], up: [0, -1, 0], fov: 60,
                zoom: false, rotate: false
            }
        ];

        // Setup Cameras and Controls
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
            controlsRef.current.push(controls);
        });

        // Animation Loop
        const animate = () => {
            requestRef.current = requestAnimationFrame(animate);

            const windowWidth = containerRef.current.clientWidth;
            const windowHeight = containerRef.current.clientHeight;

            // Handle Resize
            if (renderer.domElement.width !== windowWidth || renderer.domElement.height !== windowHeight) {
                renderer.setSize(windowWidth, windowHeight);
            }

            views.forEach((view) => {
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
            });
            stats.update();
        };
        animate();

        return () => {
            cancelAnimationFrame(requestRef.current);
            if (rendererRef.current) {
                if (containerRef.current && containerRef.current.contains(rendererRef.current.domElement)) {
                    containerRef.current.removeChild(rendererRef.current.domElement);
                }
                rendererRef.current.dispose();
            }
            if (stats.dom && containerRef.current && containerRef.current.contains(stats.dom)) {
                containerRef.current.removeChild(stats.dom);
            }
        };
    }, []);

    // Load Volume when file changes
    useEffect(() => {
        if (!file && !url) return;
        if (!sceneRef.current) return;

        const loadVolume = async () => {
            // Clear previous meshes
            const scene = sceneRef.current;
            scene.children.forEach(child => {
                if (child.isMesh) scene.remove(child);
            });

            // Determine URL
            // If url prop is provided (from backend integration), use it.
            // Function fallback for legacy/dev: /static/nrrd/${file}
            // BUT, if file is absolute path or URL, use it directly? 
            // Our App.jsx passes url from fileMap.

            let loadUrl = url;
            if (!loadUrl && file) {
                // Fallback for dev mode without backend or legacy behavior
                // file is just filename
                loadUrl = `/static/nrrd/${file}`;
            }
            if (!loadUrl) return;

            // Check extension
            const isNifti = loadUrl.endsWith('.nii') || loadUrl.endsWith('.nii.gz');

            if (isNifti) {
                try {
                    const response = await fetch(loadUrl);
                    const buffer = await response.arrayBuffer();
                    let data = buffer;

                    if (nifti.isCompressed(data)) {
                        data = nifti.decompress(data);
                    } else if (loadUrl.endsWith('.gz')) {
                        // Fallback manual decompression if nifti reader didn't detect or handle
                        data = fflate.gunzipSync(new Uint8Array(data)).buffer;
                    }

                    if (nifti.isNIFTI(data)) {
                        const header = nifti.readHeader(data);
                        const image = nifti.readImage(header, data);

                        // Dimensions
                        const dims = header.dims; // [dim, x, y, z, t, ...]
                        const xLength = dims[1];
                        const yLength = dims[2];
                        const zLength = dims[3];

                        // Convert to TypedArray suitable for Texture
                        // nifti reader returns ArrayBuffer usually
                        let typedData;
                        if (header.datatypeCode === nifti.NIFTI1.TYPE_UINT8) {
                            typedData = new Uint8Array(image);
                        } else if (header.datatypeCode === nifti.NIFTI1.TYPE_INT16) {
                            typedData = new Int16Array(image);
                        } else if (header.datatypeCode === nifti.NIFTI1.TYPE_FLOAT32) {
                            typedData = new Float32Array(image);
                        } else {
                            // Fallback or assume float for now (Three uses floats often)
                            // Or let Data3DTexture handle it? 
                            // Need to align with what NRRDLoader produces.
                            // NRRD loader often normalizes or returns based on type.
                            // For simplicity: Int16 is common in MRI. Float32 is also common.
                            // Let's rely on standard mapping arrays.
                            // Actually, let's just cast to Float32 for rendering consistency if feasible
                            // But that doubles memory.
                            // Let's try native type.
                            typedData = new Float32Array(image); // Safer for shader
                        }

                        _createVolume(scene, typedData, xLength, yLength, zLength);

                    }
                } catch (e) {
                    console.error("Failed to load NIfTI:", e);
                }
            } else {
                // NRRD Loader
                new NRRDLoader().load(loadUrl, function (volume) {
                    _createVolume(scene, volume.data, volume.xLength, volume.yLength, volume.zLength);
                });
            }
        };

        const _createVolume = (scene, data, xLength, yLength, zLength) => {
            // Texture setup
            const texture = new THREE.Data3DTexture(data, xLength, yLength, zLength);
            texture.format = THREE.RedFormat;
            texture.type = THREE.FloatType; // We cast to Float32 above if Nifti, NRRD might match
            // If NRRD loader returns Int16, we might need to adjust texture type!
            // NRRDLoader usually returns TypedArray matching the file.
            // If data is Int16Array, we should use ShortType? Or does WebGL2 support it?
            // Safer to use Float type for volume rendering in three.js usually.
            // But let's check input data type.
            if (data instanceof Float32Array) texture.type = THREE.FloatType;
            else if (data instanceof Uint8Array) texture.type = THREE.UnsignedByteType;
            else if (data instanceof Int16Array) texture.type = THREE.ShortType; // WebGL2
            else texture.type = THREE.FloatType; // Fallback

            texture.minFilter = texture.magFilter = THREE.LinearFilter;
            texture.unpackAlignment = 1;
            texture.needsUpdate = true;

            // Colormaps
            const cmtextures = {
                viridis: new THREE.TextureLoader().load('/static/textures/cm_viridis.png'),
                gray: new THREE.TextureLoader().load('/static/textures/cm_gray.png'),
            };

            // Shader Material
            const shader = VolumeRenderShader1;
            const uniforms = THREE.UniformsUtils.clone(shader.uniforms);

            uniforms['u_data'].value = texture;
            uniforms['u_size'].value.set(xLength, yLength, zLength);
            uniforms['u_renderstyle'].value = renderStyle === 'mip' ? 0 : 1;
            uniforms['u_renderthreshold'].value = isoThreshold;
            uniforms['u_cmdata'].value = cmtextures[colormap];

            const material = new THREE.ShaderMaterial({
                uniforms: uniforms,
                vertexShader: shader.vertexShader,
                fragmentShader: shader.fragmentShader,
                side: THREE.BackSide,
                clipping: true
            });

            const geometry = new THREE.BoxGeometry(xLength, yLength, zLength);
            geometry.translate(xLength / 2 - 0.5, yLength / 2 - 0.5, zLength / 2 - 0.5);

            const meshBrain = new THREE.Mesh(geometry, material);
            meshBrain.scale.set(0.5, 0.5, 0.5);
            meshBrain.rotateZ(Math.PI);
            meshBrain.position.set(xLength / 4, yLength / 4, -(zLength / 4));
            scene.add(meshBrain);
        };

        loadVolume();

    }, [file, url, colormap, renderStyle, isoThreshold]);

    return <div ref={containerRef} style={{ width: '100%', height: '100%' }} />;
};

export default VolumeViewer;
