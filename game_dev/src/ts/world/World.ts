import * as THREE from 'three';
import * as CANNON from 'cannon';
import Swal from 'sweetalert2';
import * as $ from 'jquery';

import { CameraOperator } from '../core/CameraOperator';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass';
import { ShaderPass } from 'three/examples/jsm/postprocessing/ShaderPass';
import { FXAAShader  } from 'three/examples/jsm/shaders/FXAAShader';

import { Detector } from '../../lib/utils/Detector';
import { Stats } from '../../lib/utils/Stats';
import * as GUI from '../../lib/utils/dat.gui';
import { CannonDebugRenderer } from '../../lib/cannon/CannonDebugRenderer';
import * as _ from 'lodash';

import { InputManager } from '../core/InputManager';
import * as Utils from '../core/FunctionLibrary';
import { LoadingManager } from '../core/LoadingManager';
import { InfoStack } from '../core/InfoStack';
import { UIManager } from '../core/UIManager';
import { IWorldEntity } from '../interfaces/IWorldEntity';
import { IUpdatable } from '../interfaces/IUpdatable';
import { Character } from '../characters/Character';
import { Path } from './Path';
import {Vase} from './Vase';
import { CollisionGroups } from '../enums/CollisionGroups';
import { BoxCollider } from '../physics/colliders/BoxCollider';
import { TrimeshCollider } from '../physics/colliders/TrimeshCollider';
import { Vehicle } from '../vehicles/Vehicle';
import { Scenario } from './Scenario';
import { Sky } from './Sky';
import { Ocean } from './Ocean';

import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader'
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader'

import { NeRFShader } from '../../lib/shaders/NeRFShader'
import { Vector3 } from 'three';
import { SimplifyModifier } from '../../lib/utils/THREE.SimplifyModifiers.js'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';
import {Geometry, Face3} from 'three/examples/jsm/deprecated/Geometry'
import { ConvexCollider } from '../physics/colliders/ConvexCollider';
import { SphereCollider } from '../physics/colliders/SphereCollider';
// import express from 'express';
// import cors from 'cors';

// const app = express();
// app.use(cors());

const SCALE = 1;
const external = './assets'
export class World
{
	public renderer: THREE.WebGLRenderer;
	public camera: THREE.PerspectiveCamera;
	public composer: any;
	public stats: Stats;
	public graphicsWorld: THREE.Scene;
	public sky: Sky;
	public physicsWorld: CANNON.World;
	public parallelPairs: any[];
	public physicsFrameRate: number;
	public physicsFrameTime: number;
	public physicsMaxPrediction: number;
	public clock: THREE.Clock;
	public renderDelta: number;
	public logicDelta: number;
	public requestDelta: number;
	public sinceLastFrame: number;
	public justRendered: boolean;
	public params: any;
	public inputManager: InputManager;
	public cameraOperator: CameraOperator;
	public timeScaleTarget: number = 1;
	public console: InfoStack;
	public cannonDebugRenderer: CannonDebugRenderer;
	public scenarios: Scenario[] = [];
	public characters: Character[] = [];
	public vehicles: Vehicle[] = [];
	public paths: Path[] = [];
	public scenarioGUIFolder: any;
	public updatables: IUpdatable[] = [];
	public camera_projmat: number[];
	public kitti_uv_models_loaded=[];
	public kitti_uvmodel_num=0;
	public kitti_uv_models=[];
	public intrinsic;
	public fx;
	public fy;
	public cx;
	public cy;
	public progressBar;
	public progress={};
	public onload=0;
	public total_load=0;
	public interface_loaded=false;
	public loadingManager=null;
	public total_load_progress = {};
	public base_pos;
	private lastScenarioID: string;
	public character_spawn;
	public base_rot;

	constructor(worldScenePath?: any)
	{
		const scope = this;

		// config
		let width = 1296;
		let height = 840;
		var fx = 961.2247;
		var fy = 963.0891;
		var cx = 648.375;
		var cy = 420.125;
		var f = 500.0;
		var n = 0.02;
		var gravity = 2.0;
		this.base_rot = Math.PI/2;
		let rendering_meshes = [
			external+"/garden/"
		]
		let collision_models = [
			external+"/garden/decomp_mesh_0/"
		]
		this.character_spawn = {
			'position_x': 0,
			'position_y': 4,
			'position_z': 0,
			'scale_x': 0.01,
			'scale_y': 0.01,
			'scale_z': 0.01,
		}

		// WebGL not supported
		if (!Detector.webgl)
		{
			Swal.fire({
				icon: 'warning',
				title: 'WebGL compatibility',
				text: 'This browser doesn\'t seem to have the required WebGL capabilities. The application may not work correctly.',
				footer: '<a href="https://get.webgl.org/" target="_blank">Click here for more information</a>',
				showConfirmButton: false,
				buttonsStyling: false
			});
		}

		// Renderer
		let pixelRatio = 1;
		this.renderer = new THREE.WebGLRenderer(
			{
			powerPreference: 'high-performance',
			precision: 'highp',
			}
		);

		this.renderer.setPixelRatio(pixelRatio);
		this.renderer.setSize(width, height);
		this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
		this.renderer.toneMappingExposure = 1.0;
		this.renderer.shadowMap.enabled = false;
		this.renderer.outputEncoding = THREE.sRGBEncoding;
		this.generateHTML();

		// Three.js scene
		this.graphicsWorld = new THREE.Scene();
		// this.camera = new THREE.PerspectiveCamera(80, width/height, 0.01, 1000);
		this.camera = new THREE.PerspectiveCamera();

		this.fx = fx;
		this.fy = fy;
		this.cx = cx;
		this.cy = cy;

		var n00 = 2.0 * fx / width;
		var n11 = 2.0 * fy / height;
		var n02 = 1.0 - 2.0 * cx / width;
		var n12 = 2.0 * cy / height - 1.0;
		var n32 = -1.0;
		var n22 = (f + n) / (n - f);
		var n23 = (2 * f * n) / (n - f);
		this.camera_projmat = [n00, 0, n02, 0, 
								0, n11, n12, 0, 
								0, 0, n22, n23, 
								0, 0, n32, 0];
		this.intrinsic = new THREE.Matrix3().fromArray([
			fx, 0, cx,
			0, fy, cy,
			0, 0, 1
		]).transpose();

		this.camera.projectionMatrix = new THREE.Matrix4().fromArray(this.camera_projmat).transpose();
		this.camera.projectionMatrixInverse = new THREE.Matrix4().fromArray(this.camera_projmat).transpose().invert();
		// Passes
		let renderPass = new RenderPass( this.graphicsWorld, this.camera );
		let fxaaPass = new ShaderPass( FXAAShader );

		// FXAA
		// let pixelRatio = this.renderer.getPixelRatio();
		fxaaPass.material['uniforms'].resolution.value.x = 1 / ( width * pixelRatio );
		fxaaPass.material['uniforms'].resolution.value.y = 1 / ( height * pixelRatio );

		// Composer
		this.composer = new EffectComposer( this.renderer );
		this.composer.addPass( renderPass );
		this.composer.addPass( fxaaPass );

		// Physics
		this.physicsWorld = new CANNON.World();
		this.physicsWorld.gravity.set(0, -gravity, 0);
		this.physicsWorld.broadphase = new CANNON.SAPBroadphase(this.physicsWorld);
		this.physicsWorld.solver.iterations = 10;
		this.physicsWorld.allowSleep = true;

		const defaultMaterial = new CANNON.Material('default')

		const defaultContactMaterial = new CANNON.ContactMaterial(
		defaultMaterial,
		defaultMaterial,
		{
		}
		)
		this.physicsWorld.addContactMaterial(defaultContactMaterial)
		this.physicsWorld.defaultContactMaterial = defaultContactMaterial

		this.parallelPairs = [];
		this.physicsFrameRate = 60;
		this.physicsFrameTime = 1 / this.physicsFrameRate;
		this.physicsMaxPrediction = this.physicsFrameRate;
		// RenderLoop
		this.clock = new THREE.Clock();
		this.renderDelta = 0;
		this.logicDelta = 0;
		this.sinceLastFrame = 0;
		this.justRendered = false;

		// Stats (FPS, Frame time, Memory)
		this.stats = Stats();
		// Create right panel GUI
		this.createParamsGUI(scope);

		// Initialization
		this.inputManager = new InputManager(this, this.renderer.domElement);
		this.cameraOperator = new CameraOperator(this, this.camera, this.params.Mouse_Sensitivity);
		this.sky = new Sky(this);
		
		let base_pos = {
			position_x: 0,
			position_y: 0,
			position_z: 0,
			rotate_x: 90,
			rotate_y: 0,
			rotate_z: 0,
		};

		this.base_pos = base_pos;
		this.kitti_uvmodel_num = rendering_meshes.length;
		for(let i=0; i < this.kitti_uvmodel_num;i++){
			this.kitti_uv_models_loaded.push(false);
			this.init_uvmapping(this, rendering_meshes[i], i);
		}

		for (let i=0; i < collision_models.length; i++){
			this.init_convex_collision(this, collision_models[i], base_pos, {mass: 0});
		}

		// Load scene if path is supplied
		if (worldScenePath !== undefined)
		{
			let loadingManager = new LoadingManager(this);
			this.loadingManager = loadingManager;
			loadingManager.onFinishedCallback = () =>
			{
				this.update(1, 1);
				this.setTimeScale(1);
				var prompt = 'Feel free to explore the world based of Video2Game';
				Swal.fire({
					title: 'Welcome to Video2Game Demo! ',
					html: prompt,
					footer: '<a href="https://video2game.github.io/" target="_blank">Video2Game GitHub page</a>',
					confirmButtonText: 'Okay',
					buttonsStyling: false,
					onClose: () => {
						UIManager.setUserInterfaceVisible(true);
					}
				});
			};
			loadingManager.loadGLTF(worldScenePath, (gltf) =>
				{
					this.loadScene(loadingManager, gltf);
				}
			);
		}
		else
		{
			UIManager.setUserInterfaceVisible(true);
			UIManager.setLoadingScreenVisible(false);
			Swal.fire({
				icon: 'success',
				title: 'Hello world!',
				text: 'Empty Video2Game world was succesfully initialized. Enjoy the blueness of the sky.',
				buttonsStyling: false
			});
		}

		this.renderer.compile(this.graphicsWorld, this.camera);
		this.render(this);
	}

	
	public update_progress_bar(world: World){
		var total_bar = 50.1;
		var total_bar_int = 50;
		var ok = '🟩';
		var nok = '🟥';
	    var progressBar = document.getElementById('progressBarw');
		var load_progress = 0.0;
		for (var load_i in world.total_load_progress){
			world.total_load_progress[load_i].forEach(load_i_j => {
				load_progress += load_i_j;
			})
		}
		var okblock_f = (load_progress / (this.total_load*100.0)) * total_bar;
		var okblock = 0;
		var pbstr = '';
		for (var oki=1; oki<okblock_f; oki+=1){
			pbstr = pbstr+ok;
			okblock += 1;
		}
		for (var noki=0; noki<total_bar-okblock; noki+=1){
			pbstr = pbstr+nok;
		}
		// console.log(pbstr);
		progressBar.innerText = pbstr;
		if(okblock == total_bar_int){
			progressBar.innerText = "";
		}
	}
	
	
	private init_uvmapping(world: World, path: string, idx: number): void{

		
		function createNetworkWeightTexture(network_weights) {
			let width = network_weights.length;
			let height = network_weights[0].length;
			
			let weightsData = new Float32Array(width * height);
			for (let co = 0; co < height; co++) {
				for (let ci = 0; ci < width; ci++) {
					let index = co * width + ci; // column-major
					let weight = network_weights[ci][co];
					weightsData[index] = weight;
				}
			}
			
			let width_pad = width + (4 - width % 4); // make divisible by 4
			let weightsData_pad = new Float32Array(width_pad * height);
			for (let j = 0; j < width_pad; j += 4) {
				for (let i = 0; i < height; i++) {
					for (let c = 0; c < 4; c++) {
						if (c + j >= width) { 
							weightsData_pad[j * height + i * 4 + c] = 0.0; // zero padding
						} else {
							weightsData_pad[j * height + i * 4 + c] = weightsData[j + i * width + c];
						}
					}
				}
			}

			let texture = new THREE.DataTexture(weightsData_pad, 1, width_pad * height / 4, THREE.RGBAFormat, THREE.FloatType);
			texture.magFilter = THREE.LinearFilter;
			texture.minFilter = THREE.LinearFilter;
			texture.needsUpdate = true;
			return texture;
		}

		function createViewDependenceFunctions(network_weights) {
		
			let channelsZero = network_weights['net.0.weight'].length;
			let channelsOne = network_weights['net.1.weight'].length;
			let channelsTwo = network_weights['net.1.weight'][0].length;

			console.log('[INFO] load MLP: ', channelsZero, channelsOne)

			let RenderFragShader = NeRFShader.RenderFragShader_template.trim().replace(new RegExp('NUM_CHANNELS_ZERO', 'g'), channelsZero);
			RenderFragShader = RenderFragShader.replace(new RegExp('NUM_CHANNELS_ONE', 'g'), channelsOne);
			RenderFragShader = RenderFragShader.replace(new RegExp('NUM_CHANNELS_TWO', 'g'), channelsTwo);

			return RenderFragShader;
		}

		fetch(path+'mlp.json').then(response => { return response.json(); }).then(network_weights => {

            console.log("[INFO] loading:", idx);

			let cascade = network_weights['cascade'];
            world.kitti_uv_models[idx] = [];
			world.initProgressBar(path, cascade);
			world.total_load_progress[idx] = new Array(cascade*3).fill(0);

            for (let cas = 0; cas < cascade; cas++) {

                // load feature texture
                let tex0 = new THREE.TextureLoader().load(path+'feat0_'+cas.toString()+'.png', object => {
                    console.log('[INFO] loaded diffuse tex:', idx, cas);
					world.updateProgressBar(path, cas*3+1);
					world.total_load_progress[idx][cas*3+1] = 100;
					world.update_progress_bar(world);
                },
				(xhr) => {
					world.total_load_progress[idx][cas*3+1] = (xhr.loaded / xhr.total) * 100;
					world.update_progress_bar(world);
					// // console.log((xhr.loaded / xhr.total) * 100 + '% loaded', idx);
				});
                let tex1 = new THREE.TextureLoader().load(path+'feat1_'+cas.toString()+'.png', object => {
                    console.log('[INFO] loaded specular tex:', idx, cas);
					world.updateProgressBar(path, cas*3+2);
					world.total_load_progress[idx][cas*3+2] = 100;
					world.update_progress_bar(world);
                },
				(xhr) => {
					world.total_load_progress[idx][cas*3+2] = (xhr.loaded / xhr.total) * 100;
					world.update_progress_bar(world);
					// // console.log((xhr.loaded / xhr.total) * 100 + '% loaded', idx);
				});

                // load MLP
                let RenderFragShader = createViewDependenceFunctions(network_weights);
                let weightsTexZero = createNetworkWeightTexture(network_weights['net.0.weight']);
                let weightsTexOne = createNetworkWeightTexture(network_weights['net.1.weight']);
				
				var side = THREE.FrontSide;
				tex0.magFilter = THREE.LinearFilter;
				tex0.minFilter = THREE.LinearFilter;
				tex1.magFilter = THREE.LinearFilter;
				tex1.minFilter = THREE.LinearFilter;
                let newmat = new THREE.ShaderMaterial({
					side: side,
                    vertexShader: NeRFShader.RenderVertShader.trim(),
                    fragmentShader: RenderFragShader,
                    uniforms: {
                        'mode': { value: 0 },
                        'tDiffuse': { value: tex0 },
                        'tSpecular': { value: tex1 },
                        'weightsZero': { value: weightsTexZero },
                        'weightsOne': { value: weightsTexOne },
						'gmatrix_inv': {'value': new THREE.Matrix4()},
						'intrinsic': {'value': world.intrinsic},
						'c2w_T': {'value': new THREE.Matrix3()},
						'fx': {'value': world.fx},
						'fy': {'value': world.fy},
						'cx': {'value': world.cx},
						'cy': {'value': world.cy},						
                    },
                });
            
                // load obj
                new PLYLoader().load(path+'mesh_'+cas.toString()+'.ply', geo => {
					let child = new THREE.Mesh(geo);
					child.receiveShadow = false;
					child.castShadow = false;
					child.position.x = 0;
					child.position.y = 0;
					child.position.z = 0;
					child.rotateX(world.base_rot);
					child.scale.x = 1;
					child.scale.y = 1;
					child.scale.z = 1;
					
					child.frustumCulled = false;

					(child as THREE.Mesh).material = newmat;
					world.kitti_uv_models[idx].push(child);

                    console.log('[INFO] loaded mesh:', idx, cas);
					world.updateProgressBar(path, cas*3);
	
                    world.graphicsWorld.add(child);
                },
				(xhr) => {
					// // console.log((xhr.loaded / xhr.total) * 100 + '% loaded', idx);
					world.total_load_progress[idx][cas*3] = (xhr.loaded / xhr.total) * 100;
					world.update_progress_bar(world);
					
				});

            }

            world.kitti_uv_models_loaded[idx] = true;
        });
	}
	
	private init_convex_collision(world: World, path: string, mesh_pos, options): void {
		let collider = new CANNON.Body(options);
		var vis = false;
		var vis_list = [];
		if(path == 'box'){
			let offset = new CANNON.Vec3(
				mesh_pos.position_x, 
				mesh_pos.position_y, 
				mesh_pos.position_z,
			);
			let quat = new CANNON.Quaternion(mesh_pos.rotate_x, mesh_pos.rotate_y, mesh_pos.rotate_z, mesh_pos.rotate_w);
			let box_shape = new CANNON.Box(new CANNON.Vec3(mesh_pos.scale_x, mesh_pos.scale_y, mesh_pos.scale_z));
			collider.addShape(box_shape, offset, quat);
			let basequat = new THREE.Quaternion().setFromEuler(new THREE.Euler(world.base_pos.rotate_x*Math.PI/180.));

			var quaternion_x = basequat.x;
			var quaternion_y = basequat.y;
			var quaternion_z = basequat.z;
			var quaternion_w = basequat.w;

			collider.position.x = world.base_pos.position_x;
			collider.position.y = world.base_pos.position_y;
			collider.position.z = world.base_pos.position_z;
			collider.quaternion.x = quaternion_x;
			collider.quaternion.y = quaternion_y;
			collider.quaternion.z = quaternion_z;
			collider.quaternion.w = quaternion_w;
			
			world.physicsWorld.addBody(collider);
		}
		else{
		fetch(path+"center.json").then(response => {
			return response.json();
		}).then(json => {
			let quaternion_x;
			let quaternion_y;
			let quaternion_z;
			let quaternion_w;

			let offset_list = [];

			json.forEach((element, idx) => {
				// console.log(idx)
				var off_x = element[0];
				var off_y = element[1];
				var off_z = element[2];
				let offset = new CANNON.Vec3(off_x, off_y, off_z);
				offset_list.push(offset);
			});
			var sceneFile = path+"decomp.glb";

			var loader = new GLTFLoader();
			let world = this;
			
			let rotate_quaternion = new THREE.Quaternion().setFromEuler(new THREE.Euler(mesh_pos.rotate_x*Math.PI/180.));
			quaternion_x = rotate_quaternion.x;
			quaternion_y = rotate_quaternion.y;
			quaternion_z = rotate_quaternion.z;
			quaternion_w = rotate_quaternion.w;

			loader.load(
				sceneFile,
				function (gltf) {
					gltf.scene.traverse(function (child) {
						if ((child as THREE.Mesh).isMesh) {
							var seq_n = parseInt(child.name.substring(5));
							var offset = offset_list[seq_n];
							child = (child as THREE.Mesh);
							let phys = new ConvexCollider(child, {});
							phys.body.shapes.forEach((shape) => {
								collider.addShape(shape, offset);
							});
							if (vis){
								let child_c = (child as THREE.Mesh).clone();
								var trans_matrix = new THREE.Matrix4().makeTranslation(offset.x, offset.y, offset.z);
								var trans_matrix2 = new THREE.Matrix4().makeTranslation(mesh_pos.position_x, mesh_pos.position_y, mesh_pos.position_z);
								var rotate_matrix = new THREE.Matrix4().makeRotationFromQuaternion(new THREE.Quaternion(quaternion_x, quaternion_y, quaternion_z, quaternion_w));
								child_c.matrixAutoUpdate=false;
								child_c.matrix = new THREE.Matrix4().multiplyMatrices(trans_matrix2,new THREE.Matrix4().multiplyMatrices(rotate_matrix,trans_matrix));
								child_c.material = new THREE.LineBasicMaterial({'linewidth': 0.001});
								world.graphicsWorld.add(child_c);
							}
						}
					})
				},
				(xhr) => {
				},
				(error) => {
					console.log(error);
				}
			);

			
			collider.position.x = mesh_pos.position_x;
			collider.position.y = mesh_pos.position_y;
			collider.position.z = mesh_pos.position_z;
			collider.quaternion.x = quaternion_x;
			collider.quaternion.y = quaternion_y;
			collider.quaternion.z = quaternion_z;
			collider.quaternion.w = quaternion_w;

			world.physicsWorld.addBody(collider);
		});
		}
	}

	public update(timeStep: number, unscaledTimeStep: number): void
	{	
		this.updatePhysics(timeStep);

		// Update registred objects
		this.updatables.forEach((entity) => {
			entity.update(timeStep, unscaledTimeStep);
		});

		// Lerp time scale
		this.params.Time_Scale = THREE.MathUtils.lerp(this.params.Time_Scale, this.timeScaleTarget, 0.2);

		// Physics debug
		if (this.params.Debug_Physics && this.cannonDebugRenderer != undefined) this.cannonDebugRenderer.update();

		this.camera.projectionMatrix  = new THREE.Matrix4().fromArray(this.camera_projmat).transpose();
		this.camera.projectionMatrixInverse  = new THREE.Matrix4().fromArray(this.camera_projmat).transpose().invert();

		let scale = SCALE;
		let scene_trans = new THREE.Vector3(0, 0, 0);
		let rx = new THREE.Matrix4().makeRotationX(this.base_rot);
		let rx_3 = new THREE.Matrix3().fromArray([
			rx.elements[0], rx.elements[1], rx.elements[2],
			rx.elements[4], rx.elements[5], rx.elements[6],
			rx.elements[8], rx.elements[9], rx.elements[10]
		]);

		var p00, p10, p20, p30;
		var p01, p11, p21, p31;
		var p02, p12, p22, p32;
		var p03, p13, p23, p33;
		
		[p00, p10, p20, p30,
			p01, p11, p21, p31,
			p02, p12, p22, p32,
			p03, p13, p23, p33] 
		= this.camera.matrixWorld.elements;
		
		// trans
		let trans_inv = new THREE.Vector3(p03, p13, p23).add(scene_trans.multiplyScalar(-1)).multiplyScalar(1/scale).applyMatrix4(rx.clone().invert())
		
		let c2w_rot_inv = new THREE.Matrix3().fromArray([
			p00, p01, p02,
			p10, p11, p12,
			p20, p21, p22
		]).transpose();
		c2w_rot_inv = new THREE.Matrix3().multiplyMatrices(rx_3.clone().invert(), c2w_rot_inv);
		
		var c00, c10, c20;
		var c01, c11, c21;
		var c02, c12, c22;
		[
			c00, c10, c20,
			c01, c11, c21,
			c02, c12, c22
		] = c2w_rot_inv.elements;
		c2w_rot_inv = new THREE.Matrix3().fromArray([
			c00, -c01, -c02,
			c10, -c11, -c12,
			c20, -c21, -c22
		]).transpose()

		let gmatrix_noc_inv = new THREE.Matrix4().fromArray([
			c00, c01, c02, trans_inv.x,
			c10, c11, c12, trans_inv.y,
			c20, c21, c22, trans_inv.z,
			0, 0, 0, 1
		]).transpose();
		
		this.kitti_uv_models.forEach((model_list, idx) => {
			// console.log("kitti_uv_models: ", idx);
            if (this.kitti_uv_models_loaded[idx] == true){
				// console.log("kitti_uv_models_loaded: ", idx);
				// console.log("model_list: ", model_list.length);
				model_list.forEach(model => {
					// console.log("model: ", model.position);
					model.frustumCulled = true;
					(model.material as THREE.RawShaderMaterial).uniforms['gmatrix_inv']['value'] = gmatrix_noc_inv.clone().invert();
					(model.material as THREE.RawShaderMaterial).uniforms['c2w_T']['value'] = c2w_rot_inv.clone().transpose();
				})
			}
			
		})

		this.renderer.shadowMap.enabled = false;

	}

	public updatePhysics(timeStep: number): void
	{
		// Step the physics world
		this.physicsWorld.step(this.physicsFrameTime, timeStep);

		this.characters.forEach((char) => {
			// console.log("char: ", char.characterCapsule.body.position);
			if (this.isOutOfBounds(char.characterCapsule.body.position))
			{
				this.outOfBoundsRespawn(char.characterCapsule.body);
			}
		});

		this.vehicles.forEach((vehicle) => {
			if (this.isOutOfBounds(vehicle.rayCastVehicle.chassisBody.position))
			{
				let worldPos = new THREE.Vector3();
				vehicle.spawnPoint.getWorldPosition(worldPos);
				worldPos.y += 1;
				this.outOfBoundsRespawn(vehicle.rayCastVehicle.chassisBody, Utils.cannonVector(worldPos));
			}
		});

	}

	public isOutOfBounds(position: CANNON.Vec3): boolean
	{
		let inside = position.x > -211.882 && position.x < 211.882 &&
					position.z > -169.098 && position.z < 153.232 &&
					position.y > -10.0;

		return !inside;
	}

	public outOfBoundsRespawn(body: CANNON.Body, position?: CANNON.Vec3): void
	{
		let newPos = position || new CANNON.Vec3(0, 16, 0);
		let newQuat = new CANNON.Quaternion(0, 0, 0, 1);

		body.position.copy(newPos);
		body.interpolatedPosition.copy(newPos);
		body.quaternion.copy(newQuat);
		body.interpolatedQuaternion.copy(newQuat);
		body.velocity.setZero();
		body.angularVelocity.setZero();
	}
		
		
	public initProgressBar(name, length) {
	    this.progressBar = document.getElementById('progressBar');
	    this.progress[name] = new Array(length * 3).fill('🔴');
	    this.progressBar.innerText = Object.keys(this.progress).map(key => this.progress[key].join('')).join('|');
		this.onload += (length*3);
		this.total_load += (length*3);
	}
	public updateProgressBar(name, index) {
	    this.progressBar = document.getElementById('progressBar');
	    this.progress[name][index] = '🟢';
	    this.progressBar.innerText = Object.keys(this.progress).map(key => this.progress[key].join('')).join('|');
		this.onload -= 1;
		let all_loaded = (this.onload == 0);
		// console.log(this.progressBar.innerText);
		if (this.loadingManager != null && all_loaded && this.interface_loaded ){
			if (this.loadingManager.onFinishedCallback !== undefined) 
			{
				this.loadingManager.onFinishedCallback();
			}
			else
			{
				UIManager.setUserInterfaceVisible(true);
			}

			UIManager.setLoadingScreenVisible(false);
			this.progressBar.innerText = "";
			var progressBar = document.getElementById('progressBarw');
			progressBar.innerText = "";
		}
	}

	/**
	 * Rendering loop.
	 * Implements fps limiter and frame-skipping
	 * Calls world's "update" function before rendering.
	 * @param {World} world 
	 */
	public render(world: World): void
	{
		this.requestDelta = this.clock.getDelta();

		requestAnimationFrame(() =>
		{
			world.render(world);
		});

		// Getting timeStep
		let unscaledTimeStep = (this.requestDelta + this.renderDelta + this.logicDelta) ;
		let timeStep = unscaledTimeStep * this.params.Time_Scale;
		timeStep = Math.min(timeStep, 1 / 30);    // min 30 fps

		// Logic
		world.update(timeStep, unscaledTimeStep);

		// Measuring logic time
		this.logicDelta = this.clock.getDelta();

		// Frame limiting
		let interval = 1 / 60;
		this.sinceLastFrame += this.requestDelta + this.renderDelta + this.logicDelta;
		this.sinceLastFrame %= interval;

		// Stats end
		this.stats.end();
		this.stats.begin();

		// Actual rendering with a FXAA ON/OFF switch
		this.renderer.clear(true, true, true);
		if (this.params.FXAA) this.composer.render();
		else this.renderer.render(this.graphicsWorld, this.camera);

		// Measuring render time
		this.renderDelta = this.clock.getDelta();
	}

	public setTimeScale(value: number): void
	{
		this.params.Time_Scale = value;
		this.timeScaleTarget = value;
	}

	public add(worldEntity: IWorldEntity): void
	{
		worldEntity.addToWorld(this);
		this.registerUpdatable(worldEntity);
	}

	public registerUpdatable(registree: IUpdatable): void
	{
		this.updatables.push(registree);
		this.updatables.sort((a, b) => (a.updateOrder > b.updateOrder) ? 1 : -1);
	}

	public remove(worldEntity: IWorldEntity): void
	{
		worldEntity.removeFromWorld(this);
		this.unregisterUpdatable(worldEntity);
	}

	public unregisterUpdatable(registree: IUpdatable): void
	{
		_.pull(this.updatables, registree);
	}

	public loadScene(loadingManager: LoadingManager, gltf: any): void
	{
		gltf.scene.traverse((child) => {
			if (child.hasOwnProperty('userData'))
			{
				if (child.type === 'Mesh')
				{
					Utils.setupMeshProperties(child);
					// this.sky.csm.setupMaterial(child.material);

					if (child.material.name === 'ocean')
					{
						this.registerUpdatable(new Ocean(child, this));
					}
				}

				if (child.userData.hasOwnProperty('data'))
				{
					if (child.userData.data === 'physics')
					{
						if (child.userData.hasOwnProperty('type')) 
						{
							// Convex doesn't work! Stick to boxes!
							if (child.userData.type === 'box')
							{
								let phys = new BoxCollider({size: new THREE.Vector3(child.scale.x, child.scale.y, child.scale.z)});
								phys.body.position.copy(Utils.cannonVector(child.position));
								phys.body.quaternion.copy(Utils.cannonQuat(child.quaternion));
								phys.body.computeAABB();

								phys.body.shapes.forEach((shape) => {
									shape.collisionFilterMask = ~CollisionGroups.TrimeshColliders;
								});

								this.physicsWorld.addBody(phys.body);
							}
							else if (child.userData.type === 'trimesh')
							{
								let phys = new TrimeshCollider(child, {});
								this.physicsWorld.addBody(phys.body);
							}

							child.visible = false;
						}
					}

					if (child.userData.data === 'path')
					{
						this.paths.push(new Path(child));
					}

					if (child.userData.data === 'scenario')
					{
						this.scenarios.push(new Scenario(child, this));
					}
				}
			}
		});

		this.graphicsWorld.add(gltf.scene);

		// Launch default scenario
		let defaultScenarioID: string;
		for (const scenario of this.scenarios) {
			if (scenario.default) {
				defaultScenarioID = scenario.id;
				break;
			}
		}
		if (defaultScenarioID !== undefined) this.launchScenario(defaultScenarioID, loadingManager);
	}
	
	public launchScenario(scenarioID: string, loadingManager?: LoadingManager): void
	{
		this.lastScenarioID = scenarioID;

		this.clearEntities();

		// Launch default scenario
		if (!loadingManager) loadingManager = new LoadingManager(this);
		for (const scenario of this.scenarios) {
			if (scenario.id === scenarioID || scenario.spawnAlways) {
				scenario.launch(loadingManager, this);
			}
		}
	}

	public restartScenario(): void
	{
		if (this.lastScenarioID !== undefined)
		{
			document.exitPointerLock();
			this.launchScenario(this.lastScenarioID);
		}
		else
		{
			console.warn('Can\'t restart scenario. Last scenarioID is undefined.');
		}
	}

	public clearEntities(): void
	{
		for (let i = 0; i < this.characters.length; i++) {
			this.remove(this.characters[i]);
			i--;
		}

		for (let i = 0; i < this.vehicles.length; i++) {
			this.remove(this.vehicles[i]);
			i--;
		}
	}

	public scrollTheTimeScale(scrollAmount: number): void
	{
		// Changing time scale with scroll wheel
		const timeScaleBottomLimit = 0.003;
		const timeScaleChangeSpeed = 1.3;
	
		if (scrollAmount > 0)
		{
			this.timeScaleTarget /= timeScaleChangeSpeed;
			if (this.timeScaleTarget < timeScaleBottomLimit) this.timeScaleTarget = 0;
		}
		else
		{
			this.timeScaleTarget *= timeScaleChangeSpeed;
			if (this.timeScaleTarget < timeScaleBottomLimit) this.timeScaleTarget = timeScaleBottomLimit;
			this.timeScaleTarget = Math.min(this.timeScaleTarget, 1);
		}
	}

	public updateControls(controls: any): void
	{
		let html = '';
		html += '<h2 class="controls-title">Controls:</h2>';

		controls.forEach((row) =>
		{
			html += '<div class="ctrl-row">';
			row.keys.forEach((key) => {
				if (key === '+' || key === 'and' || key === 'or' || key === '&') html += '&nbsp;' + key + '&nbsp;';
				else html += '<span class="ctrl-key">' + key + '</span>';
			});

			html += '<span class="ctrl-desc">' + row.desc + '</span></div>';
		});

		document.getElementById('controls').innerHTML = html;
	}

	private generateHTML(): void
	{
		// Fonts
		$('head').append('<link href="https://fonts.googleapis.com/css2?family=Alfa+Slab+One&display=swap" rel="stylesheet">');
		$('head').append('<link href="https://fonts.googleapis.com/css2?family=Solway:wght@400;500;700&display=swap" rel="stylesheet">');
		$('head').append('<link href="https://fonts.googleapis.com/css2?family=Cutive+Mono&display=swap" rel="stylesheet">');

		// Loader
		$(`	<div id="loading-screen">
				<div id="loading-screen-background"></div>
				<h1 id="main-title" class="sb-font">Video2Game v1.0</h1>
				<div class="cubeWrap">
					<div class="cube">
						<div class="faces1"></div>
						<div class="faces2"></div>     
					</div> 
				</div> 
				<div id="loading-text">Loading...</div>
			</div>
		`).appendTo('body');

		// UI
		$(`	<div id="ui-container" style="display: none;">
				<div class="github-corner">
					<a href="https://video2game.github.io/" target="_blank" title="Video2Game">
						<svg viewbox="0 0 100 100" fill="currentColor">
							<title>Video2Game</title>
							<path d="M0 0v100h100V0H0zm60 70.2h.2c1 2.7.3 4.7 0 5.2 1.4 1.4 2 3 2 5.2 0 7.4-4.4 9-8.7 9.5.7.7 1.3 2
							1.3 3.7V99c0 .5 1.4 1 1.4 1H44s1.2-.5 1.2-1v-3.8c-3.5 1.4-5.2-.8-5.2-.8-1.5-2-3-2-3-2-2-.5-.2-1-.2-1
							2-.7 3.5.8 3.5.8 2 1.7 4 1 5 .3.2-1.2.7-2 1.2-2.4-4.3-.4-8.8-2-8.8-9.4 0-2 .7-4 2-5.2-.2-.5-1-2.5.2-5
							0 0 1.5-.6 5.2 1.8 1.5-.4 3.2-.6 4.8-.6 1.6 0 3.3.2 4.8.7 2.8-2 4.4-2 5-2z"></path>
						</svg>
					</a>
				</div>
				<div class="left-panel">
					<div id="controls" class="panel-segment flex-bottom"></div>
				</div>
			</div>
		`).appendTo('body');

		// Canvas
		document.body.appendChild(this.renderer.domElement);
		this.renderer.domElement.id = 'canvas';
	}

	private createParamsGUI(scope: World): void
	{
		this.params = {
			Pointer_Lock: true,
			Mouse_Sensitivity: 0.3,
			Time_Scale: 1,
			Shadows: false,
			FXAA: true,
			Debug_Physics: false,
			Debug_FPS: false,
			Sun_Elevation: 50,
			Sun_Rotation: 145,
			mass: 0,
		};

		const gui = new GUI.GUI();

		// Scenario
		this.scenarioGUIFolder = gui.addFolder('Scenarios');

		// Input
		let settingsFolder = gui.addFolder('Settings');
		settingsFolder.open();
		
		settingsFolder.add(this.params, 'Debug_Physics')
			.onChange((enabled) =>
			{
				if (enabled)
				{
					this.cannonDebugRenderer = new CannonDebugRenderer( this.graphicsWorld, this.physicsWorld );
				}
				else
				{
					this.cannonDebugRenderer.clearMeshes();
					this.cannonDebugRenderer = undefined;
				}

				scope.characters.forEach((char) =>
				{
					char.raycastBox.visible = enabled;
				});
			});
		settingsFolder.add(this.params, 'Debug_FPS')
			.onChange((enabled) =>
			{
				UIManager.setFPSVisible(enabled);
			});

		gui.open();
	}
}