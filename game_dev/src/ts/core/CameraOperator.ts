import * as THREE from 'three';
import * as Utils from './FunctionLibrary';
import { World } from '../world/World';
import { IInputReceiver } from '../interfaces/IInputReceiver';
import { KeyBinding } from './KeyBinding';
import { Character } from '../characters/Character';
import _ = require('lodash');
import { IUpdatable } from '../interfaces/IUpdatable';

function distance(point, rectangle) {
	const rectangleX1 = rectangle[0][0];
	const rectangleY1 = rectangle[0][1];
	const rectangleX2 = rectangle[1][0];
	const rectangleY2 = rectangle[1][1];
	const rectangleX3 = rectangle[2][0];
	const rectangleY3 = rectangle[2][1];
	const rectangleX4 = rectangle[3][0];
	const rectangleY4 = rectangle[3][1];

	var average_x = (rectangleX1 + rectangleX2 + rectangleX3 + rectangleX4) / 4;
	var average_y = (rectangleY1 + rectangleY2 + rectangleY3 + rectangleY4) / 4;

	return (point[0] - average_x)*(point[0] - average_x) + (point[1] - average_y)*(point[1] - average_y)
}

function pointInRectangle(point, rectangle):boolean {
	// Unpack the point coordinates
	const pointX = point[0];
	const pointY = point[1];
  
	// Unpack the rectangle coordinates
	const rectangleX1 = rectangle[0][0];
	const rectangleY1 = rectangle[0][1];
	const rectangleX2 = rectangle[1][0];
	const rectangleY2 = rectangle[1][1];
	const rectangleX3 = rectangle[2][0];
	const rectangleY3 = rectangle[2][1];
	const rectangleX4 = rectangle[3][0];
	const rectangleY4 = rectangle[3][1];
  
	// Calculate the vectors of the rectangle sides
	const vector1X = rectangleX2 - rectangleX1;
	const vector1Y = rectangleY2 - rectangleY1;
	const vector2X = rectangleX3 - rectangleX2;
	const vector2Y = rectangleY3 - rectangleY2;
  
	// Calculate the dot products
	const dotProduct1 = (pointX - rectangleX1) * vector1X + (pointY - rectangleY1) * vector1Y;
	const dotProduct2 = (pointX - rectangleX2) * vector2X + (pointY - rectangleY2) * vector2Y;
  
	// Check if the dot products have the same sign
	if ((dotProduct1 >= 0 && dotProduct2 >= 0)) {
	  // Calculate the vectors of the other two sides
	  const vector3X = rectangleX4 - rectangleX3;
	  const vector3Y = rectangleY4 - rectangleY3;
	  const vector4X = rectangleX1 - rectangleX4;
	  const vector4Y = rectangleY1 - rectangleY4;
  
	  // Calculate the dot products
	  const dotProduct3 = (pointX - rectangleX3) * vector3X + (pointY - rectangleY3) * vector3Y;
	  const dotProduct4 = (pointX - rectangleX4) * vector4X + (pointY - rectangleY4) * vector4Y;
  
	  // Check if the dot products have the same sign
	  if (dotProduct3 >= 0 && dotProduct4 >= 0) {
		return true; // Point is inside the rectangle
	  }
	}

	if ((dotProduct1 <= 0 && dotProduct2 <= 0)) {
		// Calculate the vectors of the other two sides
		const vector3X = rectangleX4 - rectangleX3;
		const vector3Y = rectangleY4 - rectangleY3;
		const vector4X = rectangleX1 - rectangleX4;
		const vector4Y = rectangleY1 - rectangleY4;
	
		// Calculate the dot products
		const dotProduct3 = (pointX - rectangleX3) * vector3X + (pointY - rectangleY3) * vector3Y;
		const dotProduct4 = (pointX - rectangleX4) * vector4X + (pointY - rectangleY4) * vector4Y;
	
		// Check if the dot products have the same sign
		if (dotProduct3 <= 0 && dotProduct4 <= 0) {
		  return true; // Point is inside the rectangle
		}
	}
  
	return false; // Point is outside the rectangle
}


export class CameraOperator implements IInputReceiver, IUpdatable
{
	public updateOrder: number = 4;

	public world: World;
	public camera: THREE.Camera;
	public target: THREE.Vector3;
	public sensitivity: THREE.Vector2;
	public radius: number = 0.005;
	public theta: number;
	public theta_offset: number;
	public phi: number;
	public onMouseDownPosition: THREE.Vector2;
	public onMouseDownTheta: any;
	public onMouseDownPhi: any;
	public targetRadius: number = 0.005;

	public movementSpeed: number;
	public actions: { [action: string]: KeyBinding };

	public upVelocity: number = 0;
	public forwardVelocity: number = 0;
	public rightVelocity: number = 0;

	public followMode: boolean = false;

	public characterCaller: Character;
	public range_ctrler=[];

	constructor(world: World, camera: THREE.Camera, sensitivityX: number = 1, sensitivityY: number = sensitivityX * 0.8)
	{
		this.world = world;
		this.camera = camera;
		this.target = new THREE.Vector3();
		this.sensitivity = new THREE.Vector2(sensitivityX, sensitivityY);

		this.movementSpeed = 0.06;
		this.radius = 1.15;
		this.theta = 0;
		this.theta_offset = 160;
		this.phi = 0;

		this.onMouseDownPosition = new THREE.Vector2();
		this.onMouseDownTheta = this.theta;
		this.onMouseDownPhi = this.phi;

		this.actions = {
			'forward': new KeyBinding('KeyW'),
			'back': new KeyBinding('KeyS'),
			'left': new KeyBinding('KeyA'),
			'right': new KeyBinding('KeyD'),
			'up': new KeyBinding('KeyE'),
			'down': new KeyBinding('KeyQ'),
			'fast': new KeyBinding('ShiftLeft'),
		};

		world.registerUpdatable(this);
	}


	public limit_theta(theta: number): number
	{
		var min_dist = 1e10;
		var dist;
		var min_idx=0;
		if( this.range_ctrler.length == 0){
			return theta;
		} 
		var in_region:boolean = false;
		const cx:number = this.camera.position.x;
		const cz:number = this.camera.position.z;
		const point = [cx, cz];
		for(var idx=0; idx <= this.range_ctrler.length;idx++){
			var range_with_theta = this.range_ctrler[idx]
			var offset:number;
			var range, central_theta:number;
			
			[range, central_theta, offset] = range_with_theta;
			in_region = in_region || pointInRectangle(point, range);
			if (in_region){
				// in rect here
				var real_offset = (theta - central_theta) % 360;
				if (real_offset > 180){
					real_offset -= 360;
				}
				// return theta;
				var rs:number;
				if (real_offset > (-1*offset) && real_offset < offset){
					rs = theta;
				} 
				else if(real_offset <= -offset){
					rs =  central_theta - offset + 0.1;
				}
				else{
					rs = central_theta + offset - 0.1;
				}
				// console.log(theta, central_theta, offset, real_offset, rs, cx, cz, this.range_ctrler, in_region);
				return rs;
			}
			
			dist = distance(point, range);
			if (dist > min_dist){
				min_dist = dist;
				min_idx = idx;
			}
		}

		// return theta;
		// console.log("nowhere", theta, cx, cz, this.range_ctrler, in_region);
		var range_with_theta = this.range_ctrler[min_idx];
		var range, central_theta:number, offset:number;
		[range, central_theta, offset] = range_with_theta;
		
		// in rect here
		var real_offset = (theta - central_theta) % 360;
		if (real_offset > 180){
			real_offset -= 360;
		}
		if (real_offset > -offset && real_offset < offset){
			return theta;
		} 
		else if(real_offset <= -offset){
			return central_theta - offset + 0.1;
		}
		else{
			return central_theta + offset - 0.1;
		}

	}

	public setSensitivity(sensitivityX: number, sensitivityY: number = sensitivityX): void
	{
		this.sensitivity = new THREE.Vector2(sensitivityX, sensitivityY);
	}

	public setRadius(value: number, instantly: boolean = false): void
	{
		this.targetRadius = Math.max(0.001, value);
		if (instantly === true)
		{
			this.radius = 1.15;
			// this.radius = value;
		}
		// console.log("this.radius: ", this.radius);
	}

	public move(deltaX: number, deltaY: number): void
	{
		this.theta = this.limit_theta(this.theta - deltaX * (this.sensitivity.x / 2));
		this.theta %= 360;
		// this.phi += deltaY * (this.sensitivity.y / 2);
		// this.phi = Math.min(85, Math.max(-85, this.phi));
	}

	public update(timeScale: number): void
	{
		// console.log("this.radius: ", this.radius);
		this.radius = 1.15;
		var lowy = 0.5;
		if (this.followMode === true)
		{
			this.camera.position.y = THREE.MathUtils.clamp(this.camera.position.y, this.target.y, Number.POSITIVE_INFINITY);
			this.camera.lookAt(this.target);
			let newPos = this.target.clone().add(new THREE.Vector3().subVectors(this.camera.position, this.target).normalize().multiplyScalar(this.targetRadius));
			this.camera.position.x = newPos.x;
			this.camera.position.y = newPos.y;
			this.camera.position.z = newPos.z;

			this.camera.up = new THREE.Vector3(0, 1, 0);
		}
		else 
		{
			// this.radius = THREE.MathUtils.lerp(this.radius, this.targetRadius, 0.005);
	
			this.camera.position.x = this.target.x + this.radius * Math.sin((this.theta + this.theta_offset) * Math.PI / 180) * Math.cos(this.phi * Math.PI / 180);
			this.camera.position.y = this.target.y + this.radius * Math.sin(this.phi * Math.PI / 180);
			this.camera.position.z = this.target.z + this.radius * Math.cos((this.theta + this.theta_offset) * Math.PI / 180) * Math.cos(this.phi * Math.PI / 180);

			this.camera.up = new THREE.Vector3(0, 1, 0);
			this.camera.updateMatrix();
			this.camera.lookAt(this.target);
		}
	}

	public handleKeyboardEvent(event: KeyboardEvent, code: string, pressed: boolean): void
	{
		// Free camera
		if (code === 'KeyC' && pressed === true && event.shiftKey === true)
		{
			if (this.characterCaller !== undefined)
			{
				this.world.inputManager.setInputReceiver(this.characterCaller);
				this.characterCaller = undefined;
			}
		}
		else
		{
			for (const action in this.actions) {
				if (this.actions.hasOwnProperty(action)) {
					const binding = this.actions[action];
	
					if (_.includes(binding.eventCodes, code))
					{
						binding.isPressed = pressed;
					}
				}
			}
		}
	}

	public handleMouseWheel(event: WheelEvent, value: number): void
	{
		// this.world.scrollTheTimeScale(value);
	}

	public handleMouseButton(event: MouseEvent, code: string, pressed: boolean): void
	{
		for (const action in this.actions) {
			if (this.actions.hasOwnProperty(action)) {
				const binding = this.actions[action];

				if (_.includes(binding.eventCodes, code))
				{
					binding.isPressed = pressed;
				}
			}
		}
	}

	public handleMouseMove(event: MouseEvent, deltaX: number, deltaY: number): void
	{
		this.move(deltaX, deltaY);
	}

	public inputReceiverInit(): void
	{
		this.target.copy(this.camera.position);
		this.setRadius(0, true);
		// this.world.dirLight.target = this.world.camera;

		this.world.updateControls([
			{
				keys: ['W', 'S', 'A', 'D'],
				desc: 'Move around'
			},
			{
				keys: ['E', 'Q'],
				desc: 'Move up / down'
			},
			{
				keys: ['Shift'],
				desc: 'Speed up'
			},
			{
				keys: ['Shift', '+', 'C'],
				desc: 'Exit free camera mode'
			},
		]);
	}

	public inputReceiverUpdate(timeStep: number): void
	{
		// Set fly speed
		let speed = this.movementSpeed * (this.actions.fast.isPressed ? timeStep * 600 : timeStep * 60);

		const up = Utils.getUp(this.camera);
		const right = Utils.getRight(this.camera);
		const forward = Utils.getBack(this.camera);

		this.upVelocity = THREE.MathUtils.lerp(this.upVelocity, +this.actions.up.isPressed - +this.actions.down.isPressed, 0.3);
		this.forwardVelocity = THREE.MathUtils.lerp(this.forwardVelocity, +this.actions.forward.isPressed - +this.actions.back.isPressed, 0.3);
		this.rightVelocity = THREE.MathUtils.lerp(this.rightVelocity, +this.actions.right.isPressed - +this.actions.left.isPressed, 0.3);

		this.target.add(up.multiplyScalar(speed * this.upVelocity));
		this.target.add(forward.multiplyScalar(speed * this.forwardVelocity));
		this.target.add(right.multiplyScalar(speed * this.rightVelocity));
	}
}