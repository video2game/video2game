import * as CANNON from 'cannon';
import * as THREE from 'three';
import * as Utils from '../../core/FunctionLibrary';
import { ICollider } from '../../interfaces/ICollider';

export class CapsuleCollider implements ICollider
{
	public options: any;
	public body: CANNON.Body;
	// public visual: THREE.Mesh;

	constructor(options: any)
	{
		let defaults = {
			mass: 0,
			position: new CANNON.Vec3(),
			height: 0.5,
			radius: 0.3,
			segments: 8,
			friction: 0.3,
			base: 0,
		};
		options = Utils.setDefaults(options, defaults);
		this.options = options;

		let mat = new CANNON.Material('capsuleMat');
		mat.friction = options.friction;

		let capsuleBody = new CANNON.Body({
			mass: options.mass,
			position: options.position
		});

		// Compound shape
		let sphereShape = new CANNON.Sphere(options.radius);
		// let boxShape = new CANNON.Box(new CANNON.Vec3(0.08, 0.2, 0.08));

		// Materials
		capsuleBody.material = mat;
		// sphereShape.material = mat;

		capsuleBody.addShape(sphereShape, new CANNON.Vec3(0, options.base+0, 0));
		capsuleBody.addShape(sphereShape, new CANNON.Vec3(0, options.base+options.height / 2, 0));
		capsuleBody.addShape(sphereShape, new CANNON.Vec3(0, options.base+-options.height / 2, 0));
		// capsuleBody.addShape(boxShape, new CANNON.Vec3(0, options.base, 0));

		this.body = capsuleBody;
	}
}