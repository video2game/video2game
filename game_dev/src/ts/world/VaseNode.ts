import { Vase } from './Vase';
import { Object3D } from 'three';

export class VaseNode
{
	public object: Object3D;
	public vase: Vase;

	constructor(child: THREE.Object3D, vase: Vase)
	{
		this.object = child;
		this.vase = vase;
	}
}