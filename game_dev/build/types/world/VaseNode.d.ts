import { Vase } from './Vase';
import { Object3D } from 'three';
export declare class VaseNode {
    object: Object3D;
    vase: Vase;
    constructor(child: THREE.Object3D, vase: Vase);
}
