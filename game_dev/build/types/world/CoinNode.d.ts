import { Coins } from './Coins';
import { Object3D } from 'three';
export declare class CoinNode {
    object: Object3D;
    coins: Coins;
    constructor(child: THREE.Object3D, coins: Coins);
}
