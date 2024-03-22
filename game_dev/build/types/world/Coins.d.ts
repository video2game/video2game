import { CoinNode } from './CoinNode';
export declare class Coins {
    nodes: {
        [nodeName: string]: CoinNode;
    };
    private rootNode;
    constructor(root: THREE.Object3D);
    addNode(child: any): void;
    listcoins(): any;
}
