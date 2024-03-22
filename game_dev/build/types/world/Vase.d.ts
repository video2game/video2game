import { VaseNode } from './VaseNode';
export declare class Vase {
    nodes: {
        [nodeName: string]: VaseNode;
    };
    private rootNode;
    constructor(root: THREE.Object3D);
    addNode(child: any): void;
    listcoins(): any;
}
