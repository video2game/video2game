import * as THREE from 'three';
import { CharacterStateBase } from '../_stateLibrary';
import { Character } from '../../Character';
import { VehicleSeat } from '../../../vehicles/VehicleSeat';
export declare class OpenVehicleDoor extends CharacterStateBase {
    private seat;
    private entryPoint;
    private hasOpenedDoor;
    private startPosition;
    private endPosition;
    private startRotation;
    private endRotation;
    private factorSimluator;
    constructor(character: Character, seat: VehicleSeat, entryPoint: THREE.Object3D);
    update(timeStep: number): void;
}
