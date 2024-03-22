import { Coins } from './Coins';
import { Object3D } from 'three';

export class CoinNode
{
	public object: Object3D;
	public coins: Coins;

	constructor(child: THREE.Object3D, coins: Coins)
	{
		this.object = child;
		this.coins = coins;
	}
}