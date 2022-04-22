# API

## server to client

Server transmits the game state to each client containing the players score and information about all players and projectiles.

### Objects

#### GameState

```json
{
	"players": [Player],
	"projectiles": [Projectile]
}
```

#### Player
Representation of a player.

```json
{
	"name": string,
	"score": int,
	"position": Coordinate,
	"direction": Coordinate
}
```

#### Projectile
Representation of a projectile

```json
{
	"owner": string,
	"position": Coordinate,
	"direction": Coordinate
}
```

#### Coordinate
Representation of a 2D-Vector

```json
{
	"x": float,
	"y": float
}
```

### Example

```json
{
	"players": [
		{
			"name": "gandalf",
			"score": 10000,
			"position": { "x": 20, "y": 100 },
			"direction": { "x": 1, "y": 10 }
		},
		{
			"name": "dumbledore",
			"score": -100,
			"position": { "x": 100, "y": 200 },
			"direction": { "x": 0, "y": 50 }
		}
	],
	"projectiles": [
		{
			"owner": "gandalf",
			"position": { "x": 24, "y": 155 },
			"direction": { "x": -10, "y": -50 }
		},
		{
			"owner": "dumbledore",
			"position": { "x": 11, "y": 155 },
			"direction": { "x": 100, "y": -10 }
		}
	]
}
```

## client to server

### Description

Response to the server after receiving the current game state.
For the initial message to server `actions` is a empty list.

Values of `actions` may be: "left", "right", "forward", "backward" or "shoot".

### Objects

#### Action

```json
{
	"name": string,
	"actions": [string]
}
```

### Example

```json
{
	"name": "gandalf",
	"actions": [ "forward", "shoot" ]
}
```
