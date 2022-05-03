# API

## server to client

Server transmits the game state to each client containing the players score and information about all players and projectiles.

### Objects

#### GameState

```json
{
	"players": [Player],
	"projectiles": [Projectile],
	"scoreboard": [Score]
}
```

#### Player
Representation of a player.

```json
{
	"name": string,
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

#### Score
Representation of a score entry

```json
{
	"name": string,
	"value": int
}
```

### Example

```json
{
	"players": [
		{
			"name": "gandalf",
			"position": { "x": 20, "y": 100 },
			"direction": { "x": 1, "y": 10 }
		},
		{
			"name": "dumbledore",
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
	],
	"scoreboard": [
		{
			"name": "gandalf",
			"value": 100
		},
		{
			"name": "dumbledore",
			"value": -50
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
