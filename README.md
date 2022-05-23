# ReMu-Agent-Env 3000
### Developing an Environment for Neural Network based Reinforcement Multi-Agent Interaction
#

## Server Usage:

```python
python server.py [-h] [--addr | -a ADRESS] [--port | -p PORT] [--verbose | -v]
```
example usage:
```python
python server.py --addr "192.168.2.420" --port 1337 --verbose
```

### Server Arguments
| Argument | Type | Description|
|----------|------|------------|
| -h, --help | None| shows argument help message |
| -a, --addr | STR | specifies the address of the server (default=localhost) |
| -p, --port | INT | specifies the port of the server (default=1337) |
| --training_mode | BOOL | sets the server to training mode which updates once all clients provide their action |
| -v, --verbose | BOOL | flag to set the server to verbose mode |
#
## Client Usage:

```python
python client.py [-h] [--name | -n NAME] [--addr | -a ADRESS] [--port | -p PORT] [--verbose | -v] |--spectate]
```
example usage:
```python
python client.py --name "Dieter" --addr "192.168.2.420" --port 1337 --verbose
```

### Client Arguments
| Argument | Type | Description|
|----------|------|------------|
| -h, --help | None| shows argument help message |
| -a, --addr | STR | specifies the address of the server on which the client tries to  connect(default=localhost) |
| -p, --port | INT | specifies the port of the server on which the client tries to connect (default=1337) |
| -v, --verbose | BOOL | flag to set the client to verbose (logging) mode |
| -n, --name | STR | specifies the name of the player (it's ID) |
| --spectate | BOOL | starts the client in spectator mode without a starship |
#
## Neural Network Usage:
```python
python network.py [-h] [--name | -n NAME] [--n_models | -nm N_MODELS] [--addr | -a ADRESS] [--port | -p PORT] [--verbose | -v]
```
example usage:
```python
python network.py --name "model_0" --addr "192.168.2.420" --port 1337 --verbose
python network.py --name "model" --n_models 20 --verbose
```

### Network Arguments
| Argument | Type | Description|
|----------|------|------------|
| -h, --help | None| shows argument help message |
| -a, --addr | STR | specifies the address of the server on which the model tries to  connect(default=localhost) |
| -p, --port | INT | specifies the port of the server on which the model tries to connect (default=1337) |
| -v, --verbose | BOOL | flag to set the model to verbose (logging) mode |
| -n, --name | LIST[ STR ] | specifies the name of the model. Allows multiple strings to spawn and train multiple models with the chosen names simultaneously |
| -nm, --n_models | INT | trains N models simultaneously. They will be named with --name as a prefix and _N as a suffix |