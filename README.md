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
python network.py --name "model_0" --addr "192.168.2.420" --port 1337 --device "cuda:1" --verbose
```

### Network Arguments
| Argument | Type | Description|
|----------|------|------------|
| -h, --help | None| shows argument help message |
| -a, --addr | STR | specifies the address of the server on which the model tries to  connect(default=localhost) |
| -p, --port | INT | specifies the port of the server on which the model tries to connect (default=1337) |
| -v, --verbose | BOOL | flag to set the model to verbose (logging) mode |
| -n, --name | STR | specifies the name of the model. |
| -m, --model_type | STR | type of the model (e.g. "linear" or "lstm") |
| --test | BOOL | Sets network to testing modus |
| -d, --device | STR | specifies the device on which the model should be trained (e.g. "cpu" or "cuda:x", default="cuda:0"). Can be used to also specify the specific GPU (e.g. cuda:2)|

## Spawn multiple networks simultaneously via bash-script:
where **--num_models** flag defines the number of models to spawn. The networks will train on all available (loaded) nvidia GPUs.

```bash
chmod +x spawn_networks.sh
./spawn_nets.sh --num_models [N] --model_type [linear | lstm | cnn] --addr [IP] --device [cpu | cuda:x]
```