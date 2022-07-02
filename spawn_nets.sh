#!/bin/bash
trap "kill 0" EXIT

help()
{
    printf "Usage: spawn_nets [ -x | --num_models ] [ -m | --model_type ] [ -a | --addr ] [ -d | --device ] [-p | --parameter_search]"
    exit 2
}

while [[ "$#" -gt 0 ]];
do
	case "$1" in
		-x | --num_models )
			NUM_MODELS="$2"
			shift
			;;
		-m | --model_type )
			MODEL_TYPE="$2"
			shift
			;;
		-p | --parameter_search )
			PARAM_SEARCH=true
			;;
		-a | --addr )
			ADDR="$2"
			shift
			;;
		-d | --device )
			DEVICE="$2"
			shift
			;;
		-* | --* )
			shift
			break
			;;
		* )
			printf "Unexpected argument: $1"
			help
			;;
	esac
	shift
done

{
	LOADED_GPUS="$(ls /proc/driver/nvidia/gpus/ | wc -l)";
} || {
	LOADED_GPUS=0;
}

if [[ "$LOADED_GPUS" -eq 0 ]]; then
	DEVICE="cpu"
fi

printf "\n##########################################################"
printf "\n## Found $LOADED_GPUS GPU(s) on this machine"
printf "\n## Spawning $NUM_MODELS $MODEL_TYPE networks on device $DEVICE and addr $ADDR"
printf "\n##########################################################\n"

for i in $(seq "$NUM_MODELS"); do
	if [[ "$MODEL_TYPE" = "heuristic" ]]; then
		name="heuristic_$((i-1))"
	else
		name="model_$((i-1))"
	fi

	if [[ "$DEVICE" = "cpu" ]]; then
		if $PARAM_SEARCH; then
			python network.py -m "$MODEL_TYPE" -n $name --verbose -a "$ADDR" -d "cpu" -ps &
		else
			python network.py -m "$MODEL_TYPE" -n $name --verbose -a "$ADDR" -d "cpu" &
		fi
	else
		n=$(($i-1))
		ind=$(($n % $LOADED_GPUS))

		if $PARAM_SEARCH; then
			python network.py -m "$MODEL_TYPE" -n $name --verbose -a "$ADDR" -d "cuda:$ind" -ps &
		else
			python network.py -m "$MODEL_TYPE" -n $name --verbose -a "$ADDR" -d "cuda:$ind" &
		fi
	fi
done

wait
