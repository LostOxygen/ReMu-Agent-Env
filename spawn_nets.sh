#!/bin/bash
trap "kill 0" EXIT

help()
{
    echo "Usage: spawn_nets [ -x | --num_models ]"
	echo "					[ -m | --model_type ]"
	echo "					[ -a | --addr 		]"
	echo "					[ -d | --device		]"
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
			echo "Unexpected argument: $1"
			help
			;;
	esac
	shift
done

printf "\n############################################################################"
printf "\nSpawning $NUM_MODELS $MODEL_TYPE networks on device $DEVICE and addr $ADDR"
printf "\n############################################################################"

for i in $(seq "$NUM_MODELS"); do
	if [ "$DEVICE" = "cpu" ]; then
		python network.py -m "$MODEL_TYPE" -n "model_$((i-1))" --verbose -a "$ADDR" -d "cpu" &
	else
		n=$(($i-1))
		ind=$(($n%4))
		python network.py -m "$MODEL_TYPE" -n "model_$((i-1))" --verbose -a "$ADDR" -d "cuda:$ind" &
	fi
done

wait
