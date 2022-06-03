#!/bin/bash
trap "kill 0" EXIT

help()
{
    echo "Usage: spawn_nets [ -x | --num_models   ]
							[ -m | --model_type ]
							[ -a | --addr 		]
							[ -d | --device		]"
    exit 2
}

SHORT=x:,m:,n:,v:,a:,d:
LONG=num_models,model_type:,name:,verbose:,addr:,device:
OPTS=$(getopt -a -n spawn_nets --options $SHORT --longoptions $LONG -- "$@")

VALID_ARGUMENTS=$# # Returns the count of arguments that are in short or long options

if [ "$VALID_ARGUMENTS" -eq 0 ]; then
	echo "Not enough valid arguments passed"
	help
fi

eval set -- "$OPTS"
NUM_MODELS=1

while [[ $# -gt 0 ]]; do
	case "$1" in
	-x | --num_models )
		NUM_MODELS="$2"
		shift 2
		break
		;;
	-m | --model_type )
		MODEL_TYPE="$2"
		shift 2
		;;
		;;
	-a | --addr )
		ADDR="$2"
		shift 2
		;;
	-d | --device )
		DEVICE="$2"
		shift 2
		;;
	-- )
		shift
		break
		;;
	* )
		echo "Unexpected argument: $1"
		help
		;;
	esac
done

for i in $(seq "$NUM_MODELS"); do
	if ["$DEVICE" == "cpu"]; then
		python network.py -m "$MODEL_TYPE" -n "model_$i" --verbose -a $2 -d "cpu" &
	else
		n=$(($i-1))
		ind=$(($n%4))
		python network.py -m "$MODEL_TYPE" -n "model_$i" --verbose -a "$ADDR" -d "cuda:$ind" &
	fi
done

wait
