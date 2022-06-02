trap "kill 0" EXIT

for i in $(seq "$1"); do
	n=$(($i-1))
	ind=$(($n%4))
	python network.py -m linear -n "model_$i" --verbose -a $2 -d "cuda:$ind" &
done

wait
