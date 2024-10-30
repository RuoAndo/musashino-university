./do.sh $1 > tmp-$1
cat tmp-$1 | sort -t, -k3 > tmp2-$1
python csv-plot.py tmp2-$1
