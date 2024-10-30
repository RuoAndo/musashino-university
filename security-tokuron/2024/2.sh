while read line; do
    #echo $line
    python dbscan13.py $line
done < list
