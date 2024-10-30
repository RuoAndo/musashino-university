while read line; do
    #echo $line
    No=`echo $line | cut -d "_" -f 2 | cut -d "." -f 1`
    #echo $No

    #cat $line
    
    while read line2; do
    	echo $line2","$No | grep -v class
    done < $line
done < $1
