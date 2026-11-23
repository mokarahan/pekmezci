cd /DATA/ADSB

#for i in $(cat $(grep seen *| cut -d: -f1)| cut -d, -f7| sort|uniq | grep -v flight); do    echo -n $i:;grep $i * | wc -l ;    read; done
for i in $(cat $(grep seen *| cut -d: -f1)| cut -d, -f7| sort|uniq | grep -v flight); do    echo -n $i:;grep $i * | wc -l ;  done
