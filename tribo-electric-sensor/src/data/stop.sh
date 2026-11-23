kill -9 $(ps -ef | grep '01\.' |  grep /DATA/EM| head -1| awk '{print $2}')
kill -9 $(ps -ef | grep '01\.' | grep /DATA/ADSB| head -1| awk '{print $2}')

