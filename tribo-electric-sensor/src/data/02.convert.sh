
CURRENT_TIMESTAMP=$(date '+%Y%m%d%H%M%S')

mkdir -p /DATA/BANDS

for i in /DATA/EM/*.flac
do
	b=$(basename $i)
	f=$(echo $b | sed -e 's/.flac//' | sed -e 's/chunk_/bands_/')
	echo -n $f:
	if [ -f /DATA/BANDS/$f.npy ]
	then
		echo " /DATA/BANDS/$f.npy already exists"
	        ls /DATA/BANDS/$f.npy 
	else
                python3 02.flac_to_bands.py --out /DATA/BANDS $i
	fi
done






