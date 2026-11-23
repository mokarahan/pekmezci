#!/bin/bash


if [ "$1" == "" ]
then
	echo "Usage : $0 <band_file>"
	echo "Example : $0 /DATA/BANDS/bands_20261122T121005.545.npy"
	exit 1
fi

if [ ! -f $1 ]
then
	echo $1 does not exists
	exit 1
fi

#pip3 install torch numpy pandas scipy
#python3 train_bands_adsb.py info
#/DATA/BANDS/bands_20261122T121005.545.npy
#
python3 train_bands_adsb.py predict --ckpt runs/best.pt --input $1


