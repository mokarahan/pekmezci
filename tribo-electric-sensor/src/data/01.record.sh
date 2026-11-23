
CURRENT_TIMESTAMP=$(date '+%Y%m%d%H%M%S')

mkdir -p /DATA/EM
python3 01.mic_band_recorder.py --out /DATA/EM >& /LOG/EM.$CURRENT_TIMESTAMP.log &


# BINA : 40.86 - 29.25
# 41 29.70,  41 30

export ADSBLOL_API_KEY=sk-lol-_JfaOmzatC8U1f7usJeUgi51J7tPk2_Ahmxu4u-sdF4
mkdir -p /DATA/ADSB
#python3 01.adsb_lol_recorder.py --lat-min 40.8 --lat-max 41.2 --lon-min 29.2 --lon-max 29.8  --out /DATA/ADSB >& /LOG/ADSB.$CURRENT_TIMESTAMP.log &
python3 01.adsb_lol_recorder.py --lat-min 40.8 --lat-max 41.2 --lon-min 29.2 --lon-max 29.8  --out /DATA/ADSB --poll 122 >& /LOG/ADSB.$CURRENT_TIMESTAMP.log &

