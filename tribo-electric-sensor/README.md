# tribo-electric-sensor
Tribo electric Sensor

## MECHANISM

- We have 200 meters long (68 round) coil, around 85x60cm frame.
- We use emaye covered 0.6mm copper wire for that, we covered back and the rigth and the left of the frame with aluminium for electro magnetic isolation of local events.
- We wired the coil to the sound card (48khz), so we can get the radio waves between 20hz-20khz.
- We read 5 seconds of data from sound card continously, we record it 5 seconds length.
  - We convert 5 second sound record to 5 second sound x 20 bands matrix.
- We read one record from adsb device every 5 seconds. while reading the adsb, be careful to give a rectangle that your coil faces.
- we will feed the 5 second x 20 band matrix to the neural net, we will expect to get the corresponding adsb record.
 
## DETECTION

A commercial plane typically cruises at about 0.22 to 0.27 kilometers per second (km/s), or roughly 0.25 km/s on average.

Speed Breakdown

• Typical cruise (800–965 km/h): ~0.22–0.27 km/s
• Long-haul average (880–926 km/h): ~0.24–0.26 km/s (per Epic Flight Academy)
• Takeoff & landing (220–300 km/h): ~0.06–0.08 km/s (per World Aviation ATO)

How It Converts

Divide speed in kilometers per hour (km/h) by 3,600 (the number of seconds in an hour) to get kilometers per second.

Also think about 20 mach speed : 7 kiometers per second it travels.


SO :

  - If maximum 35km/second resolution is enough :
     - READ one request per 5 seconds from adsb : in 5 second, standard plane can travel only 1.25km , and a rocket may travel 35km.
     - READ 5 second wav file every 5 seconds : you will have 5-10 number of 20khz waves within 5 seconds. Also you will have 10-20 !0khz waves etc.



