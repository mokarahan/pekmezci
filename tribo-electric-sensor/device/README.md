# NASIL YAPILIR

## CERCEVE'nin OLUSTURULMASI

Sonucta  85x60 cm'lik bir köpügün kenarlarina 2.5cm kalinligindeki kablo yolunu kenarlarini yuvarlatarak ( 3 parcaya bolup kesikli yuvarlattim) yapistiracagiz

### Kablo yolunun kesilmesi:

Bir uctan basladim, 10cm olctum sonra kesmeye basladim ( 10 cmlik kisin diger kenardaki duz kisma denk geldii ki diger parca ile uc uca getirmesi kolay olsun)

1. 10 cm den sonra 1.4 cm bir tarafi kestim (dis cevreyi kesmedim , ic cevre, ve alt bantin oldugu yeri kestim, sonra 4.7 cm biraktim, sonra 1.4 cm daha kestim, sonra 4.7 cm biraktim, sonra 1.5 cm daha kestim.
   Sonucta 3 tane 1.4 cm'lik kesik arasinda 2 tane 4.7 cmlik oluklar kaldi.

   Toplamda 13.4 cm gibi bir seyi kestim.

  
2. Bunlari bükerek köselere yerlestirdim. 
   Yerlestirdigimde duz kenarlar , köpügün kösesinden 9-10 cm kadar icerde kaldi.
   Uzun kenarda hic keilmemis duz kenar 65cm

3. Bir kablo yolu ile 3 kose dondum.
   Diger koseyi de 65cm'lik bir kablo yolu ile dondum.
   Baslandgic ve bitis noktasi arasinda 2cm kadar bosluk biraktim.

4. Cikan bakir uclarin uclarini kazidim, Connector'e Left ve Ground olarak vidaladim. Righta takmadim.

5. ayni sekilde midex m1 microphone'un basini kestimn ve onu da L ve Ground'a bagladim

6. Araya Ground loop isolator taktim.

7. ikrofonun usb'sini bilgisayara takip, bu usb cihazini nput olarak sectim.

8. Audacity ile baktigimda bir sinyal gorunuyor.







## 1 saniye icinde 10khz lik dalganin bilgisayarda algilanmasi

isik hizi 300.000km/saniye.

10khz frekansli radyo dalgasi 30.000 km uzunluktadir, yani iki tepe noktasi arasi 30.000km 

ancak bu coil ile algilanmaya calisilirken problem olmuyor,

zira,    -------O--------      ortadaki coil olsun, duz cizgiler tam radyo dalgasi olsun, 
        |    30 km      |


coil in icinden gecen dalga, coilde magnetic field degisimine neden oluyor,

ve saniyede 10 bin kez (10khz) bu coilin icinden bir tepe noktasi geciyor ( bu tepe noktasinin genligi farkli olabilir, ancak ayni kaynaktan geliyorsa asagi yukari ayni da olabilir)

ancak aradaki degerleri de guzel olcebilmek icin aslinda 48 khz olan sampling rate'ten dolayi sadece 4.8 deger olcmus oluyoruz. (yani 4 deger :  iki tepe noktasi arasi 4 olcum yapmis oluyoruz)

Nyquist'e gore 10khz i alabilmek icin en az 20 khz sampling rate olmali
               20 khz'i alabilmek için  en az 40 khz olmali.

40 khz'in biraz uzerinde olsa iyi olur.


