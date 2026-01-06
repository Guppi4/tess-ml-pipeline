from astroquery.mast import Catalogs
from astropy.coordinates import SkyCoord
import astropy.units as u

# Твои координаты: 00 01 37.67 -09 15 43.4
coord = SkyCoord("00h01m37.67s -09d15m43.4s", frame='icrs')

# Или в градусах:
# RA = (0 + 1/60 + 37.67/3600) * 15 = 0.4069583 deg
# Dec = -(9 + 15/60 + 43.4/3600) = -9.2620556 deg

# Запрос TIC в радиусе 10 arcsec
result = Catalogs.query_region(
    coord, 
    radius=10 * u.arcsec, 
    catalog="TIC"
)

# Показать результаты
if len(result) > 0:
    for row in result:
        print(f"TIC {row['ID']}")
        print(f"  RA: {row['ra']:.6f}, Dec: {row['dec']:.6f}")
        print(f"  Tmag: {row['Tmag']:.2f}")
        print(f"  Distance: {row['dstArcSec']:.2f} arcsec")
        print()
else:
    print("Ничего не найдено")