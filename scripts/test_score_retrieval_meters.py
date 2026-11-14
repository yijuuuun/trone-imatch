import math
import unittest
from score_retrieval_meters import fit_axis, scales
from geographiclib.geodesic import Geodesic

class CoordinateTests(unittest.TestCase):
    def test_restore_known_grid(self):
        mx,my=scales(30.)
        rows=[]
        for i,(x,y) in enumerate([(-.7,.2),(.1,-.6),(.8,.3),(-.2,.9)]):
            rows.append(dict(lon=115.+(64*i+32*x)/mx,lat=30.-(64*(i+1)+32*y)/my,x_norm_raw=x,y_norm_raw=y))
        fx,rx=fit_axis(rows,'lon','x_norm_raw',1,mx*1.00001)
        fy,ry=fit_axis(rows,'lat','y_norm_raw',-1,my*.99999)
        self.assertAlmostEqual(fx,mx,places=4);self.assertAlmostEqual(fy,my,places=4)
        for i,r in enumerate(rows):
            lon=r['lon']-32*r['x_norm_raw']/fx;lat=r['lat']+32*r['y_norm_raw']/fy
            error=Geodesic.WGS84.Inverse(lat,lon,30.-64*(i+1)/my,115.+64*i/mx)['s12']
            self.assertLess(error,1e-6)
    def test_bad_offsets_rejected(self):
        mx,_=scales(30.)
        rows=[dict(lon=115.+64*i/mx,x_norm_raw=0) for i in range(5)]
        rows[2]['x_norm_raw']=.2
        with self.assertRaises(AssertionError):fit_axis(rows,'lon','x_norm_raw',1,mx)
    def test_known_wgs84_distance(self):
        self.assertAlmostEqual(Geodesic.WGS84.Inverse(0,0,0,1)['s12'],111319.490793,places=5)

if __name__=='__main__':unittest.main()