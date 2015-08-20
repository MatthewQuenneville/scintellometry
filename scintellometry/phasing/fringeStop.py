from __future__ import division

import numpy as np
from astropy.coordinates import Angle, SkyCoord, FK5
from astropy.time import Time, TimeDelta
import astropy.units as u
from astropy.utils.iers import IERS_A_URL, IERS_A
from astropy.utils.data import download_file
from astropy.table import Table
from astropy.constants import c as SPEED_OF_LIGHT

import os

class FringeStopper:
    def __init__(self,time_cal):
        # Source coordinates
        ra = Angle('05h34m31.9723236756655s')
        dec = Angle('22d00m52.0693143506095s')
        
        # Precess coordinates
        coords=SkyCoord(ra=ra, dec=dec, frame='icrs')
        coords=coords.transform_to(
            FK5(equinox=Time(time_cal, format='mjd')))
        
        self.ra=coords.ra
        self.dec=coords.dec

        # Observatory coordinates
        self.long = Angle(74.05*u.deg)
        self.lat = Angle(19.1*u.deg)
        self.elev = 588*u.m

        self.ntel = 30
        self.reference_dish = 'C02'
        self.band_center = (602+100./12)*u.MHz
        self.dtsample = 30*u.ns
        self.equiv = [(u.Unit(2*self.dtsample),u.byte)]

        self.time_cal=time_cal
        self.antenna_file = os.path.join(os.path.dirname(
                os.path.realpath(__file__)), 'antsys.hdr')
        self.delay_file_cal = os.path.join(os.path.dirname(
                os.path.realpath(__file__)), 'delays.dat')
        self.phase_file_cal = os.path.join(os.path.dirname(
                os.path.realpath(__file__)), 'phases.dat')

        self.delays = {k:int(v)*u.byte for k,v in 
                       self.load_feed_data(self.delay_file_cal).iteritems()}
        self.phases = {k:v*u.radian for k,v in
                       self.load_feed_data(self.phase_file_cal).iteritems()}

        self.antenna_order = 'CWES' 
        self.non_existing_antennas = ('C07', 'S05')  

        self.antenna_coords=self.get_antenna_coords()

        self.iers_a_file = download_file(IERS_A_URL, cache=True)
        self.iers_a = IERS_A.open(self.iers_a_file)

        self.geodelays_cal = self.get_geometric_delays(time_cal)

    def get_antenna_coords(self):
        """Read antenna coordinates from GMRT .hdr file.  First store them all
        in a dictionary, indexed by the antenna name, remove non-existing
        antennas, then get them in the order used in Ue-Li's phasing code,
        and finally make it a Table, which is easier to access than a
        dictionary.  Probably could be done more directly.
        """
        with open(self.antenna_file, 'r') as fh:
            antennas = {}
            line = fh.readline()
            while line != '':
                if line[:3] == 'ANT':
                    al = line.split()
                    antennas[al[2]] = np.array([float(item) 
                                                for item in al[3:8]])
                line = fh.readline()

        for bad in self.non_existing_antennas:
            antennas.pop(bad)

        antenna_names = self.order_antenna_names(antennas)
        # store all antenna's in a Table
        ant_tab = Table()
        ant_tab['ant'] = antenna_names
        ant_tab['xyz'] = [antennas[ant][:3] for ant in ant_tab['ant']]
        ant_tab['delay'] = [antennas[ant][3:] for ant in ant_tab['ant']]
        return ant_tab

    def order_antenna_names(self, antennas):
        """Get antenna in the correct order, grouped by C, W, E, S, and
        by number within each group.
        """
        names = list(antennas)
        order=self.antenna_order

        def cmp_names(x, y):
            value_x, value_y = [order.index(t[0])*100+int(t[1:]) for t in x, y]
            return -1 if value_x < value_y else 1 if value_x > value_y else 0

        names.sort(cmp_names)
        return names

    def get_uvw(self, ha):
        """Get delays in UVW directions between pairs of antenna's for
        given hour angle and declination of a source.
        """
        h = ha.to(u.rad).value
        d = self.dec.to(u.rad).value

        index = list(self.antenna_coords['ant']).index(self.reference_dish)
        dxyz = self.antenna_coords['xyz'][index] - self.antenna_coords['xyz']

        #  unit vectors in the U, V, W directions
        xyz_u = np.array([-np.sin(d)*np.cos(h), 
                           np.sin(d)*np.sin(h), 
                           np.cos(d)])
        xyz_v = np.array([np.sin(h), 
                          np.cos(h), 
                          0.])
        xyz_w = np.array([np.cos(d)*np.cos(h), 
                          -np.cos(d)*np.sin(h), 
                          np.sin(d)])
        return np.vstack([(xyz_u*dxyz).sum(1),
                          (xyz_v*dxyz).sum(1),
                          (xyz_w*dxyz).sum(1)]).T

    def get_parallactic_angle(self, ha):
        return np.arctan2(-np.cos(self.lat.radian) *
                           np.sin(ha.radian),
                           np.sin(self.lat.radian) *
                           np.cos(self.dec.radian) -
                           np.cos(self.lat.radian) *
                           np.sin(self.dec.radian) *
                           np.cos(ha.radian)) * u.rad

    def get_hour_angle(self,time):
        local_time=Time(time,location=(self.long, self.lat, self.elev))
        local_time.delta_ut1_utc = self.iers_a.ut1_utc(local_time)
        LST=local_time.sidereal_time('apparent')
        return LST-self.ra

    def get_geometric_delays(self, time):
        HA = self.get_hour_angle(time)

        # write out delays for all time stamps, looping over baselines
        uvw = self.get_uvw(HA)
        uvw_us = (uvw*u.m/SPEED_OF_LIGHT).to(u.us)
        return {ant:delay for ant,delay in 
                zip(self.antenna_coords['ant'],uvw_us[:,2])}

    def load_feed_data(self, datafile):
        with open(datafile,'r') as f:
            return {line.split()[0]:float(line.split()[1]) for line in f}

    def get_delay_and_phase(self, time, feeds):
        geodelays = self.get_geometric_delays(time)
        offsets={}
        phases={}
        for feed in feeds:
            feed=feed[:-1]
            delay=geodelays[feed]-self.geodelays_cal[feed]
            offsets[feed]=self.delays[feed] + \
                delay.to(u.byte,equivalencies=self.equiv)
            phases[feed]=2*np.pi*self.band_center*delay*u.radian
            print feed, phases[feed]
        return {k:(v, (phases[k]+self.phases[k]).to(u.radian)) 
                for k,v in offsets.iteritems()}
"""        
if __name__ == '__main__':
    TIME1=57139.6262222566
    TIME2=57139.6418560581
    pulsetimes=np.array([57139.6232994795, 57139.6254291956, 57139.6262054797, 57139.6262222566, 57139.6288855723, 57139.6331243979, 57139.6362793292, 57139.6418560581])#+0.25/3600/24
    #phasedict1={'C06R': -0.4291328755315007, 'C14R': 2.8274917823985604, 'C04R': 1.3528628732204906, 'E05R': 2.7489248651813263, 'W02R': -2.5591891655271057, 'C10R': 0.0032130679391433326, 'C09R': -0.26384625525029426, 'W05R': -1.0978377478183576, 'S02R': -0.8998566475256338, 'E03R': 0.08909132048591717, 'C13R': 1.0899436694403422, 'C05R': -1.934850382492848, 'E04R': -0.07919774955525627, 'S06R': -2.055818267552569, 'S04R': 2.2980337103075605, 'W01R': 0.25758431667848614, 'C11R': 1.7291378940931565, 'S01R': 0.2583638926751233, 'W06R': 0.0, 'S03R': -1.8085270811713663, 'C08R': 0.8621401652212093, 'E06R': -1.4072514197306276, 'E02R': 1.2713924653651842} #nearest sample
    phasedict1={'C06R': -0.4215668828327175, 'C14R': 2.826631071484469, 'C04R': 1.3114998598389347, 'E05R': 2.735721323234296, 'W02R': -2.4797079978138736, 'C10R': -0.0018494501149421971, 'C09R': -0.32617452740373576, 'W05R': -1.115380591751614, 'S02R': -1.019231670975418, 'E03R': 0.06568674631810101, 'C13R': 1.0532013777651592, 'C05R': -1.9859783310613466, 'E04R': -0.2181092404291749, 'S06R': -2.111569718364705, 'S04R': 2.314314914039631, 'W01R': 0.2506928490514566, 'C11R': 1.684467598479972, 'S01R': 0.29495072585543625, 'W06R': 0.0, 'S03R': -1.9036565963243923, 'C08R': 0.868066281166163, 'E06R': -1.483372143029733, 'E02R': 1.3076053351602903} #nearest sample quarter band

    #phasedict2_meas={'C06R': -2.5603388568882282, 'C14R': 1.7841253343244801, 'C04R': -2.3713542363860443, 'E05R': 1.951675921614322, 'W02R': -1.6117416558417521, 'C10R': -0.0, 'C09R': -2.5558401040511929, 'W05R': 1.6961066574282073, 'S02R': 1.7015211002854795, 'E03R': 1.5718669134574632, 'C13R': 0.55023184412445136, 'C05R': -0.14667213535054899, 'E04R': 1.0689635230258319, 'S06R': 2.1641716014018288, 'S04R': 2.6980113097322258, 'W01R': 0.25439096713364978, 'C11R': 2.8139595701599132, 'S01R': -0.89115151786582625, 'W06R': 2.0807320356522956, 'S03R': 0.045686291485786283, 'C08R': 1.8121442532266734, 'E06R': -1.1296673092864487, 'E02R': -1.7736295428950595}#15:00:36
    #phasedict2_meas={'C06R': -3.0201320646006886, 'C14R': -2.1640921527649555, 'C04R': -1.5848584997972672, 'E05R': -1.4085353776937679, 'W02R': -2.2337195355010713, 'C10R': -1.0635207563373235, 'C09R': -2.4377121145479852, 'W05R': -2.6740767012066091, 'S02R': -3.1093770894581443, 'E03R': -0.29727713271706291, 'C13R': -2.9673198203553421, 'C05R': -2.9940669643850875, 'E04R': -0.0, 'S06R': -1.1888636796320453, 'S04R': 3.0441253206773169, 'W01R': -1.8577169149126569, 'C11R': 0.15539324193509252, 'S01R': -2.97102286971954, 'W06R': 1.0057533495342892, 'S03R': 2.6990299518927015, 'C08R': -1.6565572778713802, 'E06R': 1.7872305667742869, 'E02R': -2.8682070389861436} #15:05
    #phasedict2_meas={'C06R': 1.5703567477885179, 'C14R': -1.5474754589490733, 'C04R': -2.6686451397123521, 'E05R': 1.9396198985708375, 'W02R': 1.3097440972828625, 'C10R': 1.7751786788836008, 'C09R': 1.6787362487380126, 'W05R': 1.1790205627346668, 'S02R': -0.46014158814987022, 'E03R': 0.92026426771539516, 'C13R': 2.9003364771239641, 'C05R': -0.0, 'E04R': -0.034473778436551368, 'S06R': 0.96182293303055755, 'S04R': 0.32492961876353282, 'W01R': -3.1102319238945091, 'C11R': -2.3752199656266844, 'S01R': 0.92650605039319511, 'W06R': -2.3542730510410688, 'S03R': -2.4809403404393153, 'C08R': 2.6135572038931274, 'E06R': -2.497388281420434, 'E02R': 2.4841030619442841} #15:01:43
    #phasedict2_meas={'C06R': 0.16157941205795989, 'C14R': -1.1287135179491745, 'C04R': -1.7808023010634098, 'E05R': 1.6928041037979489, 'W02R': 1.8644815941669444, 'C10R': -2.3701995135772589, 'C09R': -1.0146796494295005, 'W05R': 2.4501492322274925, 'S02R': -2.589012475122161, 'E03R': 2.9870744864272818, 'C13R': -1.6854755193244744, 'C05R': 2.5891002489373842, 'E04R': 2.5351861805131612, 'S06R': 1.8686196104336255, 'S04R': -2.2936996306067035, 'W01R': -1.0955118234838817, 'C11R': 1.5345275197145436, 'S01R': 2.2527382430886207, 'W06R': -0.0, 'S03R': -0.49194718933726905, 'C08R': -1.7902648731843758, 'E06R': -1.1014483910068524, 'E02R': -2.3970450116720028} #15:24
    #phasedict2_meas={'C06R': -2.1285305877943355, 'C14R': 2.3601499875263618, 'C04R': -0.092703649205452432, 'E05R': 0.76906845165280457, 'W02R': 2.3138281910316101, 'C10R': 0.034479495067012582, 'C09R': -2.2279751368174212, 'W05R': -2.3505799696742584, 'S02R': 0.93553689648803717, 'E03R': -3.1214768836709887, 'C13R': 0.4346887377685929, 'C05R': -0.0, 'E04R': -2.6015196422079256, 'S06R': 1.7036931236615074, 'S04R': 2.2068423326250248, 'W01R': -0.75988855292325386, 'C11R': -2.0693893128830072, 'S01R': -0.74365435192092222, 'W06R': 0.68783847664950426, 'S03R': -0.19295455168114839, 'C08R': 1.2831312776937067, 'E06R': 2.5201439046164706, 'E02R': 1.4739563092677839} #15:00:29 narrow band
    phasedict2_meas={'C06R': -2.5685917183801332, 'C14R': 1.7927241774141593, 'C04R': -2.3679755700459606, 'E05R': 1.922367949853093, 'W02R': -1.4274779919536649, 'C10R': -0.0, 'C09R': -2.6228338791115013, 'W05R': 1.6857105720221788, 'S02R': 1.6662860046470838, 'E03R': 1.5701885083542211, 'C13R': 0.49537858985629679, 'C05R': -0.19116868979607649, 'E04R': 0.70358213447366846, 'S06R': 2.1237767263319327, 'S04R': 2.6993584270010245, 'W01R': 0.23208126150267297, 'C11R': 2.7878558503163258, 'S01R': -0.89686088790185048, 'W06R': 2.0414735977341807, 'S03R': 0.099901544723575222, 'C08R': 1.8071464565570081, 'E06R': -1.1722377351859592, 'E02R': -1.7805791788585066} #15:00:36 narrow band
    #phasedict2_meas={'C06R': -2.1291104409323949, 'C14R': 2.3590621653383423, 'C04R': -0.093883584296778441, 'E05R': 0.77082854910646825, 'W02R': 2.3136320725921848, 'C10R': 0.033998821723043908, 'C09R': -2.2274584596487625, 'W05R': -2.3502275362050096, 'S02R': 0.93585612762127646, 'E03R': -3.1214580344976159, 'C13R': 0.43583266170988821, 'C05R': -0.0, 'E04R': -2.3990386853631391, 'S06R': 1.7043319144962008, 'S04R': 2.2065091218166355, 'W01R': -0.75757511591843285, 'C11R': -2.070085448510127, 'S01R': -0.74381903798166737, 'W06R': 0.68789597582269735, 'S03R': -0.19427800710101317, 'C08R': 1.2835829668226379, 'E06R': 2.5198226038345468, 'E02R': 1.4740034904639112} #15:00:29 wrapped, narrow band
    phasedict2_meas={k:v-phasedict2_meas['C04R'] for k,v in phasedict2_meas.iteritems()}
    phasedict2={}
    delayDict2={}
        
    ant_ref='C04R'

    geodelaydict1=getGeometricDelays(pulsetimes[3])
    #geodelaydict2=getGeometricDelays(57139.6253451507)
    geodelaydict2=getGeometricDelays(pulsetimes[1])
    dtau_frac_dict={}
    f0=(602+100./12)*u.MHz

    for ant in phasedict1.keys():

        tau_B_A_1_G=(geodelaydict1[ant[:-1]]-geodelaydict1[ant_ref[:-1]])*u.byte
        tau_B_A_2_G=(geodelaydict2[ant[:-1]]-geodelaydict2[ant_ref[:-1]])*u.byte
        dtau_B_A_G=tau_B_A_2_G-tau_B_A_1_G
        
        dtau_B_A_G_int=np.round(dtau_B_A_G.value)*u.byte
        dtau_B_A_G_frac=(dtau_B_A_G.value-dtau_B_A_G_int.value)*u.byte

        dphi_B_A_G_frac=Angle(2*np.pi*dtau_B_A_G_frac.to(u.s,equivalencies=[(u.Unit(60*u.ns),u.byte)])*f0*u.radian).to(u.radian)
        dphi_B_A_G=Angle(2*np.pi*dtau_B_A_G.to(u.s,equivalencies=[(u.Unit(60*u.ns),u.byte)])*f0*u.radian).to(u.radian)

        delayDict2[ant]=delayDict1[ant]+dtau_B_A_G_int.value
    
        phi_B_A_1=Angle((phasedict1[ant]-phasedict1[ant_ref])*u.radian)
        phi_B_A_2=(phi_B_A_1+dphi_B_A_G).wrap_at(np.pi*u.radian).value
        #print ant, dtau_B_A_G_int, phi_B_A_1.value, dphi_B_A_G_frac.wrap_at(np.pi*u.radian).value, phi_B_A_2, Angle(phasedict2_meas[ant]*u.radian).wrap_at(np.pi*u.radian).value
        phasedict2[ant]=phi_B_A_2
        dtau_frac_dict[ant]=dtau_B_A_G.to(u.s,equivalencies=[(u.Unit(60*u.ns),u.byte)])
        print ant, dtau_B_A_G.value, '\t', int(dtau_B_A_G_int.value), '\t', dtau_B_A_G_frac.to(u.ns,equivalencies=[(u.Unit(60*u.ns),u.byte)])
    print delayDict2

    print ""
    
    print phasedict2
    
    names=[i+'R' for i in order_antenna_names([i[:-1] for i in phasedict2.keys()])]

    pred=[phasedict2[i] for i in names]
    calc=[Angle(phasedict2_meas[i]*u.radian).wrap_at(np.pi*u.radian).value 
          for i in names]
    plt.figure()
    plt.plot(pred,'rx',markersize=10,markeredgewidth=2,label='Predicted')
    plt.plot(calc,'b+',markersize=10,markeredgewidth=2,label='Measured')
    plt.xlim(-1,len(names))
    plt.xticks(range(len(names)),names,rotation=90)
    plt.ylabel('Phase (radians)')
    plt.legend(loc='upper left')
    plt.show()

    plt.figure()
    plt.plot((np.array(calc)-np.array(pred)),'x',markersize=10,markeredgewidth=2)
    plt.xlim(-1,len(names))
    plt.xticks(range(len(names)),names,rotation=90)
    plt.show()


    delays_fine=np.linspace(-100,100,100000)
    phases_pred=Angle(2*np.pi*delays_fine*610.3333333333*1e-3*u.radian).wrap_at(np.pi*u.radian).value

    delays=[dtau_frac_dict[i].to(u.ns).value for i in names]# if 'C' in i]#-(0 if not(i=='E04R') else 60)
    print zip(names, delays)
    phases=[Angle((phasedict2_meas[i]-phasedict1[i]+phasedict1[ant_ref])*u.radian).wrap_at(np.pi*u.radian).value for i in names]# if 'C' in i]
    #print zip(names, phases)
    plt.figure()
    plt.scatter(delays,phases)
    plt.plot(delays_fine,phases_pred,'.',markersize=1)
    plt.xlim(-100.,100.)
    plt.ylim(-np.pi, np.pi)
    plt.xlabel('Total delay (ns)')
    plt.ylabel('Phase (radians)')
    plt.show()
"""
