"""
Python software to read data (.rad files) coming from the UWM SRT

usage:

import read_srt
S=read_srt.readSpectra(filename)

can then plot/manipulate the data in S

"""

import numpy
import datetime
import math


# the lat and long of UBC, in degrees
Latitude = 49.24966 
Longitude = -123.11934

######################################################################


class Spectrum:
    """
    A class to store a single spectrum instance
    along with the spectrum (a ndarray) it also stores the position, date, time
    and the information related to the last calibration run (if any)

    s=Spectrum(datastring)
    where datastring is the line from a .rad file that contains the
    date/time info and the spectrum

    properties:
    calibration: Tspill, Tload, Tsys, CALCONS, Trec

    date/time: UT (hours), year, month, day, doy, MJD, LST
    date (y, m, d)
    datetime (y, m, d, h, m, s)

    coordinate: az, el, d_az, d_el, ha (decimal degrees)
    ra, dec (decimal degrees)

    frequency: freq0, dfreq (MHz), nfreq

    data: freq (ndarray, MHz), spectrum (ndarray)

    """

    ##############################
    def __init__(self, datastring=None):
        """
        constructor
        """
        self.Tspill = 20
        self.Tload = 100
        self.Tsys = 200
        self.CALCONS = 1
        self.Trec = self.Tsys - self.Tspill

        # UT time in hours
        self.UT = None
        self.year = 2011
        self.month = None
        self.day = None
        self.doy = None
        self.MJD = None
        # decimal degrees
        self.az = None
        self.el = None
        # offsets
        self.d_az = None
        self.d_el = None
        # in MHz
        self.freq0 = 1420
        self.dfreq = 0.00781250
        self.nfreq = 64
        self.spectrum = None
        # last command
        self.command = None

        if (datastring is not None):
            # parse it
            d = datastring.split()
            datestring = d[0]
            d2 = datestring.split(':')
            self.year = int(d2[0])
            self.doy = int(d2[1])
            self.UT = int(d2[2]) + int(d2[3]) / 60.0 + int(d2[4]) / 3600.0

            self.az = float(d[1])
            self.el = float(d[2])
            self.d_az = float(d[3])
            self.d_el = float(d[4])
            self.freq0 = float(d[5])
            self.dfreq = float(d[6])
            self.nfreq = int(d[8])
            self.spectrum = numpy.zeros((self.nfreq,))
            for i in range(self.nfreq):
                self.spectrum[i] = float(d[9 + i])
            self.vlsr = float(d[9 + self.nfreq + 1])

    ##############################
    def __repr__(self):
        """
        returns string representation
        """
        datetime = self.datetime
        s = '%.5f %04d/%02d/%02d-%02d:%02d:%04.1f (%.3f, %.3f): %s' % (
            self.MJD + self.UT / 24.0, datetime[0], datetime[1], datetime[2],
            datetime[3], datetime[4], datetime[5], self.az + self.d_az,
            self.el + self.d_el, self.spectrum)
        return s

    ##############################
    def __len__(self):
        """
        len() returns the length of the spectrum
        """
        return len(self.spectrum)

    ##############################
    def __getitem__(self, item):
        """
        s[i]=s.spectrum[i]
        """
        return self.spectrum[item]

    ##############################
    def __setitem__(self, item, value):
        """
        s[i]=s.spectrum[i]
        """
        self.spectrum[item] = value

    ##############################
    def __setattr__(self, item, value):
        if (item == 'doy'):
            # if we set the doy, automatically translate that to y, m, d and
            # MJD
            self.__dict__[item] = value
            if (self.year is not None and value is not None):
                d = datetime.date(self.year, 1, 1) + \
                    datetime.timedelta(days=value - 1)
                self.month = d.month
                self.day = d.day
                self.MJD = cal_mjd(self.year, self.month, self.day)
        else:
            self.__dict__[item] = value

    ##############################
    def __getattr__(self, item):
        """
        provide for some extras calculated on the fly
        like date, datetime, LST, ha, ra, dec, freq
        """

        if (item == 'date'):
            return (self.year, self.month, self.day)
        elif (item == 'datetime'):
            return (self.year,
                    self.month,
                    self.day,
                    int(self.UT),
                    int(60 * (self.UT - int(self.UT))),
                    3600 * (self.UT - int(self.UT)
                            - int(60 * (self.UT - int(self.UT))) / 60.0))
        elif (item == 'LST'):
            return utc_lmst(self.MJD + self.UT / 24.0, Longitude)
        elif (item == 'ha'):
            [HA, Dec] = horz2eq(self.az + self.d_az,
                                self.el + self.d_el, Latitude)
            return HA
        elif (item == 'ra' or item == 'dec'):
            [HA, Dec] = horz2eq(self.az + self.d_az,
                                self.el + self.d_el, Latitude)
            if (item == 'ra'):
                return putrange(self.LST - HA / 15.0) * 15
            else:
                return Dec
        elif (item == 'freq'):
            return self.freq0 + numpy.arange(self.nfreq) * self.dfreq
        else:
            return self.__dict__[item]

    ##############################
    def average_power(self, lower=10, upper=10):
        """
        m, dm = s.average_power(lower=10, upper=10)
        averages the power over spectrum[lower:-upper]
        returns the average and the rms
        """
        return self.spectrum[lower:-upper].mean(), \
            self.spectrum[lower:-upper].std() / math.sqrt(len(self.spectrum[lower:-upper]))


######################################################################
class Spectra(Spectrum, list):
    """
    This class contains many spectra
    groups them for convenience, allows for operations across
    many spectra

    can create from a list of Spectrum objects

    S=Spectra([s1,s2,s3])

    can access with special functions:
    S.MJD, S.UT, S.AZ, S.EL, S.RA, S.DEC, S.FREQ0, S.NFREQ, S.DFREQ
    S.TREC, S.CALCONS, S.TSYS, S.COMMAND
    all will return lists of those values across all elements

    Also S.FREQ, S.SPECTRA will return ndarray (len(S), max(S.NFREQ))

    """

    ##############################
    def __init__(self, spectrumlist=None):
        """
        create from a list
        """
        self.spectra = spectrumlist

    def __getattr__(self, item):
        """
        make it so that you can get items from across all elements
        """
        if (item == 'MJD'):
            return [s.MJD for s in self.spectra]
        elif (item == 'UT'):
            return [s.UT for s in self.spectra]
        elif (item == 'AZ'):
            return [s.az + s.d_az for s in self.spectra]
        elif (item == 'EL'):
            return [s.el + s.d_el for s in self.spectra]
        elif (item == 'RA'):
            return [s.ra for s in self.spectra]
        elif (item == 'DEC'):
            return [s.dec for s in self.spectra]
        elif (item == 'FREQ0'):
            return [s.freq0 for s in self.spectra]
        elif (item == 'NFREQ'):
            return [s.nfreq for s in self.spectra]
        elif (item == 'DFREQ'):
            return [s.dfreq for s in self.spectra]
        elif (item == 'TREC'):
            return [s.Trec for s in self.spectra]
        elif (item == 'CALCONS'):
            return [s.CALCONS for s in self.spectra]
        elif (item == 'TSYS'):
            return [s.Tsys for s in self.spectra]
        elif (item == 'COMMAND'):
            return [s.command for s in self.spectra]
        elif (item == 'FREQ'):
            maxnfreq = max(self.NFREQ)
            f = numpy.zeros((len(self.spectra), maxnfreq))
            for i in range(len(self.spectra)):
                f[i, :self.spectra[i].nfreq] = self.spectra[i].freq
            return f
        elif (item == 'SPECTRA'):
            maxnfreq = max(self.NFREQ)
            s = numpy.zeros((len(self.spectra), maxnfreq))
            for i in range(len(self.spectra)):
                s[i, :self.spectra[i].nfreq] = self.spectra[i].spectrum
            return s
        else:
            return self.__dict__[item]

    ##############################
    def __repr__(self):
        """
        string representation
        """
        return repr(self.spectra)

    ##############################
    def __getitem__(self, item):
        """ just return the self.spectra[item] data
        """
        return self.spectra[item]

    ##############################
    def __setitem__(self, item, value):
        """ set the self.spectra[item] data
        """
        self.spectra[item] = value

    ##############################
    def __len__(self):
        """
        len(S) returns the number of Spectrum elements
        """
        return len(self.spectra)

    ##############################

    def average_power(self, lower=10, upper=10):
        """
        M,dM=S.average_power(lower=10,upper=10)
        returns the average power and rms for each element
        """
        M = numpy.zeros((len(self.spectra),))
        S = numpy.zeros((len(self.spectra),))
        for i in range(len(self)):
            M[i], S[i] = self[i].average_power(lower=lower, upper=upper)
        return M, S

    ##############################
    def write_average_power(self, filename, lower=10, upper=10):
        """
        result=S.writeaveragepower(filename, lower=10, upper=10)
        writes the average power data to a file
        each line has the MJD (fractional), Az, El, RA, Dec, Power, rms
        """
        M, S = self.average_power(lower=lower, upper=upper)
        try:
            f = open(filename, 'w')
        except IOError:
            print("Could not open file %s for writing" % filename)
            return None
        f.write(
            '# result of average, lower channel=%d, upper channel=%d\n' %
            (lower, max(
                self.NFREQ) - upper))
        f.write('# i  MJD         Az      El      RA      Dec     Power     dPower\n')
        for i in range(len(self)):
            f.write(
                '%04d %.5f %7.3f %7.3f %7.3f %7.3f %9.3f %9.3f\n' %
                (i,
                 self[i].MJD +
                    self[i].UT /
                    24.0,
                    self[i].az +
                    self[i].d_az,
                    self[i].el +
                    self[i].d_el,
                    self[i].ra,
                    self[i].dec,
                    M[i],
                    S[i]))
        f.close()
        return True

######################################################################


def read_spectra(filename=None):
    """
    S=read_spectra(filename)
    reads the spectra in filename into the Spectra object S
    """

    #try:
    #    f = open(filename)
    #except IOError:
    #    print("Could not open file: %s" % filename)
    #    return None

    # default values to start
    Tsys = 100
    CALCONS = 1.0
    Tspill = 20
    Tload = 200
    Trec = Tsys - Tspill
    with open(filename) as f:
        lines = [l.rstrip() for l in f]
    #lines = f.readlines()
    S = []
    command = None
    for line in lines:
        if (line.startswith('*')):
            # comment line
            if ('STATION' in line):
                continue
            elif ('tsys' in line):
                d = line.split()
                Tsys = float(d[2])
                CALCONS = float(d[4])
                Trec = float(d[6])
                Tload = float(d[8])
                Tspill = float(d[10])
                command = 'CAL'
            elif ('NPOINT' in line):
                command = 'NPOINT'
            else:
                command = line.rstrip('* ')
        else:
            s = Spectrum(line)
            s.Tsys = Tsys
            s.CALCONS = CALCONS
            s.Trec = Trec
            s.Tload = Tload
            s.Tspill = Tspill
            s.command = command
            S.append(s)
            #command = None

    return Spectra(S)


######################################################################
# Utility functions
######################################################################
def mjd_cal(mjd):
    """convert MJD to calendar date (yr,mn,dy)
    """

    JD = mjd + 2400000.5

    JD += .5
    Z = int(JD)
    F = JD - Z
    if (Z < 2299161):
        A = Z
    else:
        alpha = int((Z - 1867216.25) / 36524.25)
        A = Z + 1 + alpha - int(alpha / 4)
    B = A + 1524
    C = int((B - 122.1) / 365.25)
    D = int(365.25 * C)
    E = int((B - D) / 30.6001)
    day = B - D - int(30.6001 * E) + F
    if (E < 14):
        month = E - 1
    else:
        month = E - 13
    if (month <= 2):
        year = C - 4715
    else:
        year = C - 4716

    return (year, month, day)
######################################################################


def cal_mjd(yr, mn, dy):
    """ convert calendar date to MJD
    year,month,day (may be decimal) are normal parts of date (Julian)"""

    m = mn
    if (yr < 0):
        y = yr + 1
    else:
        y = yr
    if (m < 3):
        m += 12
        y -= 1
    if (yr < 1582 or (yr == 1582 and (mn < 10 or (mn == 10 and dy < 15)))):
        b = 0
    else:
        a = int(y / 100)
        b = int(2 - a + a / 4)

    jd = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + dy + b - 1524.5
    mjd = jd - 2400000.5

    return (mjd)

######################################################################


def horz2eq(Az, El, lat):
    """
    [HA,Dec]=horz2eq(Az,El,lat)
    horizon coords to equatorial
    all decimal degrees
    The sign convention for azimuth is north zero, east +pi/2.
    from slalib sla_h2e
    https://starlink.jach.hawaii.edu/viewvc/trunk/libraries/sla/h2e.f?view=markup
    """

    if (isinstance(Az, numpy.ndarray)):
        sa = numpy.sin(Az * math.pi / 180)
        ca = numpy.cos(Az * math.pi / 180)
        se = numpy.sin(El * math.pi / 180)
        ce = numpy.cos(El * math.pi / 180)
        sl = numpy.sin(lat * math.pi / 180)
        cl = numpy.cos(lat * math.pi / 180)

        # HA,Dec as (x,y,z)
        x = -ca * ce * sl + se * cl
        y = -sa * ce
        z = ca * ce * cl + se * sl

        r = numpy.sqrt(x * x + y * y)
        ha = numpy.arctan2(y, x)
        ha[numpy.where(r == 0)] = 0

        dec = numpy.arctan2(z, r)

    else:
        sa = math.sin(Az * math.pi / 180)
        ca = math.cos(Az * math.pi / 180)
        se = math.sin(El * math.pi / 180)
        ce = math.cos(El * math.pi / 180)
        sl = math.sin(lat * math.pi / 180)
        cl = math.cos(lat * math.pi / 180)

        # HA,Dec as (x,y,z)
        x = -ca * ce * sl + se * cl
        y = -sa * ce
        z = ca * ce * cl + se * sl

        r = math.sqrt(x * x + y * y)
        if (r == 0):
            ha = 0
        else:
            ha = math.atan2(y, x)
        dec = math.atan2(z, r)

    return [ha * 180 / math.pi, dec * 180 / math.pi]

######################################################################


def utc_gmst(ut):
    """ *  Conversion from universal time to sidereal time (double precision)
    given input time ut expressed as MJD
    result is GMST in hours
    """

    ut1 = ut

    D2PI = 6.283185307179586476925286766559
    S2R = 7.272205216643039903848711535369e-5

    #  Julian centuries from fundamental epoch J2000 to this UT
    TU = (ut1 - 51544.5) / 36525

    # GMST at this UT
    gmst = math.modf(ut1)[0] * D2PI + (24110.54841 +
                                       (8640184.812866 + (0.093104 - 6.2 - 6 * TU) * TU) * TU) * S2R
    gmst = gmst * 24 / D2PI

    gmst = putrange(gmst)

    return gmst

######################################################################


def utc_lmst(ut, longitude):
    """ returns the LMST given the UT date/time (expressed as MJD),
    and longitude (degrees, + going to east)
    LMST is in hours
    """

    longitude = checksex(longitude)

    lmst = utc_gmst(ut)
    lmst += longitude / 15

    if (lmst < 0):
        lmst += 24
    if (lmst >= 24):
        lmst -= 24
    return lmst


######################################################################
def dec2sex(x):
    """ convert decimal to sexadecimal
    """

    sign = 1
    if (x < 0):
        sign = -1
    x = math.fabs(x)

    d = int(x)
    m = int(60 * (x - d))
    s = 60 * (60 * (x - d) - m)
    if (sign == -1):
        d *= -1

    return (d, m, s)

######################################################################


def sex2dec(d, m, s):
    """ convert sexadecimal d,m,s to decimal
    """

    sign = 1
    if (d < 0):
        sign = -1
        d = math.fabs(d)
    x = d + m / 60.0 + s / 3600.0
    x = x * sign

    return x

######################################################################


def dec2sexstring(x, includesign=0, digits=2):
    """ convert a decimal to a sexadecimal string
    if includesign=1, then always use a sign
    can specify number of digits on seconds
    """

    (d, m, s) = dec2sex(float(x))

    sint = int(s)
    if (digits > 0):
        sfrac = (10**digits) * (s - sint)
        ss2 = '%02' + 'd' + '.%0' + ('%d' % digits) + 'd'
        ss = ss2 % (sint, sfrac)
    elif (digits == 0):
        ss = '%02d' % sint
    else:
        mfrac = 10**(math.fabs(digits)) * (s / 60.0)
        ss2 = '%02' + 'd' + '.%0' + ('%d' % math.fabs(digits)) + 'd'
        ss = ss2 % (m, mfrac)

    if (not includesign):
        if (digits >= 0):
            sout = "%02d:%02d:%s" % (d, m, ss)
        else:
            sout = "%02d:%s" % (d, ss)
    else:
        sign = '+'
        if (d < 0):
            sign = '-'
        if (digits >= 0):
            sout = "%s%02d:%02d:%s" % (sign, math.fabs(d), m, ss)
        else:
            sout = "%s%02d:%s" % (sign, math.fabs(d), ss)

    return sout

######################################################################


def sexstring2dec(sin):
    """ convert a sexadecimal string to a float
    """

    [d, m, s] = sin.split(':')
    return sex2dec(int(d), int(m), float(s))

######################################################################


def checksex(x):
    """ check and see if the argument is a sexadecimal string
    or a float

    return the float version
    """

    y = 0
    try:
        if ((x).count(':') == 2):
            y = sexstring2dec(x)
    except BaseException:
        y = float(x)

    return y

######################################################################


def putrange(x, r=24):
    """ puts a value in the range [0,r)
    """

    while (x < 0):
        x += r
    while (x >= r):
        x -= r
    return x
