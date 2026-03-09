import math
from obspy import UTCDateTime
from numpy import nan, isnan, abs
import numpy as np
import copy
from obspy.core import Stream, read


def floor_decimal(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier) / multiplier


def traceshift(trace, tt):
    """
    Function to shift traces in time given travel time

    """

    # Define frequencies
    nt = trace.stats.npts
    dt = trace.stats.delta
    freq = np.fft.fftfreq(nt, d=dt)

    # Fourier transform
    ftrace = np.fft.fft(trace.data)

    # Shift
    for i in range(len(freq)):
        ftrace[i] = ftrace[i]*np.exp(-2.*np.pi*1j*freq[i]*tt)

    # Back Fourier transform and return as trace
    rtrace = trace.copy()
    rtrace.data = np.real(np.fft.ifft(ftrace))

    # Update start time
    rtrace.stats.starttime -= tt

    return rtrace

def download_data(client=None, sta=None, start=UTCDateTime(),
                  end=UTCDateTime(), new_sr=0., verbose=False,
                  remove_response=False, zcomp='Z'):
    """
    Function to build a stream object for a seismogram in a given time window
    by getting data from a client object, either from a local SDS archive or
    from an FDSN web-service. The function performs sanity checks for
    the start times, sampling rates and window lengths.

    Parameters
    ----------
    client : :class:`~obspy.client.fdsn.Client`
        Client object
    sta : Dict
        Station metadata from :mod:`~StDb` data base
    start : :class:`~obspy.core.utcdatetime.UTCDateTime`
        Start time for request
    end : :class:`~obspy.core.utcdatetime.UTCDateTime`
        End time for request
    new_sr : float
        New sampling rate (Hz)
    verbose : bool
        Whether or not to print messages to screen during run-time
    remove_response : bool
        Remove instrument response from seismogram and resitute to true ground
        velocity (m/s) using obspy.core.trace.Trace.remove_response()
    zcomp: str
        Vertical Component Identifier. Should be a single character.
        This is different then 'Z' only for fully unknown component
        orientation (i.e., components are 1, 2, 3)

    Returns
    -------
    err : bool
        Boolean for error handling (`False` is associated with success)
    trN : :class:`~obspy.core.Trace`
        Trace of North component of motion
    trE : :class:`~obspy.core.Trace`
        Trace of East component of motion
    trZ : :class:`~obspy.core.Trace`
        Trace of Vertical component of motion

    """

    # # Output
    # print(("*     {0:s}.{1:2s} - ZNE:".format(sta.station,
    #                                           sta.channel.upper())))

    for loc in sta.location:

        # Construct location name
        if loc == "--":
            tloc = ""
        else:
            tloc = copy.copy(loc)

        # Construct Channel List
        cha = sta.channel.upper() + '?'
        msg = "*     {0:s}.{1:2s}?.{2:2s} - Checking Network".format(
            sta.station, sta.channel.upper(), loc)
        print(msg)

        # Get waveforms, with extra 1 second to avoid
        # traces cropped too short - traces are trimmed later
        try:
            st = client.get_waveforms(
                network=sta.network,
                station=sta.station,
                location=tloc,
                channel=cha,
                starttime=start,
                endtime=end+1.)
        except Exception as e:
            if verbose:
                print("* Met exception:")
                print("* " + e.__repr__())
            st = None
        else:
            if len(st) == 3:
                # It's possible if len(st)==1 that data is Z12
                print("*                  - Data Downloaded")
                # break

    # Check the correct 3 components exist
    if st is None or len(st) < 3:
        print("* Error retrieving waveforms")
        print("**************************************************")
        return True, None

    # Three components successfully retrieved
    if remove_response:
        try:
            st.remove_response()
            if verbose:
                print("*")
                print("* Restituted stream to true ground velocity.")
                print("*")
        except Exception as e:
            print("*")
            print('* Cannot remove response, moving on.')
            print("*")

    # Detrend and apply taper
    st.detrend('demean').detrend('linear').taper(
        max_percentage=0.05, max_length=5.)

    # Check start times
    if not np.all([tr.stats.starttime == start for tr in st]):
        if verbose:
            print("* Start times are not all close to true start: ")
            [print("*   "+tr.stats.channel+" " +
                   str(tr.stats.starttime)+" " +
                   str(tr.stats.endtime)) for tr in st]
            print("*   True start: "+str(start))
            print("*   -> Shifting traces to true start")
        delay = [tr.stats.starttime - start for tr in st]
        st_shifted = Stream(
            traces=[traceshift(tr, dt) for tr, dt in zip(st, delay)])
        st = st_shifted.copy()

    # Check sampling rate
    sr = st[0].stats.sampling_rate
    sr_round = float(floor_decimal(sr, 0))
    if not sr == sr_round:
        if verbose:
            print("* Sampling rate is not an integer value: ", sr)
            print("* -> Resampling")
        st.resample(sr_round, no_filter=False)

    # Try trimming
    try:
        st.trim(start, end)
    except Exception as e:
        print("* Unable to trim")
        print("* -> Skipping")
        print("**************************************************")

        return True, None

    # Check final lengths - they should all be equal if start times
    # and sampling rates are all equal and traces have been trimmed
    if not np.allclose([tr.stats.npts for tr in st[1:]], st[0].stats.npts):
        print("* Lengths are incompatible: ")
        [print("*     "+str(tr.stats.npts)) for tr in st]
        print("* -> Skipping")
        print("**************************************************")

        return True, None

    elif not np.allclose([st[0].stats.npts], int((end - start)*sr),
                        atol=1):
        print("* Length is too short: ")
        print("*    "+str(st[0].stats.npts) +
            " ~= "+str(int((end - start)*sr)))
        print("* -> Skipping")
        print("**************************************************")

        return True, None

    else:
        pass
        #print("* Waveforms Retrieved...")
        #return False, st


# ---------------------------------------------------------------------------
# Quality factor helper -----------------------------------------------------

def null_quality_factor(phiSC, phiRC, dtSC, dtRC, snrT=None):
    """
    Numerically evaluate the quality factor ``Q`` used in the MATLAB
    version of SplitLab (see :file:`NullCriterion.m`).  The result is
    positive for non-null solutions (larger is better) and negative for
    nulls.  Values range between -1 and +1; zeros are returned for very
    noisy transverse components when ``snrT`` is provided and falls
    below 3.

    Parameters
    ----------
    phiSC : float
        Fast axis from Silver-Chan method (degrees)
    phiRC : float
        Fast axis from Rotation-Correlation method (degrees)
    dtSC : float
        Delay time from Silver-Chan (seconds)
    dtRC : float
        Delay time from Rotation-Correlation (seconds)
    snrT : float, optional
        Signal-to-noise ratio of the transverse component; if supplied
        and ``snrT < 3`` the returned Q is forced to zero.

    Returns
    -------
    float
        Computed quality factor.
    """

    # mimic behaviour of MATLAB code line-by-line
    x1 = dtSC if dtSC != 0 else 1e-5
    x2 = dtRC
    X = x2 / x1

    Y = abs(phiSC - phiRC) % 90
    if Y > 45:
        Y = 90 - Y
    Y = Y / 45.0

    # compute the two distances used in the criterion
    Dist1 = np.sqrt(X**2 + (Y - 1) ** 2) / np.sqrt(0.5)
    Dist2 = np.sqrt((X - 1) ** 2 + Y**2) / np.sqrt(0.5)

    # select the smaller distance and assign sign accordingly
    if Dist1 < Dist2:
        Q = -1.0 * (1.0 - Dist1)
    else:
        Q = 1.0 * (1.0 - Dist2)

    if snrT is not None and snrT < 3:
        Q = 0.0

    return Q