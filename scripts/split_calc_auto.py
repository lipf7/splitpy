#!/usr/bin/env python

# Copyright 2019 Pascal Audet & Andrew Schaeffer
# Modified to use local event data files instead of SDS or network download

# -*- coding: utf-8 -*-


#2025-12-01修改 Added more robust handling of location codes in local event data files
#不再使用stdb管理本地事件数据文件，改为直接从指定目录读取
##新增了get_adaptive_sks_window函数，用于自适应地确定SKS波的分析窗口
##2025-12-19 修改：新增了search_best_window_and_filters_and_snr函数，用于在多个过滤器和SNR阈值下搜索最佳窗口和过滤器

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
import stdb
import copy
import csv

from obspy.core.event import Catalog, Event, Origin, Magnitude
from obspy import UTCDateTime
from obspy import read
from obspy import Stream
from obspy.clients.fdsn import Client as FDSN_Client
from splitpy import utils
from splitpy import Split, DiagPlot

from argparse import ArgumentParser
from os.path import exists as exist
from pathlib import Path

matplotlib.use('Agg')

def get_arguments_calc_auto(argv=None):

    parser = ArgumentParser(
        usage="%(prog)s [arguments] <station database>",
        description="Script for SKS splitting analysis using local event data files.")
    parser.add_argument(
        "indb",
        help="Station Database to process from.",
        type=str)
    parser.add_argument(
        "--keys",
        action="store",
        type=str,
        dest="stkeys",
        default="",
        help="Specify a comma separated list of station keys for " +
        "which to perform the analysis.")
    parser.add_argument(
        "-V", "--verbose",
        action="store_true",
        dest="verb",
        default=False,
        help="Specify to increase verbosity.")
    parser.add_argument(
        "-O", "--overwrite",
        action="store_true",
        dest="ovr",
        default=False,
        help="Force the overwriting of pre-existing Split results.")
    parser.add_argument(
        "--zcomp", 
        dest="zcomp",
        type=str,
        default="Z",
        help="Specify the Vertical Component Channel Identifier. "+
        "[Default Z].")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        dest="skip",
        default=False,
        help="Skip any event for which existing splitting results are " +
        "saved to disk.")
    parser.add_argument(
        "--calc",
        action="store_true",
        dest="calc",
        default=False,
        help="Analyze data for shear-wave splitting. [Default saves data "+
        "to folders for subsequent analysis]")
    parser.add_argument(
        "--plot-diagnostic",
        dest="diagplot",
        type=str,
        default=None,
        help="If set, save diagnostic plots to this path instead of showing them.")
    parser.add_argument(
        "--recalc",
        action="store_true",
        dest="recalc",
        default=False,
        help="Re-calculate estimates and overwrite existing splitting "+
        "results without re-downloading data. [Default False]")

    # Local Event Data Settings
    DataGroup = parser.add_argument_group(
        title="Local Event Data Settings",
        description="Settings for using local event data files.")
    DataGroup.add_argument(
        "--event-datadir",
        action="store",
        type=str,
        dest="eventdatadir",
        default=None,
        required=True,
        help="Path to directory containing event data directories. " +
        "Each subdirectory should be named by event time (YYYYMMDD_HHMMSS) " +
        "and contain SAC files named as NETWORK.STATION.LOC.CHANNEL.sac " +
        "(e.g., IU.ANMO.00.BHZ.sac)")
    DataGroup.add_argument(
        "--data-format",
        action="store",
        type=str,
        dest="dataformat",
        default="SAC",
        help="Data file format (SAC or MSEED). [Default SAC]")

    # Constants Settings
    ConstGroup = parser.add_argument_group(
        title='Parameter Settings',
        description="Miscellaneous default values and settings")
    ConstGroup.add_argument(
        "--sampling-rate",
        action="store",
        type=float,
        dest="new_sampling_rate",
        default=10.,
        help="Specify new sampling rate in Hz. [Default 10.]")
    ConstGroup.add_argument(
        "--min-snr",
        action="store",
        type=float,
        dest="msnr",
        default=4.,
        help="Minimum SNR value calculated on the radial (Q) component "+
        "to proceed with analysis (dB). [Default 5.]")
    ConstGroup.add_argument(
        "--window",
        action="store",
        type=float,
        dest="dts",
        default=120.,
        help="Specify time window length before and after the SKS "
        "arrival. The total window length is 2*dst (sec). [Default 120]")
    ConstGroup.add_argument(
        "--max-delay",
        action="store",
        type=float,
        dest="maxdt",
        default=4.,
        help="Specify the maximum delay time in search (sec). "+
        "[Default 4]")
    ConstGroup.add_argument(
        "--dt-delay",
        action="store",
        type=float,
        dest="ddt",
        default=0.1,
        help="Specify the time delay increment in search (sec). "+
        "[Default 0.1]")
    ConstGroup.add_argument(
        "--dphi",
        action="store",
        type=float,
        dest="dphi",
        default=1.,
        help="Specify the fast angle increment in search (degree). "+
        "[Default 1.]")
    ConstGroup.add_argument(
        "--snrT",
        action="store",
        type=float,
        dest="snrTlim",
        default=1.,
        help="Specify the minimum SNR Threshold for the Transverse " +
        "component to be considered Non-Null. [Default 1.]")
    ConstGroup.add_argument(
        "--fmin",
        action="store",
        type=float,
        dest="fmin",
        default=0.02,
        help="Specify the minimum frequency corner for bandpass " +
        "filter (Hz). [Default 0.02]")
    ConstGroup.add_argument(
        "--fmax",
        action="store",
        type=float,
        dest="fmax",
        default=0.5,
        help="Specify the maximum frequency corner for bandpass " +
        "filter (Hz). [Default 0.5]")
    ConstGroup.add_argument(
        "--filter-set",
        action="store",
        type=str,
        dest="filterset",
        default=None,
        help=("Predefined bandpass filter set. "
            "Options: low (0.02-0.2), mid (0.05-0.5), high (0.1-1.0). "
            "If set, overrides --fmin and --fmax.")
        )
    ConstGroup.add_argument(
        "--filter-bands",
        action="store",
        type=str,
        dest="filterbands",
        default="0.02-0.2,0.04-0.5,0.1-1.0",
        help=("Comma-separated bandpass frequency bands (Hz), "
            "used to test SNR. Format: fmin-fmax,fmin-fmax,... "
            "[Default: 0.02-0.2,0.05-0.5,0.1-1.0]")
        )

    # Event Selection Criteria
    EventGroup = parser.add_argument_group(
        title="Event Settings",
        description="Settings associated with refining "
        "the events to include in matching station pairs")
    EventGroup.add_argument(
        "--local-event",
        action="store",
        type=str,
        dest="localevent",
        default=None,
        help="Path to local CSV file containing event catalogue in format: "
            "time(UTC),latitude,longitude,depth(km),magnitude")
    EventGroup.add_argument(
        "--start",
        action="store",
        type=str,
        dest="startT",
        default="",
        help="Specify a UTCDateTime compatible string representing " +
        "the start time for the event search.")
    EventGroup.add_argument(
        "--end",
        action="store",
        type=str,
        dest="endT",
        default="",
        help="Specify a UTCDateTime compatible string representing " +
        "the end time for the event search.")
    EventGroup.add_argument(
        "--reverse",
        action="store_true",
        dest="reverse",
        default=False,
        help="Reverse order of events.")
    EventGroup.add_argument(
        "--min-mag",
        action="store",
        type=float,
        dest="minmag",
        default=6.0,
        help="Specify the minimum magnitude of event for which to " +
        "search. [Default 6.0]")
    EventGroup.add_argument(
        "--max-mag",
        action="store",
        type=float,
        dest="maxmag",
        default=None,
        help="Specify the maximum magnitude of event for which to " +
        "search. [Default None]")

    # Geometry Settings
    GeomGroup = parser.add_argument_group(
        title="Geometry Settings",
        description="Settings associatd with the "
        "event-station geometries")
    GeomGroup.add_argument(
        "--min-dist",
        action="store",
        type=float,
        dest="mindist",
        default=85.,
        help="Specify the minimum great circle distance (degrees) " +
        "between the station and event. [Default 85]")
    GeomGroup.add_argument(
        "--max-dist",
        action="store",
        type=float,
        dest="maxdist",
        default=120.,
        help="Specify the maximum great circle distance (degrees) " +
        "between the station and event. [Default 120]")
    GeomGroup.add_argument(
        "--phase",
        action="store",
        type=str,
        dest="phase",
        default='SKS',
        help="Specify the phase name to use. Options are 'SKS' or 'SKKS'. [Default 'SKS']")

    args = parser.parse_args(argv)

    # Check inputs
    if not exist(args.indb):
        parser.error("Input file " + args.indb + " does not exist")
    
    if not args.eventdatadir or not exist(args.eventdatadir):
        parser.error("Event data directory does not exist or not specified: " + str(args.eventdatadir))

    # create station key list
    if len(args.stkeys) > 0:
        args.stkeys = args.stkeys.split(',')

    # construct start time
    if len(args.startT) > 0:
        try:
            args.startT = UTCDateTime(args.startT)
        except:
            parser.error(
                "Cannot construct UTCDateTime from start time: " +
                args.startT)
    else:
        args.startT = None

    # construct end time
    if len(args.endT) > 0:
        try:
            args.endT = UTCDateTime(args.endT)
        except:
            parser.error(
                "Cannot construct UTCDateTime from end time: " +
                args.endT)
    else:
        args.endT = None

    # Check existing file behaviour
    if args.skip and args.ovr:
        args.skip = False
        args.ovr = False

    # Check Datatype specification
    if args.dataformat.upper() not in ['SAC', 'MSEED']:
        parser.error(
            "Error: Local Event Data must be of type 'SAC' or 'MSEED'.")

    # Check selected phase
    if args.phase not in ['SKS', 'SKKS', 'PKS']:
        parser.error(
            "Error: choose between 'SKS', 'SKKS and 'PKS'.")

    # Check distances for all phases
    if not args.mindist:
        if args.phase == 'SKS':
            args.mindist = 85.
        elif args.phase == 'SKKS':
            args.mindist = 90.
        elif args.phase == 'PKS':
            args.mindist = 130.
    if not args.maxdist:
        if args.phase == 'SKS':
            args.maxdist = 120.
        elif args.phase == 'SKKS':
            args.maxdist = 130.
        elif args.phase == 'PKS':
            args.maxdist = 150.
    if args.mindist < 85. or args.maxdist > 180.:
        parser.error(
            "Distances should be between 85 and 180 deg. for " +
            "teleseismic 'SKS', 'SKKS' and 'PKS' waves.")

    # --------------------------------------------------
    # Predefined filter frequency sets
    # --------------------------------------------------
    FILTER_SETS = {
        "low":  (0.02, 0.2),   # 你要求的第一个选项
        "mid":  (0.05, 0.5),
        "high": (0.1, 1.0),
    }

    if args.filterset is not None:
        if args.filterset not in FILTER_SETS:
            parser.error(
                f"Unknown filter-set '{args.filterset}'. "
                f"Available options: {', '.join(FILTER_SETS.keys())}"
            )
        args.fmin, args.fmax = FILTER_SETS[args.filterset]
    # --------------------------------------------------
    # Parse multiple filter bands
    # --------------------------------------------------
    args.filterbands_list = []

    for band in args.filterbands.split(","):
        try:
            fmin, fmax = map(float, band.split("-"))
            if fmin <= 0 or fmax <= fmin:
                raise ValueError
            args.filterbands_list.append((fmin, fmax))
        except Exception:
            parser.error(
                f"Invalid filter band format: '{band}'. "
                "Expected fmin-fmax (e.g. 0.02-0.2)"
            )


    return args


class LocalEventDataClient:
    """Client for reading local event data files"""
    
    def __init__(self, event_dir, data_format="SAC"):
        self.event_dir = Path(event_dir)
        self.data_format = data_format.upper()
    
    def get_waveforms(self, network, station, location, channel, starttime=None, endtime=None):
        """
        Read waveforms from local event directory.
        Assumes files are named as: NETWORK.STATION.LOC.CHANNEL.sac
        """
        # 确保 location 是字符串且已清理
        if isinstance(location, list):
            if len(location) > 0:
                location = str(location[0]).strip().strip("'\"")
            else:
                location = "00"
        else:
            location = str(location).strip().strip("'\"") if location else "00"
        
        # 如果 location 是空字符串，转换为 "00"（SDS格式常用）
        if location == "":
            location = "00"
        
        # 构建可能的文件名列表
        possible_filenames = []
        
        # 1. 标准格式: NETWORK.STATION.LOCATION.CHANNEL.sac
        if self.data_format == "SAC":
            possible_filenames.append(f"{network}.{station}.{location}.{channel}.sac")
        else:  # MSEED
            possible_filenames.append(f"{network}.{station}.{location}.{channel}.mseed")
        
        # 2. 无定位码格式: NETWORK.STATION.CHANNEL.sac
        if self.data_format == "SAC":
            possible_filenames.append(f"{network}.{station}.{channel}.sac")
        else:
            possible_filenames.append(f"{network}.{station}.{channel}.mseed")
        
        # 3. 尝试空定位码（双点号）
        if self.data_format == "SAC":
            possible_filenames.append(f"{network}.{station}..{channel}.sac")
        else:
            possible_filenames.append(f"{network}.{station}..{channel}.mseed")
        
        # 4. 尝试不同的定位码表示
        for loc in ["", "00", "01", "10", "20"]:
            if self.data_format == "SAC":
                possible_filenames.append(f"{network}.{station}.{loc}.{channel}.sac")
            else:
                possible_filenames.append(f"{network}.{station}.{loc}.{channel}.mseed")
        
        # 去重
        possible_filenames = list(set(possible_filenames))
        
        # 尝试所有可能的文件名
        for filename in possible_filenames:
            filepath = self.event_dir / filename
            
            if filepath.exists():
                # 读取数据文件
                try:
                    st = read(str(filepath))
                    return st
                except Exception as e:
                    continue  # 继续尝试下一个文件名
        
        # 如果所有尝试都失败
        raise FileNotFoundError(
            f"No data file found for {network}.{station}.{location}.{channel}\n"
            f"Tried: {possible_filenames[:5]}\n"
            f"in directory: {self.event_dir}"
        )


def normalize_station_info(sta):
    """规范化台站信息，确保字段类型正确"""
    
    # 创建副本
    import copy
    sta_norm = copy.copy(sta)
    
    # 处理location字段
    if isinstance(sta_norm.location, list):
        if len(sta_norm.location) > 0:
            sta_norm.location = str(sta_norm.location[0]).strip()
        else:
            sta_norm.location = ""
    
    # 确保location是字符串
    sta_norm.location = str(sta_norm.location) if sta_norm.location else ""
    
    return sta_norm


def load_local_event_data(split, event_dir, station_info, data_format="SAC", 
                        new_sampling_rate=20.0, verb=False):
    """
    Load data from local event directory for a specific station.
    
    Parameters:
    -----------
    split : Split object
    event_dir : Path
        Directory containing event data files
    station_info : dict
        Station information including network, station, channel
    data_format : str
        Data format (SAC or MSEED)
    new_sampling_rate : float
        Target sampling rate
    verb : bool
        Verbose output
    
    Returns:
    --------
    bool : True if data loaded successfully, False otherwise
    """
    try:
        # Initialize LocalEventDataClient
        client = LocalEventDataClient(event_dir, data_format)
        
        # 规范化台站信息
        sta = normalize_station_info(station_info)
        
        # 确定通道前缀（例如从"BHZ"中提取"BH"）
        # 首先从channel字段获取通道列表
        channel_list = []
        if isinstance(sta.channel, str):
            # 按逗号分割通道名
            if ',' in sta.channel:
                channel_list = [ch.strip() for ch in sta.channel.split(',')]
            else:
                channel_list = [sta.channel.strip()]
        elif isinstance(sta.channel, list):
            channel_list = [str(ch).strip() for ch in sta.channel]
        else:
            # 默认使用BHZ, BHN, BHE
            channel_list = ["BHZ", "BHN", "BHE"]
        
        # 从通道列表中提取前缀
        chan_prefix = "BH"  # 默认值
        if len(channel_list) > 0:
            first_channel = channel_list[0]
            if len(first_channel) >= 2:
                chan_prefix = first_channel[:2]  # 取前两个字符作为前缀
        
        # 通道组件 - 使用BHE/BHN/BHZ，而不是BH1/BH2/BHZ
        z_chan = f"{chan_prefix}Z"  # 垂直分量
        n_chan = f"{chan_prefix}N"  # 北分量
        e_chan = f"{chan_prefix}E"  # 东分量
        
        # 处理location
        if verb:
            print(f"  Station: {sta.network}.{sta.station}")
            print(f"  Location: '{sta.location}' (normalized)")
            print(f"  Channels: {z_chan}, {n_chan}, {e_chan}")
        
        # 读取垂直分量
        if verb:
            print(f"  Reading vertical component: {sta.network}.{sta.station}.{sta.location}.{z_chan}")
        try:
            st_z = client.get_waveforms(sta.network, sta.station, sta.location, z_chan)
        except FileNotFoundError:
            # 如果垂直分量找不到，尝试使用zcomp参数
            z_chan_alt = f"{chan_prefix}{args.zcomp}"
            if verb:
                print(f"  Trying alternative vertical component: {z_chan_alt}")
            st_z = client.get_waveforms(sta.network, sta.station, sta.location, z_chan_alt)
        
        # 读取北分量
        if verb:
            print(f"  Reading north component: {sta.network}.{sta.station}.{sta.location}.{n_chan}")
        st_n = client.get_waveforms(sta.network, sta.station, sta.location, n_chan)
        
        # 读取东分量
        if verb:
            print(f"  Reading east component: {sta.network}.{sta.station}.{sta.location}.{e_chan}")
        st_e = client.get_waveforms(sta.network, sta.station, sta.location, e_chan)
        
        # 选择每个分量的第一个轨迹并合并为统一的 Stream（确保单一 Trace per component）
        try:
            tr_z = st_z[0].copy()
        except Exception:
            tr_z = None
        try:
            tr_n = st_n[0].copy()
        except Exception:
            tr_n = None
        try:
            tr_e = st_e[0].copy()
        except Exception:
            tr_e = None

        # 确保都有轨迹
        st_all = Stream()
        if tr_z is not None:
            st_all.append(tr_z)
        if tr_n is not None:
            st_all.append(tr_n)
        if tr_e is not None:
            st_all.append(tr_e)
        
        # 确保所有三分量轨迹都有数据
        if len(st_all) != 3:
            if verb:
                print(f"  Warning: Expected 3 traces, got {len(st_all)}")
                for tr in st_all:
                    try:
                        print(f"    {tr.id}")
                    except Exception:
                        print("    <trace without id>")
            return False

        # 对齐时间范围：取最早 start 和最晚 end，并用 pad=True 填充缺失区段
        try:
            t_start = min([tr.stats.starttime for tr in st_all])
            t_end = max([tr.stats.endtime for tr in st_all])
            st_all.trim(starttime=t_start, endtime=t_end, pad=True, fill_value=0)
        except Exception:
            if verb:
                print("  Warning: Failed to align trace time spans")
            return False
        
        # 如果需要，重采样
        if new_sampling_rate and new_sampling_rate > 0:
            original_sr = st_all[0].stats.sampling_rate
            if abs(original_sr - new_sampling_rate) > 0.1:
                if verb:
                    print(f"  Resampling from {original_sr} Hz to {new_sampling_rate} Hz")
                st_all.resample(new_sampling_rate)
        
        # 分配给split对象
        split.dataZNE = st_all
        
        # 在元数据中设置开始和结束时间
        split.meta.tstart = st_all[0].stats.starttime
        split.meta.tend = st_all[0].stats.endtime
        
        return True
        
    except Exception as e:
        if verb:
            print(f"  Error loading local event data: {e}")
        return False

from obspy.taup import TauPyModel

def get_adaptive_sks_window(split):
    model = TauPyModel(model="iasp91")

    arrivals = model.get_travel_times(
        distance_in_degree=split.meta.gac,
        source_depth_in_km=split.meta.dep,
        phase_list=["SKS", "S"]
    )

    t_sks = None
    t_s_list = []

    for a in arrivals:
        if a.name == "SKS":
            t_sks = a.time
        elif a.name == "S":
            t_s_list.append(a.time)

    if t_sks is None:
        raise RuntimeError("No SKS arrival")

    dt = np.nan   # 永远先定义
    if t_s_list and t_sks:
        t_s = min(t_s_list, key=lambda t: abs(t - t_sks))
        dt = abs(t_s - t_sks)
        print(dt)
        if dt >= 20:
            half = 20.0
        else:
            half = max(dt - 5.0, 15.0)
    else:
        half = 25.0
    if np.isnan(dt):
        print(f" ##Using fixed half-window of {half} sec (no nearby S phase)")
    else:
        print(f" ##Using fixed half-window of {half} sec (SKS-SKKS dt = {dt:.2f} sec)")
    t0 = split.meta.time + t_sks
    return t0 - half, t0 + half

def search_best_window_and_filter(
    split,
    base_t1, base_t2,
    filterbands,
    shift_sec=5,
    step=1.0,
    snr_comp="R"
):
    best = {
        "snr": -1e9,
        "t1": None,
        "t2": None,
        "fmin": None,
        "fmax": None
    }

    for ishift in np.arange(-shift_sec, shift_sec , step):
        t1 = base_t1 + ishift
        t2 = base_t2 + ishift
        dt = t2 - t1

        for fmin, fmax in filterbands:
            split.calc_snr(t1=t1, dt=dt, fmin=fmin, fmax=fmax)

            snr = split.meta.snrt if snr_comp == "T" else split.meta.snrq

            if snr > best["snr"]:
                best.update(
                    snr=snr,
                    t1=t1,
                    t2=t2,
                    fmin=fmin,
                    fmax=fmax
                )

    return best


def main(args=None):

    print()
    print(r"###################################################################")
    print(r"#            _ _ _                _                     _         #")
    print(r"#  ___ _ __ | (_) |_     ___ __ _| | ___     __ _ _   _| |_ ___   #")
    print(r"# / __| '_ \| | | __|   / __/ _` | |/ __|   / _` | | | | __/ _ \  #")
    print(r"# \__ \ |_) | | | |_   | (_| (_| | | (__   | (_| | |_| | || (_) | #")
    print(r"# |___/ .__/|_|_|\__|___\___\__,_|_|\___|___\__,_|\__,_|\__\___/  #")
    print(r"#     |_|          |_____|             |_____|                    #")
    print(r"#                                                                 #")
    print(r"###################################################################")
    print()

    if args is None:
        # Run Input Parser
        args = get_arguments_calc_auto()
    
    # 添加参数检查
    if args is None:
        print("错误：无法解析命令行参数")
        return
    
    if not hasattr(args, 'eventdatadir'):
        print("错误：缺少eventdatadir参数")
        print("请使用 --event-data-dir 参数指定事件数据目录")
        return
    
    if args.eventdatadir is None:
        print("错误：event-data-dir参数未设置")
        return
    
    print("Using local event data from:", args.eventdatadir)

    # Load Database
    # stdb=0.1.4
    try:
        db, stkeys = stdb.io.load_db(fname=args.indb, keys=args.stkeys)

    # stdb=0.1.3
    except:
        db = stdb.io.load_db(fname=args.indb)

        # Construct station key loop
        allkeys = db.keys()
        sorted(allkeys)

        # Extract key subset
        if len(args.stkeys) > 0:
            stkeys = []
            for skey in args.stkeys:
                stkeys.extend([s for s in allkeys if skey in s])
        else:
            stkeys = db.keys()
            sorted(stkeys)

    # Loop over station keys
    for stkey in list(stkeys):

        # Extract station information from dictionary
        sta = db[stkey]

        # Output directory
        datapath = Path('DATA') / stkey
        if not datapath.is_dir():
            datapath.mkdir(parents=True)

        # Establish client for events (only for catalog if not using local event CSV)
        event_client = FDSN_Client()

        # Get catalogue search start time
        if args.startT is None:
            tstart = sta.startdate
        else:
            tstart = args.startT

        # Get catalogue search end time
        if args.endT is None:
            tend = sta.enddate
        else:
            tend = args.endT
        if tstart > sta.enddate or tend < sta.startdate:
            continue

        # Temporary print locations
        tlocs = copy.copy(sta.location)
        if len(tlocs) == 0:
            tlocs = ['']
        for il in range(0, len(tlocs)):
            if len(tlocs[il]) == 0:
                tlocs.append("--")

        # Update Display
        print(" ")
        print(" ")
        print("|"+"="*50+"|")
        print("|                   {0:>8s}                       |".format(
            sta.station))
        print("|"+"="*50+"|")
        print("|  Station: {0:>2s}.{1:5s}                               |".format(
            sta.network, sta.station))
        print("|      Channel: {0:2s}; Locations: {1:15s}     |".format(
            sta.channel, ",".join(tlocs)))
        print("|      Lon: {0:7.2f}; Lat: {1:6.2f}                   |".format(
            sta.longitude, sta.latitude))
        print("|      Start time: {0:19s}             |".format(
            sta.startdate.strftime("%Y-%m-%d %H:%M:%S")))
        print("|      End time:   {0:19s}             |".format(
            sta.enddate.strftime("%Y-%m-%d %H:%M:%S")))
        print("|"+"-"*50+"|")
        print("| Searching Possible events:                       |")
        print("|   Start: {0:19s}                     |".format(
            tstart.strftime("%Y-%m-%d %H:%M:%S")))
        print("|   End:   {0:19s}                     |".format(
            tend.strftime("%Y-%m-%d %H:%M:%S")))
        if args.maxmag is None:
            print("|   Mag:   >{0:3.1f}".format(args.minmag) +
                  "                                    |")
        else:
            msg = "|   Mag:   {0:3.1f}".format(args.minmag) + \
                " - {0:3.1f}".format(args.maxmag) + \
                "                           |"
            print(msg)

        print("| ...                                              |")

        # Get catalogue using deployment start and end
        if args.localevent is not None:
            # 从本地CSV加载事件
            cat = Catalog()
            with open(args.localevent, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)  # 跳过表头
                for row in reader:
                    try:
                        otime = UTCDateTime(row[0])
                        lat = float(row[1])
                        lon = float(row[2])
                        dep = float(row[3])
                        mag = float(row[4])
                        ev = Event()
                        origin = Origin(time=otime, latitude=lat, longitude=lon, depth=dep)
                        magnitude = Magnitude(mag=mag)
                        ev.origins = [origin]
                        ev.magnitudes = [magnitude]
                        cat.events.append(ev)
                    except Exception as e:
                        print("Error reading event row:", row, "Error:", e)
        else:
            # 原FDSN下载方式获取事件目录
            cat = event_client.get_events(
                starttime=tstart,
                endtime=tend,
                minmagnitude=args.minmag,
                maxmagnitude=args.maxmag)

        # 先扫描本地事件数据目录，按 start/end 限定可用事件目录
        available_timekeys = set()
        try:
            event_root = Path(args.eventdatadir)
            for p in sorted(event_root.iterdir()):
                if p.is_dir():
                    name = p.name
                    # 期望目录名格式: YYYYMMDD_HHMMSS
                    try:
                        evtime = UTCDateTime.strptime(name, "%Y%m%d_%H%M%S")
                    except Exception:
                        # 如果目录名不是标准时间，跳过
                        continue
                    # 如果指定了 start/end，按时间范围过滤目录
                    if args.startT and evtime < args.startT:
                        continue
                    if args.endT and evtime > args.endT:
                        continue
                    available_timekeys.add(name)
        except Exception:
            available_timekeys = set()

        # Total number of events in Catalogue (只保留有本地数据目录的事件)
        nevK = 0
        filtered_cat = []
        for ev in cat:
            try:
                # 获取事件时间（优先 origin.time）
                if hasattr(ev, 'preferred_origin') and ev.preferred_origin() is not None:
                    otime = ev.preferred_origin().time
                elif hasattr(ev, 'origins') and len(ev.origins) > 0:
                    otime = ev.origins[0].time
                else:
                    continue
                timekey = otime.strftime("%Y%m%d_%H%M%S")
                # 只保留在本地目录中存在的数据
                if len(available_timekeys) == 0 or timekey in available_timekeys:
                    # 如果也设置了 start/end，确保事件在范围内（安全检查）
                    if args.startT and otime < args.startT:
                        continue
                    if args.endT and otime > args.endT:
                        continue
                    filtered_cat.append(ev)
            except Exception:
                continue

        cat = filtered_cat
        nevtT = len(cat)
        print("|  Found {0:5d} possible events (with local data) from {1}            |".format(
            nevtT, "local CSV" if args.localevent else "FDSN"))
        ievs = range(0, nevtT)

        # Select order of processing
        if args.reverse:
            ievs = range(0, nevtT)
        else:
            ievs = range(nevtT-1, -1, -1)

        # Read through catalogue
        for iev in ievs:

            # Extract event
            ev = cat[iev]

            # Initialize Split object with station info
            split = Split(sta, zcomp=args.zcomp)

            # Add event to split object
            accept = split.add_event(
                ev,
                gacmin=args.mindist,
                gacmax=args.maxdist,
                phase=args.phase,
                returned=True)

            # Define time stamp
            yr = str(split.meta.time.year).zfill(4)
            jd = str(split.meta.time.julday).zfill(3)
            hr = str(split.meta.time.hour).zfill(2)

            # If event is accepted (data exists)
            if accept:

                # Display Event Info
                nevK = nevK + 1
                if args.reverse:
                    inum = iev + 1
                else:
                    inum = nevtT - iev + 1
                print(" ")
                print("*"*50)
                print("* #{0:d} ({1:d}/{2:d}):  {3:13s} {4}".format(
                    nevK, inum, nevtT, split.meta.time.strftime(
                        "%Y%m%d_%H%M%S"), stkey))
                if args.verb:
                    print("*   Phase: {}".format(args.phase))
                    print("*   Origin Time: " +
                          split.meta.time.strftime("%Y-%m-%d %H:%M:%S"))
                    print(
                        "*   Lat: {0:6.2f};        Lon: {1:7.2f}".format(
                            split.meta.lat, split.meta.lon))
                    print(
                        "*   Dep: {0:6.2f} km;     Mag: {1:3.1f}".format(
                            split.meta.dep, split.meta.mag))
                    print("*   Dist: {0:7.2f} km;".format(split.meta.epi_dist) +
                          "   Epi dist: {0:6.2f} deg\n".format(split.meta.gac) +
                          "*   Baz:  {0:6.2f} deg;".format(split.meta.baz) +
                          "   Az: {0:6.2f} deg".format(split.meta.az))

                # Event Folder
                timekey = split.meta.time.strftime("%Y%m%d_%H%M%S")
                datadir = datapath / timekey
                ZNEfile = datadir / 'ZNE_data.pkl'
                LQTfile = datadir / 'LQT_data.pkl'
                metafile = datadir / 'Meta_data.pkl'
                stafile = datadir / 'Station_data.pkl'
                splitfile = datadir / 'Split_results_auto.pkl'

                # Check if RF data already exist and overwrite has been set
                if datadir.exists():
                    if splitfile.exists():
                        if not args.ovr:
                            continue

                # Check if event data directory exists
                event_data_dir = Path(args.eventdatadir) / timekey
                if not event_data_dir.exists():
                    if args.verb:
                        print(f"* Event data directory not found: {event_data_dir}")
                        print("*"*50)
                    continue

                if args.recalc:
                    if np.sum([file.exists() for file in
                               [ZNEfile, metafile, stafile]]) < 3:
                        continue
                    sta = pickle.load(open(stafile, "rb"))
                    split = Split(sta)
                    meta = pickle.load(open(metafile, "rb"))
                    split.meta = meta
                    dataZNE = pickle.load(open(ZNEfile, "rb"))
                    split.dataZNE = dataZNE

                    # --------------------------------------------------
                    # Rotate from ZNE to LQT (only once)
                    # --------------------------------------------------
                    split.rotate(align='LQT')

                    # 1. 自适应窗口
                    base_t1, base_t2 = get_adaptive_sks_window(split)

                    # 2. 搜索最优窗口 + 滤波
                    best = search_best_window_and_filter(
                        split,
                        base_t1, base_t2,
                        args.filterbands_list,
                        shift_sec=5,
                        step=1.0,
                        snr_comp="Q"
                    )

                    # 3. 记录 SNR
                    split.analysis_cfg["snr"] = best["snr"]
                    split.analysis_cfg["snr_component"] = "Q"


                    # --------------------------------------------------
                    # 检查是否找到有效频段
                    # --------------------------------------------------
                    if best["snr"] is None:
                        if args.verb:
                            print("* No valid SNR found for any filter band")
                            print("*" * 50)
                        continue

                    # --------------------------------------------------
                    # 使用 SNR 最优的滤波结果
                    # --------------------------------------------------
                    split.meta.snrq = best["snr"]
                    split.meta.best_filter_band = (best["fmin"], best["fmax"])

                    if args.verb:
                        print(
                            f"* Selected filter band: "
                            f"{best_band[0]:.2f}-{best_band[1]:.2f} Hz "
                            f"(SNRQ = {best["snr"]:.2f})"
                        )
                    # Save LQT Traces
                    pickle.dump(split.dataLQT, open(LQTfile, "wb"))
                else:
                    # Load data from local event directory
                    has_data = load_local_event_data(
                        split=split,
                        event_dir=event_data_dir,
                        station_info=sta,
                        data_format=args.dataformat,
                        new_sampling_rate=args.new_sampling_rate,
                        verb=args.verb)

                    if not has_data:
                        if args.verb:
                            print("* Failed to load data from local event directory")
                            print("*"*50)
                        continue

                    # --------------------------------------------------
                    # Rotate from ZNE to LQT (only once)
                    # --------------------------------------------------
                    split.rotate(align='LQT')
                    # 1. 自适应窗口
                    base_t1, base_t2 = get_adaptive_sks_window(split)

                    # 2. 搜索最优窗口 + 滤波
                    best = search_best_window_and_filter(
                        split,
                        base_t1, base_t2,
                        args.filterbands_list,
                        shift_sec=5,
                        step=1.0,
                        snr_comp="R"
                    )
                    if best is None or best.get("fmin") is None or best.get("fmax") is None:
                        print("WARNING: No valid filter band selected for this event/station.")
                        best = {
                            "fmin": np.nan,
                            "fmax": np.nan,
                            "snr": np.nan,
                            "result": None
                        }
                    if args.verb:
                        print(
                            f"* Best filter band: "
                            f"{best["fmin"]:.2f}-{best["fmax"]:.2f} Hz "
                            f"(SNRQ = {best["snr"]:.2f})"
                        )

                    # 3. 记录 SNR
                    split.analysis_cfg["snr"] = best["snr"]
                    split.analysis_cfg["snr_component"] = "Q"

                    # Make sure no processing happens for NaNs
                    if np.isnan(best["snr"]):
                        if args.verb:
                            print("* SNR NaN, continuing")
                            print("*"*50)
                        continue
                    
                    # If SNR lower than user-specified threshold, continue
                    if best["snr"] < args.msnr:
                        if args.verb:
                            print(
                                "* SNRQ < {0:.1f}, continuing".format(args.msnr))
                            print("*"*50)
                        continue

                    # --------------------------------------------------
                    # 使用 SNR 最优的滤波结果
                    # --------------------------------------------------
                    split.meta.snrq = best["snr"]
                    split.meta.best_filter_band = (best["fmin"], best["fmax"])

                    # Create Folder if it doesn't exist
                    if not datadir.exists():
                        datadir.mkdir(parents=True)

                    # Save ZNE Traces
                    pickle.dump(split.dataZNE, open(ZNEfile, "wb"))

                    # Save LQT Traces
                    pickle.dump(split.dataLQT, open(LQTfile, "wb"))

                if args.verb:
                    print("* SNRQ: {}".format(split.meta.snrq))
                    print("* SNRT: {}".format(split.meta.snrt))

                if args.calc or args.recalc:
                    dt_late = best["t2"] - best["t1"]
                    if dt_late == 40.0:
                        best["t1"] += 15.0
                    else:
                        # 计算窗口长度（秒）
                        dt_win = best["t2"] - best["t1"]   # float, seconds

                        # 窗口中点（UTCDateTime）
                        t_mid = best["t1"] + 0.5 * dt_win

                        # 新窗口（前 5 秒）
                        best["t1"] = t_mid - 5.0
                    
                    time1 = best["t1"] - split.meta.time
                    time2 = best["t2"] - split.meta.time
                    print("analyze time window: {0:.2f} - {1:.2f} sec".format(
                        time1, time2))
                    #4. 用最优参数做分裂计算
                    split.analyze(
                        t1=best["t1"],
                        t2=best["t2"],
                        apply_filter=True,
                        fmin=best["fmin"],
                        fmax=best["fmax"],
                        verbose=args.verb
                    )

                    # Continue if problem with analysis
                    if split.RC_res.edtt is None or split.SC_res.edtt is None:
                        if args.verb:
                            print("* !!! DOF Error. --> Skipping...")
                            print("*"*50)
                        continue

                    # Determine if Null and Quality of estimate
                    split.is_null(args.snrTlim, verbose=args.verb)
                    split.get_quality(verbose=args.verb)

                # Display results
                if args.verb:
                    split.display_meta()
                    if args.calc or args.recalc:
                        split.display_results()
                        split.display_null_quality()

                # Save event meta data
                pickle.dump(split.meta, open(metafile, "wb"))

                # Save Station Data
                pickle.dump(split.sta, open(stafile, "wb"))

                if args.calc or args.recalc:
                    # Save Split Data
                    file = open(splitfile, "wb")
                    pickle.dump(split.SC_res, file)
                    pickle.dump(split.RC_res, file)
                    pickle.dump(split.null, file)
                    pickle.dump(split.quality, file)
                    pickle.dump(split.meta.best_filter_band, file)
                    file.close()

                    # Initialize diagnostic figure and plot it
                    if args.diagplot:
                        #if (split.null is False) and (split.quality in ["Good", "Fair"]):
                        if (split.null in ["False", "True"]) and (split.quality in ["Good", "Fair", "Poor"]):
                            dplot = DiagPlot(split)
                            dplot.plot_diagnostic(t1=best["t1"], t2=best["t2"], f1=best["fmin"], f2=best["fmax"])
                            fig = plt.figure(dplot.axes[0].number)
                            # 仅在非Null且质量为good或fair时保存图片，否则跳过
                            save_root = Path(args.diagplot) / sta.station
                            save_root.mkdir(parents=True, exist_ok=True)
                            event_time_str = split.meta.time.strftime("%Y%m%d_%H%M%S")
                            save_path = save_root / f"{event_time_str}_{sta.station}.png"
                            # ---------------------------------------------
                            # Annotate selected filter band on figure
                            # ---------------------------------------------
                            fig.savefig(save_path, dpi=300)
                            plt.close(fig)
                        else:
                            continue


if __name__ == "__main__":

    # Run main program
    main()
