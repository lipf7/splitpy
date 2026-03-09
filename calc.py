# Copyright 2019 Pascal Audet & Andrew Schaeffer
#
# This file is part of SplitPy.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# -*- coding: utf-8 -*-
import numpy as np
from obspy.core import Trace, Stream
from numpy.linalg import inv


def split_SilverChan(trQ, trT, baz, t1, t2, maxdt, ddt, dphi):
    phi = np.arange(-90.0, 90.0, dphi)*np.pi/180.
    dtt = np.arange(0., maxdt, ddt)

    M = np.zeros((2, 2, len(phi)))
    M[0, 0, :] = np.cos(phi); M[0, 1, :] = -np.sin(phi)
    M[1, 0, :] = np.sin(phi); M[1, 1, :] = np.cos(phi)

    Ematrix = np.zeros((len(phi), len(dtt)))

    trQ_tmp = trQ.copy().trim(t1, t2)
    trT_tmp = trT.copy().trim(t1, t2)

    trQ_tmp.taper(max_percentage=0.1, type='hann')
    trT_tmp.taper(max_percentage=0.1, type='hann')

    # 预准备波形数据和 FFT 环境
    q_data = trQ_tmp.data
    t_data = trT_tmp.data
    nt = trQ_tmp.stats.npts
    dt_samp = trQ_tmp.stats.delta
    freq = np.fft.fftfreq(nt, d=dt_samp)

    for p in range(len(phi)):
        # 应用旋转矩阵测试快慢方向
        inv_M = inv(np.array(M[:, :, p]))
        FS_test = np.dot(np.array(M[:, :, p]), np.array([q_data, t_data]))

        # 只在前向做一次 FFT 即可处理所有的相移计算
        f0_fft = np.fft.fft(FS_test[0])
        f1_fft = np.fft.fft(FS_test[1])

        for t in range(len(dtt)):
            shift = dtt[t]
            
            # 使用频域相乘直接进行时间平移
            tmpFast_data = np.real(np.fft.ifft(f0_fft * np.exp(2.*np.pi*1j*freq*(-shift/2.))))
            tmpSlow_data = np.real(np.fft.ifft(f1_fft * np.exp(2.*np.pi*1j*freq*(shift/2.))))

            # 旋转回 Q 和 T 坐标系
            corrected_QT = np.dot(inv_M, np.array([tmpFast_data, tmpSlow_data]))

            # 计算横向能量并记录
            Ematrix[p, t] = np.sum(np.square(corrected_QT[1]))

    # 定位能量最小值的索引
    ind = np.where(Ematrix == Ematrix.min())
    ind_phi = ind[0][0]
    ind_dtt = ind[1][0]

    shift = dtt[ind_dtt]
    phiSC_min = phi[ind_phi]*180./np.pi
    phiSC = np.mod((phiSC_min + baz), 180.)

    if phiSC > 90.:
        phiSC -= 180.

    # 基于最佳解最后一次还原成 Trace 实例，消除内层循环损耗
    FS_test = np.dot(np.array(M[:, :, ind_phi]), np.array([q_data, t_data]))
    
    f0_fft_best = np.fft.fft(FS_test[0])
    f1_fft_best = np.fft.fft(FS_test[1])
    
    tmpFast_best = np.real(np.fft.ifft(f0_fft_best * np.exp(2.*np.pi*1j*freq*(-shift/2.))))
    tmpSlow_best = np.real(np.fft.ifft(f1_fft_best * np.exp(2.*np.pi*1j*freq*(shift/2.))))

    corrected_QT_best = np.dot(inv(np.array(M[:, :, ind_phi])), np.array([tmpFast_best, tmpSlow_best]))

    trQ_c = Trace(data=corrected_QT_best[0], header=trQ_tmp.stats)
    trT_c = Trace(data=corrected_QT_best[1], header=trT_tmp.stats)

    trFast = Trace(data=tmpFast_best, header=trT_tmp.stats)
    trSlow = Trace(data=tmpSlow_best, header=trQ_tmp.stats)

    return Ematrix, trQ_c, trT_c, trFast, trSlow, phiSC, shift, phiSC_min


def split_RotCorr(trQ, trT, baz, t1, t2, maxdt, ddt, dphi):
    phi = np.arange(-90.0, 90.0, dphi)*np.pi/180.
    dtt = np.arange(0., maxdt, ddt)

    M = np.zeros((2, 2, len(phi)))
    M[0, 0, :] = np.cos(phi); M[0, 1, :] = -np.sin(phi)
    M[1, 0, :] = np.sin(phi); M[1, 1, :] = np.cos(phi)

    Cmatrix_pos = np.zeros((len(phi), len(dtt)))
    Cmatrix_neg = np.zeros((len(phi), len(dtt)))

    trQ_tmp = trQ.copy().trim(t1, t2)
    trT_tmp = trT.copy().trim(t1, t2)

    trQ_tmp.taper(max_percentage=0.1, type='hann')
    trT_tmp.taper(max_percentage=0.1, type='hann')

    q_data = trQ_tmp.data
    t_data = trT_tmp.data
    nt = trQ_tmp.stats.npts
    dt_samp = trQ_tmp.stats.delta
    freq = np.fft.fftfreq(nt, d=dt_samp)

    for p in range(len(phi)):
        FS_test = np.dot(np.array(M[:, :, p]), np.array([q_data, t_data]))

        # 互相关计算
        ns0 = np.sum(np.square(FS_test[0]))
        ns1 = np.sum(np.square(FS_test[1]))
        norm = np.sqrt(ns0*ns1)

        cor_data = np.fft.ifftshift(np.correlate(FS_test[0], FS_test[1], mode='same')/norm)
        cor_fft = np.fft.fft(cor_data)

        for t in range(len(dtt)):
            shift = dtt[t]
            
            # 使用频域操作一次完成正负平移
            cor_pos = np.real(np.fft.ifft(cor_fft * np.exp(2.*np.pi*1j*freq*(shift))))
            cor_neg = np.real(np.fft.ifft(cor_fft * np.exp(2.*np.pi*1j*freq*(-shift))))
            
            Cmatrix_pos[p, t] = cor_pos[0]
            Cmatrix_neg[p, t] = cor_neg[0]

    if abs(Cmatrix_pos).max() > abs(Cmatrix_neg).max():
        ind = np.where(Cmatrix_pos == max(Cmatrix_pos.max(), Cmatrix_pos.min(), key=abs))
        ind_phi = ind[0][0]; ind_dtt = ind[1][0]
        dtRC = dtt[ind_dtt]
        phiRC_max = phi[ind_phi]*180./np.pi
        phiRC = np.mod((phiRC_max + baz - 90.), 180.)
        Cmap = Cmatrix_pos
        theta = (phiRC_max - 90.)/180.*np.pi
        S = np.sign(max(Cmatrix_pos.max(), Cmatrix_pos.min(), key=abs))
    else:
        ind = np.where(Cmatrix_neg == max(Cmatrix_neg.max(), Cmatrix_neg.min(), key=abs))
        ind_phi = ind[0][0]; ind_dtt = ind[1][0]
        dtRC = dtt[ind_dtt]
        phiRC_max = phi[ind_phi]*180./np.pi
        phiRC = np.mod((phiRC_max + baz), 180.)
        Cmap = Cmatrix_neg
        theta = (phiRC_max)/180.*np.pi
        S = np.sign(max(Cmatrix_neg.max(), Cmatrix_neg.min(), key=abs))

    Cmap = Cmap * (-S)
    shift = dtRC
    theta = theta + np.pi/2.

    if phiRC > 90.:
        phiRC -= 180.

    # 还原最佳状态生成 Trace 对象
    M2 = np.zeros((2, 2))
    M2[0, 0] = np.cos(theta); M2[0, 1] = -np.sin(theta)
    M2[1, 0] = np.sin(theta); M2[1, 1] = np.cos(theta)

    FS_test = np.dot(np.array(M2[:, :]), np.array([q_data, t_data]))
    
    f0_fft_best = np.fft.fft(FS_test[0])
    f1_fft_best = np.fft.fft(FS_test[1])

    tmpFast_best = np.real(np.fft.ifft(f0_fft_best * np.exp(2.*np.pi*1j*freq*(shift/2.))))
    tmpSlow_best = np.real(np.fft.ifft(f1_fft_best * np.exp(2.*np.pi*1j*freq*(-shift/2.))))

    trFast = Trace(data=tmpFast_best, header=trT_tmp.stats)
    trSlow = Trace(data=tmpSlow_best, header=trQ_tmp.stats)

    corrected_QT = np.dot(inv(np.array(M2[:, :])), np.array([tmpFast_best, tmpSlow_best]))

    trQ_c = Trace(data=corrected_QT[0], header=trQ_tmp.stats)
    trT_c = Trace(data=corrected_QT[1], header=trT_tmp.stats)

    return Cmap, trQ_c, trT_c, trFast, trSlow, phiRC, dtRC, phiRC_max

def split_dof(tr):
    """
    Determines the degrees of freedom to calculate the
    confidence region of the misfit function

    Parameters
    ----------
    tr : :class:`~obspy.core.Trace`
        Seismogram 

    Returns
    -------
    dof : float
        Degrees of freedom

    From Walsh, JGR, 2013

    """

    F = np.abs(np.fft.fft(tr.data)[0:int(len(tr.data)/2) + 1])

    E2 = np.sum(F**2)
    E2 -= (F[0]**2 + F[-1]**2)/2.
    E4 = (1./3.)*(F[0]**4 + F[-1]**4)
    for i in range(1, len(F) - 1):
        E4 += (4./3.)*F[i]**4

    dof = int(4.*E2**2/E4 - 2.)

    return dof


def split_errorSC(tr, t1, t2, q, Emat, maxdt, ddt, dphi):
    """
    Calculate error bars based on a F-test and 
    a given confidence interval q

    Parameters
    ----------
    tr : :class:`~obspy.core.Trace`
        Seismogram 
    t1 : :class:`~obspy.core.utcdatetime.UTCDateTime`
        Start time of picking window
    t2 : :class:`~obspy.core.utcdatetime.UTCDateTime`
        End time of picking window
    q : float
        Confidence level
    Emat : :class:`~numpy.ndarray`
        Energy minimization matrix
    maxdt : float
        Maximum delay time considered in grid search (sec)
    ddt : float
        Delay time interval in grid search (sec)
    dphi : float
        Angular interval in grid search (deg)

    Returns
    -------
    err_dtt : float
        Error in dt estimate (sec)
    err_phi : float
        Error in phi estimate (degrees)
    err_contour : :class:`~numpy.ndarray`
        Error contour for plotting

    """

    from scipy import stats

    # Bounds on search
    phi = np.arange(-90.0, 90.0, dphi)*np.pi/180.
    dtt = np.arange(0., maxdt, ddt)

    # Copy trace to avoid overriding
    tr_tmp = tr.copy()
    tr_tmp.trim(t1, t2)

    # Get degrees of freedom
    dof = split_dof(tr_tmp)
    if dof < 3:
        dof = 3
        print(
            "Degrees of freedom < 3. Fixing to DOF = 3, which may " +
            "result in accurate errors")
    n_par = 2

    # Error contour
    vmin = Emat.min()
    vmax = Emat.max()
    err_contour = vmin*(1. + n_par/(dof - n_par) *
                        stats.f.ppf(1. - q, n_par, dof - n_par))

    # Estimate uncertainty (q confidence interval)
    err = np.where(Emat < err_contour)
    
    #if len(err) == 0:
        #return False, False, False
    
    # 修复：检查第一个数组是否为空
    if len(err[0]) == 0 or len(err[1]) == 0:
        # 返回默认误差值，避免后续计算出错
        err_dtt = 0.5 * ddt  # 默认误差为网格间距的一半
        err_phi = 0.5 * dphi  # 默认误差为角度间隔的一半
        return err_dtt, err_phi, err_contour
        
    err_phi = max(
        0.25*(phi[max(err[0])] - phi[min(err[0])])*180./np.pi, 0.25*dphi)
    err_dtt = max(0.25*(dtt[max(err[1])] - dtt[min(err[1])]), 0.25*ddt)

    return err_dtt, err_phi, err_contour


def split_errorRC(tr, t1, t2, q, Emat, maxdt, ddt, dphi):
    """
    Calculates error bars based on a F-test and 
    a given confidence interval q.

    Note
    ----
    This version uses a Fisher transformation for 
    correlation-type misfit.

    Parameters
    ----------
    tr : :class:`~obspy.core.Trace`
        Seismogram 
    t1 : :class:`~obspy.core.utcdatetime.UTCDateTime`
        Start time of picking window
    t2 : :class:`~obspy.core.utcdatetime.UTCDateTime`
        End time of picking window
    q : float
        Confidence level
    Emat : :class:`~numpy.ndarray`
        Energy minimization matrix
    maxdt : float
        Maximum delay time considered in grid search (sec)
    ddt : float
        Delay time interval in grid search (sec)
    dphi : float
        Angular interval in grid search (deg)

    Returns
    -------
    err_dtt : float
        Error in dt estimate (sec)
    err_phi : float
        Error in phi estimate (degrees)
    err_contour : :class:`~numpy.ndarray`
        Error contour for plotting

    """
    from scipy import stats

    phi = np.arange(-90.0, 90.0, dphi)*np.pi/180.
    dtt = np.arange(0., maxdt, ddt)

    # Copy trace to avoid overriding
    tr_tmp = tr.copy()
    tr_tmp.trim(t1, t2)

    # Get degrees of freedom
    dof = split_dof(tr_tmp)
    if dof <= 3:
        dof = 3.01
        print(
            "Degrees of freedom < 3. Fixing to DOF = 3, which may " +
            "result in inaccurate errors")
    n_par = 2

    # Fisher transformation
    vmin = np.arctanh(Emat.min())

    # Error contour
    zrr_contour = vmin + (vmin*np.sign(vmin)*n_par/(dof - n_par) *
                          stats.f.ppf(1. - q, n_par, dof - n_par)) *\
        np.sqrt(1./(dof-3))

    # Back transformation
    err_contour = np.tanh(zrr_contour)

    # Estimate uncertainty (q confidence interval)
    err = np.where(Emat < err_contour)
    err_phi = max(
        0.25*(phi[max(err[0])] - phi[min(err[0])])*180./np.pi, 0.25*dphi)
    err_dtt = max(0.25*(dtt[max(err[1])] - dtt[min(err[1])]), 0.25*ddt)

    return err_dtt, err_phi, err_contour
