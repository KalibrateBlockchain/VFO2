from __future__ import unicode_literals
import IPython 
from IPython import embed 

import random 

import os, sys
import io 
import shutil
import numpy as np
import pandas as pd
import scipy as scp
from scipy.io import wavfile
from scipy.integrate import cumtrapz
from scipy.fftpack import fft
#import lmfit as lmf
import argparse
import logging
import copy
import pwd
import grp
import librosa as lr
import librosa.display
import soundfile as sf
import time
import datetime
import json
import matplotlib.pyplot as plt
from math import pi, sin, sqrt, pow, floor, ceil
#from external.pypevoc.speech.glottal import iaif_ola, lpcc2pole
from pypevoc.speech.glottal import iaif_ola, lpcc2pole
import pylab
from PIL import Image
#from termcolor import colored

#from utils_odes import residual_ode, ode_solver, ode_sys, physical_props
#from utils_odes import foo_main, sys_eigenvals, plot_solution
#from models.vocal_fold.vocal_fold_model_displacement import vdp_coupled, vdp_jacobian
#from solvers.ode_solvers.ode_solver import ode_solver
#from fitter import vfo_fitter, vfo_vocal_fold_estimator
#from vocal_fold_estimator import vocal_fold_estimator
#from solvers.ode_solvers.ode_solver import ode_solver_1
#from models.vocal_fold.adjoint_model_displacement import adjoint_model
#from models.vocal_fold.vocal_fold_model_displacement import (
#    vdp_coupled,
#    vdp_jacobian,
#)
#from solvers.ode_solvers.dae_solver import dae_solver
#from solvers.ode_solvers.ode_solver import ode_solver
#from solvers.optimization import optim_adapt_step, optim_grad_step
from math import floor, ceil
import logging
#import ray
from typing import Callable, List
from scipy.integrate import ode
from assimulo.problem import Implicit_Problem
from assimulo.solvers import IDA

sys.path.append("/home/ec2-user/SageMaker/VFO2/version_1")
from model_paseCNN import * 
from gen_plot import *
from pase.models.frontend import wf_builder
import gen_plot
import importlib
importlib.reload(gen_plot)

MODEL='paseCNN_model'
CUDA = False
BATCH_SIZE = 1
NUM_WORKERS = 1
normalize = 0


def get_model():    
    #if args.MODEL == "spectrogramCNN_model":
    if MODEL == "spectrogramCNN_model":
        return spectrogramCNN_model()
    #if args.MODEL == "paseLSTM_model":
    if MODEL == "paseLSTM_model":
        return paseLSTM_model()

    return paseCNN_model()

def test_epoch(model, dataloader, optimizer, mode):
    pred, true, all_spks = [], [], []
    spk_wise = {}
    all_features = []
    with torch.no_grad():
        losses = []
        for i, (data, labels, audio_type, spks) in enumerate(dataloader):
            if CUDA:
                data=data.cuda().float()
                labels=torch.LongTensor(labels).cuda()
            #if args.normalize:
            if normalize:
                data = normalize_data(data)

            logits = model(data)
            loss1 = F.cross_entropy(logits, labels.unsqueeze(0).unsqueeze(0))
            # logits = torch.argmax(F.softmax(logits, dim=1))
            logits = F.softmax(logits, dim=1)
            
            # pdb.set_trace()
            # curr_pred = [logits[0][1].cpu().detach().numpy()]
            curr_pred = logits[0][1][0].cpu().detach().numpy()
            curr_label = labels.cpu().numpy()
            losses.append(loss1.item())
            pred.extend(curr_pred)
            true.extend(curr_label)
            all_spks.extend(spks)

            for j, spk in enumerate(spks):
                spk = spk.item()
                spk_wise[spk] = spk_wise.get(spk, []) + [(curr_pred[j], curr_label[j])]
            #             print(logits.cpu().detach().numpy().shape, labels.cpu().numpy().shape)
    
    # all_features = np.array(all_features)
    # all_features = np.concatenate(all_features, axis=0)
    all_features = None
    losses = np.array(losses)
    #print("Losses: mean", np.mean(losses))
    #print("Losses =",losses)
    return pred, true, all_spks, spk_wise, all_features

class TestDataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, audio_feature_list, audio_type_list):
        self.audio_feature_list = audio_feature_list
        self.audio_type_list = audio_type_list
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.audio_feature_list)
    
    
    def __getitem__(self, index):
        return self.audio_feature_list[index], 0, self.audio_type_list[index], 0



def optim_adam(
    p: float,
    dp: float,
    m_t: float,
    v_t: float,
    itr: float,
    eta: float = 0.01,
    beta_1: float = 0.9,
    beta_2: float = 0.999,
    eps: float = 1e-8,) -> float:
    """ Perform Adam update.
    Args:
        p: float
            Parameter.
        dp: float
            Gradient.
        m_t: float
            Moving average of gradient.
        v_t: float
            Moving average of gradient squared.
        itr: float
            Iteration.
        eta: float
            Learning rate.
        beta_1: float
            Decay for gradient.
        beta_2: float
            Decay for gradient squared.
        eps: float
            Tolerance.
    Returns:
        p: float
            Updated parameter.
    """
    m_t = beta_1 * m_t + (1 - beta_1) * dp
    v_t = beta_2 * v_t + (1 - beta_2) * (dp * dp)
    m_cap = m_t / (1 - (beta_1 ** itr))  # correct bias
    v_cap = v_t / (1 - (beta_2 ** itr))
    p = p - (eta * m_cap) / (np.sqrt(v_cap) + eps)
    return p


def optim_grad_step(alpha, beta, delta, d_alpha, d_beta, d_delta, stepsize=0.01):
    """ Perform one step of gradient descent for model parameters.
    """
    alpha = alpha - stepsize * d_alpha
    beta = beta - stepsize * d_beta
    delta = delta - stepsize * d_delta
    return alpha, beta, delta


def optim_adapt_step(alpha, beta, delta, d_alpha, d_beta, d_delta, default_step=0.01):
    """ Perform one step of gradient descent for model parameters.
    Stepsize is adaptive.
    """
    stepsize = default_step / np.max([d_alpha, d_beta, d_delta])

    if (alpha - stepsize * d_alpha) > 0 and (alpha - stepsize * d_alpha) < 2:
        alpha = alpha - stepsize * d_alpha

    if (beta - stepsize * d_beta) > 0 and (beta - stepsize * d_beta) < 2:
        beta = beta - stepsize * d_beta

    if (delta - stepsize * d_delta) > 0 and (delta - stepsize * d_delta) < 2:
        delta = delta - stepsize * d_delta

    return alpha, beta, delta

def adjoint_model(
    alpha: float,
    beta: float,
    delta: float,
    X: List[List[float]],
    dX: List[List[float]],
    R: List[float],
    fs: int,
    t0: float,
    tf: float,):
    """ Adjoint model for the 1-d vocal fold displacement model.
    Used to solve the derivatives of right/left vocal fold displacements w.r.t. 
        model parameters (alpha, beta, delta).
    Args:
        alpha: float
            Glottal pressure coupling parameter.
        beta: float
            Mass, damping, stiffness parameter.
        delta: float
            Asymmetry parameter.
        X: List[List[float]]
            Vocal fold displacements [x_r, x_l].
        dX: List[List[fliat]]
            Vocal fold velocity [dx_r, dx_l].
        R: List[float]
            Term c.r.t. the difference between predicted and actual volume velocity flows.
        fs: int
            Sample rate.
        t0: float
            Start time.
        tf: float
            Stop time.
    Returns:
        residual: Callable[[float, List[float], List[float]], np.ndarray]
            Adjoint model.
        jac: Callable[[float, float, List[float], List[float]], np.ndarray]
            Jacobian of the adjoint model.
    """

    def residual(t: float, M: List[float], dM: List[float]) -> np.ndarray:
        """ Defines the adjoint model, which should be in the implicit form:
                0 <-- res = F(t, M, dM)
        Args:
            t: float
                Time.
            M: List[float]
                State variables [L, dL, E, dE].
            dM: List[float]
                Derivatives of state variables [dL, ddL, dE, ddE].
        Returns:
            res: np.ndarray[float], shape (len(M),)
                Residual vector.
        """
        # Convert t to [0, T]
        t = t - t0
        # Convert t(s) to idx(#sample)
        idx = int(round(t * fs) - 1)
        if idx < 0:
            idx = 0
        # print(f't: {t:.4f}    adjoint idx: {idx:d}')

        x = X[idx]
        dx = dX[idx]
        r = R[idx]

        res_1 = dM[1] + (2 * beta * x[0] * dx[0] + 1 - 0.5 * delta) * M[0] + r

        res_2 = beta * M[0] * (1 + x[0] ** 2) - alpha * (M[0] + M[2])

        res_3 = dM[3] + (2 * beta * x[1] * dx[1] + 1 + 0.5 * delta) * M[2] + r

        res_4 = beta * M[2] * (1 + x[1] ** 2) - alpha * (M[0] + M[2])

        res = np.array([res_1, res_2, res_3, res_4])

        return res

    def jac(c: float, t: float, M: List[float], Md: List[float]) -> np.ndarray:
        """ Defines the Jacobian of the adjoint model, which should be in the form:
                J = dF/dM + c*dF/d(dM)
        Args:
            c: float
                Constant.
            t: float
                Time.
            M: List[float]
                State variables [L, dL, E, dE].
            dM: List[float]
                Derivative of state variables [dL, ddL, dE, ddE].
        Returns:
            jacobian: np.ndarray[float], shape (len(M), len(M))
                Jacobian matrix.
        """
        # Convert t to [0, T]
        T = tf - t0
        t = (t - t0) / (tf - t0) * T
        # Convert t(s) to idx(#sample)
        idx = int(round(t * fs) - 1)
        if idx < 0:
            idx = 0

        x = X[idx]
        dx = dX[idx]

        jacobian = np.zeros((len(M), len(M)))

        # jacobian[0, 0] = 2 * beta * x[0] * dx[0] + 1 - 0.5 * delta
        # jacobian[0, 1] = c
        # jacobian[1, 2] = 2 * beta * x[1] * dx[1] + 1 + 0.5 * delta
        # jacobian[1, 3] = c
        # jacobian[2, 0] = beta * (1 + x[0] ** 2) - alpha
        # jacobian[2, 2] = -alpha
        # jacobian[3, 0] = -alpha
        # jacobian[3, 2] = beta * (1 + x[1] ** 2) - alpha

        jacobian[0, 0] = 2 * beta * x[0] * dx[0] + 1 - 0.5 * delta
        jacobian[0, 1] = c

        jacobian[1, 0] = beta * (1 + x[0] ** 2) - alpha
        jacobian[1, 2] = -alpha

        jacobian[2, 2] = 2 * beta * x[1] * dx[1] + 1 + 0.5 * delta
        jacobian[2, 3] = c

        jacobian[3, 0] = -alpha
        jacobian[3, 2] = beta * (1 + x[1] ** 2) - alpha

        return jacobian

    return residual, jac

def vdp_coupled(t: float, Z: List[float], alpha: float, beta: float, delta: float) -> List[float]:
    """ Physical model of the displacement of vocal folds.
    The model is in the explicit form of a pair of coupled van der Pol oscillators:
        dZ = f(Z)
    which include two second order, nonlinear, constant coefficients, inhomogeneous ODEs.
    Args:
        t: float
            Time.
        Z: List[float]
            State variables [u1(t), u2(t), v1(t), v2(t)], u c.r.t right, v c.r.t. left.
        alpha: float
            Glottal pressure coupling parameter.
        beta: float
            Mass, damping, stiffness parameter.
        delta: float
            Asymmetry parameter.
    Returns:
        dZ: List[float]
            Drivatives of state variables [du1, du2, dv1, dv2].
    """
    du1 = Z[1]

    dv1 = Z[3]

    du2 = -beta * (1 + Z[0] ** 2) * Z[1] - (1 - delta / 2) * Z[0] + alpha * (Z[1] + Z[3])

    dv2 = -beta * (1 + Z[2] ** 2) * Z[3] - (1 + delta / 2) * Z[2] + alpha * (Z[1] + Z[3])

    dZ = [du1, du2, dv1, dv2]
    return dZ


def vdp_jacobian(t: float, Z: List[float], alpha: float, beta: float, delta: float) -> List[List[float]]:
    """ Jacobian of the above system of the form:
            J[i, j] = df[i] / dZ[j]
    """
    J = [
        [0, 1, 0, 0],
        [-2 * beta * Z[1] * Z[0] - (1 - delta / 2), -beta * (1 + Z[0] ** 2) + alpha, 0, alpha],
        [0, 0, 0, 1],
        [0, alpha, -2 * beta * Z[3] * Z[2] - (1 + delta / 2), -beta * (1 + Z[2] ** 2) + alpha],
    ]

    return J

def ode_solver(
    model: Callable,
    model_jacobian: Callable,
    model_params: List[float],
    init_state: List[float],
    init_t: float,
    solver: str = "lsoda",
    ixpr: int = 1,
    dt: float = 0.1,
    tmax: float = 1000,) -> np.ndarray:
    """ ODE solver.
    Args:
        model: Callable
            ODE model dy = f(t, y).
        model_jacobian: Callable
            Jacobian of ODE model.
        model_params: List[float]
            Model parameters.
        init_state: List[float]
            Initial model state.
        init_t: float
            Initial simulation time.
        solver: str
            Solver name. Options: vode, dopri5, dop853, lsoda; depends on stiffness and precision.
        ixpr: int
            Whether to generate extra printing at method switches.
        dt: float
            Time step increment.
        tmax: float
            Maximum simulation time.
    Returns:
        sol: np.ndarray[float]
            Solution [time, model states].
    """
    sol = []

    r = ode(model, model_jacobian)

    r.set_f_params(*model_params)
    r.set_jac_params(*model_params)
    r.set_initial_value(init_state, init_t)
    r.set_integrator(solver, with_jacobian=True, ixpr=ixpr)

    while r.successful() and r.t < tmax:
        r.integrate(r.t + dt)
        sol.append([r.t, *list(r.y)])

    return np.array(sol)  # (t, [p, dp]) tangent bundle

def dae_solver(residual,jac,y0,yd0,t0,ncp):
    #print(sys.path)
    #sys.path.insert(0,"/usr/local/lib/python3.7/site-packages")
    #residual, #: Callable,
    #jac,
    #y0, #: List[float],
    #yd0, #: List[float],
    #t0, #: float,
    #tfinal: float = 10.0 [tfinal=0]
    #backward: bool = False [backward=True]
    ncp, #: int = 500, #[ncp=len(wav_samples)]
    #solver: str = "IDA" [solver="IDA"]
    #algvar: Optional[List[bool]] = None [algvar=[0, 1, 0, 1]]
    #suppress_alg: bool = False, [suppress_alg=True]
    #atol: float = 1e-6 [atol=1e-6]
    #rtol: float = 1e-6 [rtol=1e-6]
    #usejac: bool = False, [usejac=True]
    #jac: Optional[Callable] = None [jac=jac]
    #usesens: bool = False [usesens=False]
    #sensmethod: str = "STAGGERED",
    #p0: Optional[List[float]] = None,
    #pbar: Optional[List[float]] = None,
    #suppress_sens: bool = False,
    #display_progress: bool = True [display_progress=True]
    #report_continuously: bool = False [report_continuously=False]
    #verbosity: int = 30 [verbosity=50]
    #name: str = "DAE",) -> List[float]:
    #): 
 
    """ DAE solver.
    Args:
        residual: Callable
            Implicit DAE model.
        y0: List[float]
            Initial model state.
        yd0: List[float]
            Initial model state derivatives.
        t0: float
            Initial simulation time.
        tfinal: float
            Final simulation time.
        backward: bool
            Specifies if the simulation is done in reverse time.
        ncp: int
            Number of communication points (number of returned points).
        solver: str
            Solver name.
        algvar: List[bool]
            Defines which variables are differential and which are algebraic.
            The value True(1.0) indicates a differential variable and the value False(0.0) indicates an algebraic variable.
        suppress_alg: bool
            Indicates that the error-tests are suppressed on algebraic variables.
        atol: float
            Absolute tolerance.
        rtol: float
            Relative tolerance.
        usejac: bool
            Whether to use the user defined jacobian.
        jac: Callable
            Model jacobian.
        usesens: bool
            Aactivates or deactivates the sensitivity calculations.
        sensmethod: str
            Specifies the sensitivity solution method.
            Can be either ‘SIMULTANEOUS’ or ‘STAGGERED’. Default is 'STAGGERED'.
        p0: List[float]
            Parameters for which sensitivites are to be calculated.
        pbar: List[float]
            An array of positive floats equal to the number of parameters. Default absolute values of the parameters.
            Specifies the order of magnitude for the parameters. Useful if IDAS is to estimate tolerances for the sensitivity solution vectors.    
        suppress_sens: bool
            Indicates that the error-tests are suppressed on the sensitivity variables.
        display_progress: bool
            Actives output during the integration in terms of that the current integration is periodically printed to the stdout.
            Report_continuously needs to be activated.
        report_continuously: bool
            Specifies if the solver should report the solution continuously after steps.    
        verbosity: int
            Determines the level of the output.
            QUIET = 50 WHISPER = 40 NORMAL = 30 LOUD = 20 SCREAM = 10.
        name: str
            Model name.
    Returns:
        sol: List[float]
            Solution [time, model states].
    """
    #if usesens is True:  # parameter sensitivity
        #model = Implicit_Problem(residual, y0, yd0, t0, p0=p0)
    #else:
    from assimulo.problem import Implicit_Problem
    from assimulo.solvers import IDA
    model = Implicit_Problem(residual, y0, yd0, t0)

    model.name = "DAE"

    #if algvar is not None:  # differential or algebraic variables
    model.algvar = [0, 1, 0, 1]

    #if usejac is True:  # jacobian
    model.jac = jac

    #if solver == "IDA":  # solver
    #from assimulo.solvers import IDA

    sim = IDA(model)

    sim.backward=True  # backward in time
    sim.suppress_alg=True
    sim.atol=1e-6
    sim.rtol=1e-6
    sim.display_progress=True
    sim.report_continuously=False
    sim.verbosity=50 

    #if usesens is True:  # sensitivity usesens=False

        #sim.sensmethod = sensmethod
        #sim.pbar = np.abs(p0)
        #sim.suppress_sens = suppress_sens

    # Simulation
    # t, y, yd = sim.simulate(tfinal, ncp=(ncp - 1))
    tfinal=0
    ncp_list = np.linspace(t0, tfinal, num=ncp, endpoint=True)
    t, y, yd = sim.simulate(tfinal, ncp=0, ncp_list=ncp_list)

    # Plot
    # plt.figure()
    # plt.subplot(221)
    # plt.plot(t, y[:, 0], 'b.-')
    # plt.legend([r'$\lambda$'])
    # plt.subplot(222)
    # plt.plot(t, y[:, 1], 'r.-')
    # plt.legend([r'$\dot{\lambda}$'])
    # plt.subplot(223)
    # plt.plot(t, y[:, 2], 'k.-')
    # plt.legend([r'$\eta$'])
    # plt.subplot(224)
    # plt.plot(t, y[:, 3], 'm.-')
    # plt.legend([r'$\dot{\eta}$'])

    # plt.figure()
    # plt.subplot(221)
    # plt.plot(t, yd[:, 0], 'b.-')
    # plt.legend([r'$\dot{\lambda}$'])
    # plt.subplot(222)
    # plt.plot(t, yd[:, 1], 'r.-')
    # plt.legend([r'$\ddot{\lambda}$'])
    # plt.subplot(223)
    # plt.plot(t, yd[:, 2], 'k.-')
    # plt.legend([r'$\dot{\eta}$'])
    # plt.subplot(224)
    # plt.plot(t, yd[:, 3], 'm.-')
    # plt.legend([r'$\ddot{\eta}$'])

    # plt.figure()
    # plt.subplot(121)
    # plt.plot(y[:, 0], y[:, 1])
    # plt.xlabel(r'$\lambda$')
    # plt.ylabel(r'$\dot{\lambda}$')
    # plt.subplot(122)
    # plt.plot(y[:, 2], y[:, 3])
    # plt.xlabel(r'$\eta$')
    # plt.ylabel(r'$\dot{\eta}$')

    # plt.figure()
    # plt.subplot(121)
    # plt.plot(yd[:, 0], yd[:, 1])
    # plt.xlabel(r'$\dot{\lambda}$')
    # plt.ylabel(r'$\ddot{\lambda}$')
    # plt.subplot(122)
    # plt.plot(yd[:, 2], yd[:, 3])
    # plt.xlabel(r'$\dot{\eta}$')
    # plt.ylabel(r'$\ddot{\eta}$')

    # plt.figure()
    # plt.subplot(121)
    # plt.plot(y[:, 0], y[:, 2])
    # plt.xlabel(r'$\lambda$')
    # plt.ylabel(r'$\eta$')
    # plt.subplot(122)
    # plt.plot(y[:, 1], y[:, 3])
    # plt.xlabel(r'$\dot{\lambda}$')
    # plt.ylabel(r'$\dot{\eta}$')

    # plt.figure()
    # plt.subplot(121)
    # plt.plot(yd[:, 0], yd[:, 2])
    # plt.xlabel(r'$\dot{\lambda}$')
    # plt.ylabel(r'$\dot{\eta}$')
    # plt.subplot(122)
    # plt.plot(yd[:, 1], yd[:, 3])
    # plt.xlabel(r'$\ddot{\lambda}$')
    # plt.ylabel(r'$\ddot{\eta}$')

    # plt.show()

    sol = [t, y, yd]
    return sol

def sys_eigenvals(l,a,b,d):
    """
    Here we obtain the characteristic equation for the model, and the eigenvalues.
    Helps to study the stability.
    """
    #P = -2.0*a*b*l**2 - 2.0*a*l**3 - 2.0*a*l + 1.0*b**2*l**2 + 2.0*b*l**3 + 2.0*b*l - 0.25*d**2 + 1.0*l**4 + 2.0*l**2 + 1.0
    p4 = 1.0
    p3 = -2.0*a + 2.0*b
    p2 = -2.0*a*b + b**2 + 2.0
    p1 = -2.0*a + 2.0*b
    p0 = -0.25*d**2 + 1.0
    
    P = p4*np.power(l,4) + p3*np.power(l,3) + p2*np.power(l,2) + p1*l + p0
    
    coeff = np.array([p4,p3,p2,p1,p0])
    
    l1,l2,l3,l4 = np.roots(coeff)[0],np.roots(coeff)[1],np.roots(coeff)[2],np.roots(coeff)[3]
    
    r1 , i1 = l1.real , l1.imag
    r2 , i2 = l3.real , l3.imag

    """
    print('#############################################################')
    print('Analyzing the Eigenvalues of the system.')
    print('Real negative is asympt. stable')
    print('Real positive is unstable')
    print('Complex part not zero is a spiral')
    print('Real part equal to zero is a center')
    print('#############################################################')
    print('Real part mode 1:',r1,'Complex part mode 1:',i1)
    print('Real part mode 2:',r2,'Complex part mode 2:',i2)
    """
    
    return P,r1,i1,r2,i2


def vfo_vocal_fold_estimator(glottal_flow,wav_samples,sample_rate):
    print("len(glottal_flow),len(wav_samples)",len(glottal_flow),len(wav_samples))
    alpha=0.30
    beta=0.20
    delta=0.50
    verbose=0
    t_patience = 20
    f_delta=0.0
    cut_off=0.4
    section = -1
    i_delta=delta
    """
    Inputs: wav_samples: audio wavfile
            glottal_flow: numpy array of glottal flow from IAIF
    returns: dictionary best_results:
    ["iteration", "R", "Rk", "alpha", "beta", "delta", "sol", "u0"]
    """

    
    #delta = np.random.random()  # asymmetry parameter
    delta=0.65
    alpha = 0.6 * delta  # if > 0.5 delta, stable-like oscillator
    beta = 0.2
    
    # Set constants
    M = 0.5  # mass, g/cm^2
    B = 100  # damping, dyne s/cm^3
    d = 1.75  # length of vocal folds, cm
    x0 = 0.1  # half glottal width at rest position, cm
    tau = 1e-3  # time delay for surface wave to travel half glottal height, ms
    c = 5000  # air particle velocity, cm/s
    eta = 1.0  # nonlinear factor for energy dissipation at large amplitude
     
    #compute d_1; distance of glottal_flow signal
    i_1=1
    g_1=0
    
    #set timers for analysis of processing resources
    dae_t=0
    ode_t=0
    
    while i_1<(len(glottal_flow)-1):
        i_1=i_1+1
        g_1=g_1+np.abs(glottal_flow[i_1-1]-glottal_flow[i_1])
 

    """CISCO
    sample_rate, wav_samples = wavfile.read(wav_file_path)
    if section == 1:
        wav_samples = wav_samples[floor(len(wav_samples)/2): ceil(len(wav_samples)/2 + sample_rate)]
    """
    
    # NOTE: If you want to plot glottal flow and original waveform together
    # fig = plt.figure()
    # plt.plot(np.linspace(0, len(wav_samples) / sample_rate, len(wav_samples)), wav_samples)
    # plt.plot(np.linspace(0, len(glottal_flow) / sample_rate, len(glottal_flow)), glottal_flow)
    # plt.legend(["speech sample", "glottal flow"])
    # plt.show()

    # Set model initial conditions
    #delta = np.random.random()  # asymmetry parameter
    #alpha = 0.6 * delta  # if > 0.5 delta, stable-like oscillator
    #beta = 0.2

    vdp_init_t = 0.0
    vdp_init_state = [0.0, 0.1, 0.0, 0.1]  # (xr, dxr, xl, dxl), xl=xr=0
    num_tsteps = len(wav_samples)  # total number of time steps
    T = len(wav_samples) / float(sample_rate)  # total time, s

    if verbose==1:
        print("Initial parameters: alpha = ",alpha," beta = ",beta," delta = ",delta)

    # Optimize
    best_results: Dict[str, List[float]] = {  # store best results over iterations
        "iteration": [],  # optimize iter
        "R": [],  # estimation residual @ k
        "Rk": [],  # estimation residual w.r.t L2 norm @ k
        "Rk_s":[],
        "min_distance": [],
        "alpha": [],
        "beta": [],
        "delta": [],
        "sol": [],  # model ouputs
        "u0": [],  # estimated glottal flow
    }
    iteration = 0
    Rk = 1e16
    Rk_best = 1e16
    Rk_s_s_best=100
    patience = 0  # number of patient iterations of no improvement before stopping optimization
    if_adjust = 0
    

    while patience < t_patience: # this was 400 default
        if verbose>-1:
          print("")
          print("")
          print("New solution:")
          print("patience=",patience,", patience count=",t_patience,", iteration=",iteration,", length of clip=",len(wav_samples),", length of glottal signal=",len(glottal_flow))
        if f_delta==1:
            delta=i_delta
        # Solve vocal fold displacement model
        # logger.info("Solving vocal fold displacement model")
        K = B ** 2 / (beta ** 2 * M)
        Ps = (alpha * x0 * np.sqrt(M * K)) / tau
        time_scaling = np.sqrt(K / float(M))  # t -> s

        x_scaling = np.sqrt(eta)
        vdp_params = [alpha, beta, delta]
        ode_t=ode_t-time.process_time() #cacluate ode time
        sol = ode_solver(
            vdp_coupled,
            vdp_jacobian,
            vdp_params,
            vdp_init_state,
            (time_scaling * vdp_init_t),
            solver="lsoda",
            ixpr=0,
            dt=(time_scaling / float(sample_rate)),  # dt -> ds
            tmax=(time_scaling * T),
        )
        ode_t=ode_t+time.process_time() #cacluate ode time
        if len(sol) > len(wav_samples):
            sol = sol[:-1]
        ##assert len(sol) == len(wav_samples)
          ##  if verbose==1:
            ##    print("Inconsistent length: ODE sol;",len(sol),len(wav_samples))

        # Calculate glottal flow
        try:
            assert sol.size > 0
            X = sol[:, [1, 3]]  # vocal fold displacement (right, left), cm
            dX = sol[:, [2, 4]]  # cm/s
            u0 = c * d * (np.sum(X, axis=1) + 2 * x0)  # volume velocity flow, cm^3/s
            u0 = u0 / np.linalg.norm(u0) * np.linalg.norm(glottal_flow)  # normalize
        except AssertionError as e:
            logger.error(e)
            logger.warning("Skip")
            break

        # Estimation residual
        R = u0 - glottal_flow

        # Solve adjoint model
        # logger.info("Solving adjoint model")

        residual, jac = adjoint_model(alpha, beta, delta, X, dX, R, sample_rate, 0, T)
        M_T = [0.0, 0.0, 0.0, 0.0]  # initial states of adjoint model at T
        dM_T = [0.0, -R[-1], 0.0, -R[-1]]  # initial ddL = ddE = -R(T)
        dae_t=dae_t-time.process_time() #cacluate dae time
        adjoint_sol = dae_solver(residual,jac,M_T,dM_T,T,len(wav_samples))
        #try:
            #adjoint_sol = dae_solver(residual,jac,M_T,dM_T,T,len(wav_samples))
            #adjoint_sol = dae_solver(
                #residual,
                #M_T,
                #dM_T,
                #T,
                #tfinal=0,  # simulate (tfinal-->t0)s backward
                #backward=True,
                #ncp=len(wav_samples),
                #solver="IDA",
                #algvar=[0, 1, 0, 1],
                #suppress_alg=True,
                #atol=1e-6,
                #rtol=1e-6,
                #usejac=True,
                #jac=jac,
                #usesens=False,
                #display_progress=True,
                #report_continuously=False,  # NOTE: report_continuously should be False
                #verbosity=50,
            #)
        #except Exception as e:
            #if verbose==1:
                #print("exception: ",e)
            #break
        dae_t=dae_t+time.process_time() #cacluate dae time
        # Compute adjoint lagrange multipliers
        L = adjoint_sol[1][:, 0][::-1]  # reverse time 0 --> T
        E = adjoint_sol[1][:, 2][::-1]
        assert (len(L) == num_tsteps) and (len(E) == num_tsteps), "Size mismatch"
        L = L / np.linalg.norm(L)  # normalize
        E = E / np.linalg.norm(E)

        # Update parameters
        # logger.info("Updating parameters")

        # Record parameters @ current step
        alpha_k = alpha
        beta_k = beta
        delta_k = delta
        Rk = np.sqrt(np.sum(R ** 2))
        #Rs=Rr[int(len(Rr)/5) :]
        #Rk = np.sqrt(np.sum(Rs ** 2)) 
        #R_s=librosa.resample(R, sample_rate, 8000) 
        Rk_s = np.sqrt(np.sum(R[int(len(R)/3):] ** 2))
        if sample_rate>40000:
          Rk_s=Rk_s-.1
        #Rk = np.sqrt((np.sum(R[int(len(R)*.2):] ** 2))/len(R)*22050)
        
        #compute d_1; distance of u0 signal
        i_1=1
        d_1=0
        while i_1<(len(u0)-1):
            i_1=i_1+1
            d_1=d_1+np.abs(u0[i_1-1]-u0[i_1])
  
        t_max_1 = 500

        vdp_init_t_1 = 0.0
        vdp_init_state_1 = [0.0, 0.1, 0.0, 0.1]  # (xr, dxr, xl, dxl), xl=xr=0
        vdp_params_1 = alpha_k, beta_k, delta_k

        # Solve vocal fold displacement model
        ode_t=ode_t-time.process_time() #cacluate ode time
        sol_1 = ode_solver(
            vdp_coupled,
            vdp_jacobian,
            vdp_params_1,
            vdp_init_state_1,
            vdp_init_t_1,
            solver="lsoda",
            ixpr=0,
            dt=1,
            tmax=t_max_1,
        )
        ode_t=ode_t+time.process_time() #cacluate ode time

        # Get steady state
        Sr_1 = sol_1[int(t_max_1 / 2) :, [1, 2]]  # right states, (xr, dxr)
        Sl_1 = sol_1[int(t_max_1 / 2) :, [3, 4]]  # left states, (xl, dxl)

        i=0
        min_distance=100
        max_distance=0
        while i<len(Sr_1):
          distance=sqrt((Sr_1[i,0]*Sr_1[i,0])+(Sr_1[i,1]*Sr_1[i,1]))
          if distance<min_distance:
            min_distance=distance
          if distance>max_distance:
            max_distance=distance
          distance=sqrt((Sl_1[i,0]*Sl_1[i,0])+(Sl_1[i,1]*Sl_1[i,1]))
          if distance<min_distance:
            min_distance=distance
          if distance>max_distance:
            max_distance=distance
          i=i+1

        if min_distance>=0.98 and Rk_s<=1.02 and d_1/g_1>=.5: #We have a solution that meets threshold criteria
          #if min_distance>=1 and Rk_s<=1: #We have a solution that meets threshold criteria
          if Rk_s<Rk_s_s_best: #and it is better than prior ones
            Rk_s_s_best=Rk_s
            alpha_s_best=alpha
            beta_s_best=beta
            delta_s_best=delta
            min_distance_s_best=min_distance
            sol_s_best=sol
            u0_s_best=u0
            iteration_s_best=iteration
            R_s_best=R
            d_1_s_best=d_1

        
        if verbose==1:
            print(f"[{patience:d}:{iteration:d}] Residual: Rk = {Rk:.4f}, Rks (steady state) {Rk_s:.4f}, Rk_best = {Rk_best:.4f} | alpha = {alpha_k:.4f}   "
            f"beta = {beta_k:.4f}   delta = {delta_k:.4f}")
            
            print(f"stiffness K = {K:.4f} dyne/cm^3    subglottal Ps = {Ps:.4f} dyne/cm^2   time_scaling = {time_scaling:.4f}")
            print("len(R)=",len(R)," len(u0)=",len(u0)," len(glottal_flow)=",len(glottal_flow))
            #f_sum=np.sum(np.abs(u0[int(len(R)/5):]))
            #l_sum=np.sum(np.abs(u0[:int(len(R)/5)]))
          
            print("distance u0 = ",d_1,", distance glottal =",g_1,", ratio =",d_1/g_1)
            
               
            plt.figure()
            fig, ax = plt.subplots(figsize=(20,3)) 
            plt.plot(glottal_flow, "k.-")
            plt.plot(u0, "b.-")
            #plt.plot(sol[:, 0], glottal_flow, "k.-")
            #plt.plot(sol[:, 0], u0, "b.-")
            #plt.plot(sol[:, 0], R, "r.-")
            plt.xlabel("t")
            plt.legend(["glottal flow", "estimated glottal flow", "residual"])
            plt.show()     
            
            print("distance from center; min=",min_distance,", max=",max_distance)


            # Plot states
            plt.figure()
            plt.subplot(121)
            plt.plot(Sl_1[:, 0], Sl_1[:, 1], 'k.-')
            #plt.xlabel(r'$\xi_r$')
            #plt.ylabel(r'$\dot{\xi}_r$')
            plt.tight_layout()
            ax = plt.gca()
            ax.axes.xaxis.set_visible(True)
            ax.axes.yaxis.set_visible(True)
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
            ax.axes.xaxis.set_ticklabels([])
            ax.axes.yaxis.set_ticklabels([])
            ax.set_facecolor('none')
            plt.grid(False)

            plt.subplot(122)
            plt.plot(Sr_1[:, 0], Sr_1[:, 1], 'k.-')
            #plt.xlabel(r'$\xi_l$')
            #plt.ylabel(r'$\dot{\xi}_l$')
            plt.figtext(0.5, 0.01, "Residual = {:.3f} , alpha = {:.3f} , beta = {:.3f} , delta = {:.3f}".format(Rk, alpha_k, beta_k, delta_k), wrap=True, horizontalalignment='center', fontsize=12)
            plt.tight_layout()
            ax = plt.gca()
            ax.axes.xaxis.set_visible(True)
            ax.axes.yaxis.set_visible(True)
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
            ax.axes.xaxis.set_ticklabels([])
            ax.axes.yaxis.set_ticklabels([])
            ax.set_facecolor('none')
            plt.grid(False)
            
            plt.show()

            plt.figure()
            plt.subplot(121)
            plt.plot(sol_1[int(t_max_1/2):, 1], sol_1[int(t_max_1/2):, 3], "b.-")
            plt.xlabel(r"$\xi_r$")
            plt.ylabel(r"$\xi_l$")
            plt.subplot(122)
            plt.plot(sol_1[int(t_max_1/2):, 2], sol_1[int(t_max_1/2):, 4], "b.-")
            plt.xlabel(r"$\dot{\xi}_r$")
            plt.ylabel(r"$\dot{\xi}_l$")
            plt.tight_layout()
            plt.show()


          #if ((Rk < Rk_best) and ((d_1/g_1)>(cut_off)) and ((d_1/g_1)<(1/cut_off))):  # has improvement
          #if verbose>-1:
          #print(Rk,Rk_s,Rk_best,d_1,g_1,iteration)            #if ((Rk < Rk_best) and ((d_1/g_1)>cut_off)) or (iteration==0):  # has improvement
          #if (Rk < Rk_best) and ((d_1/g_1)>cut_off):  # has improvement
          #if ((Rk < Rk_best) and ((d_1/g_1)>cut_off)) or (iteration==0):  # has improvement
        if (Rk < Rk_best):  # has improvement
            # Record best
            iteration_best = iteration
            R_best = R
            Rk_best = Rk
            Rk_s_best = Rk_s
            min_distance_best = min_distance
            alpha_best = alpha_k
            beta_best = beta_k
            delta_best = delta_k
            sol_best = sol
            u0_best = u0
            pv_best = np.array([alpha_best, beta_best, delta_best])  # param vector
            d_1_best = d_1

            # Compute gradients
            d_alpha = -np.dot((dX[:num_tsteps, 0] + dX[:num_tsteps, 1]), (L + E))
            d_beta = np.sum(
                L * (1 + np.square(X[:num_tsteps, 0])) * dX[:num_tsteps, 0]
                + E * (1 + np.square(X[:num_tsteps, 1])) * dX[:num_tsteps, 1]
            )
            d_delta = np.sum(0.5 * (X[:num_tsteps, 1] * E - X[:num_tsteps, 0] * L))
            dpv = np.array([d_alpha, d_beta, d_delta])  # param grad vector
            dpv = dpv / np.linalg.norm(dpv)  # normalize
            d_alpha, d_beta, d_delta = dpv

            # Update
            alpha, beta, delta = optim_grad_step(
                alpha, beta, delta, d_alpha, d_beta, d_delta, stepsize=0.1,
                #alpha, beta, delta, d_alpha, d_beta, d_delta, stepsize=((np.random.randn()*0.06)+.001),
            )
            iteration += 1
            # logger.info(
            #     f"[{patience:d}:{iteration:d}] IMPROV: alpha = {alpha:.4f}   beta = {beta:.4f}   "
            #     f"delta = {delta:.4f}"
            # )
        else:  # no improvement
            patience = patience + 1

            # Compute conjugate gradients
            dpv = np.array([d_alpha, d_beta, d_delta])  # param grad vector
            dpv = dpv / np.linalg.norm(dpv)  # normalize
            ov = np.random.randn(len(dpv))  # orthogonal vector
            ov = ov - (np.dot(ov, dpv) / np.dot(dpv, dpv)) * dpv  # orthogonalize
            ov = ov / np.linalg.norm(ov)  # normalize
            d_alpha, d_beta, d_delta = ov

            # Reverse previous update & update in conjugate direction
            alpha, beta, delta = optim_grad_step(
                alpha_best, beta_best, delta_best, d_alpha, d_beta, d_delta, stepsize=0.1,
                #alpha, beta, delta, d_alpha, d_beta, d_delta, stepsize=((np.random.randn()*0.06)+.001),
          )
            # alpha, beta, delta = optim_adapt_step(
            #     alpha_best,
            #     beta_best,
            #     delta_best,
            #     d_alpha,
            #     d_beta,
            #     d_delta,
            #     default_step=0.1,
            # )
            iteration += 1
            # logger.info(
            #     f"[{patience:d}:{iteration:d}] NO IMPROV: alpha = {alpha:.4f}   beta = {beta:.4f}   "
            #     f"delta = {delta:.4f}"
            # )

        while (alpha <= 0.01) or (beta <= 0.01) or (delta <= 0.01):  # if param goes below 0
            if_adjust = 1
            rv = np.random.randn(len(pv_best))  # radius
            rv = rv / np.linalg.norm(rv)  # normalize to 1
            pv = pv_best + 0.01 * rv  # perturb within a 0.01 radius ball
            alpha, beta, delta = pv
        if if_adjust:
            # logger.info(
            #     f"[{patience:d}:{iteration:d}] ADJUST: alpha = {alpha:.4f}   beta = {beta:.4f}   "
            #     f"delta = {delta:.4f}"
            # )
            if_adjust = 0


    if Rk_s_s_best<100:
      R_best=R_s_best
      Rk_best=Rk_s_best
      Rk_s_best=Rk_s_s_best
      min_distance_best=min_distance_s_best
      alpha_best=alpha_s_best
      beta_best=beta_s_best
      delta_best=delta_s_best
      sol_best=sol_s_best
      u0_best=u0_s_best
      iteration_best=iteration_s_best
      d_1_best=d_1_s_best


    best_results["iteration"].append(iteration_best)
    best_results["R"].append(R_best)
    best_results["Rk"].append(Rk_best)
    best_results["Rk_s"].append(Rk_s_best)
    best_results["min_distance"].append(min_distance_best)
    best_results["alpha"].append(alpha_best)
    best_results["beta"].append(beta_best)
    best_results["delta"].append(delta_best)
    best_results["sol"].append(sol_best)
    best_results["u0"].append(u0_best)
    if verbose==1:
      print(f"BEST@{iteration_best:d}: L2 Residual = {Rk_best:.4f} | alpha = {alpha_best:.4f}   "
      f"beta = {beta_best:.4f}   delta = {delta_best:.4f}")

      plt.figure()
      fig, ax = plt.subplots(figsize=(20,3)) 
      plt.plot(sol_best[:, 0], glottal_flow, "k.-")
      plt.plot(sol_best[:, 0], u0_best, "b.-")
      #plt.plot(sol_best[:, 0], R_best, "r.-")
      plt.xlabel("t")
      plt.legend(["glottal flow", "estimated glottal flow", "residual"])
      plt.figure()
      plt.subplot(121)
      plt.plot(sol_best[:, 1], sol_best[:, 3], "b.-")
      plt.xlabel(r"$\xi_r$")
      plt.ylabel(r"$\xi_l$")
      plt.subplot(122)
      plt.plot(sol_best[:, 2], sol_best[:, 4], "b.-")
      plt.xlabel(r"$\dot{\xi}_r$")
      plt.ylabel(r"$\dot{\xi}_l$")
      plt.tight_layout()
      plt.show()

    
    l = np.linspace(-5,5,100)
    p,r1,i1,r2,i2 = sys_eigenvals(l,alpha_best,beta_best,delta_best)
    
    res = {
        'alpha':float(alpha_best),
        'beta':float(beta_best),
        'delta':float(delta_best),
        'Rk':float(Rk_best),
        'Rk_s':float(Rk_s_best),
        'min_distance':float(min_distance_best),
        'distanceRatio':float(d_1_best/g_1),
        'eigenreal1':float(r1),
        'eigenreal2':float(r2),
        'eigensign':int(np.sign(r1*r2)),
        'timestamp': datetime.datetime.now().isoformat(),
        'dae_time': dae_t,
        'ode_time': ode_t,
    }
       # NOTE: If you want to plot glottal flow, estimatted glottal flow and residual

    return res


def load_model(path='/home/ec2-user/SageMaker/VFO2/version_1/FE_e199.ckpt'):
    pase = wf_builder('/home/ec2-user/SageMaker/VFO2/version_1/cfg/frontend/PASE+.cfg').eval()
    pase.load_pretrained(path, load_last=True, verbose=True)
    #pase.cuda() #test
    print("pase.cpu")
    pase.cpu()
    print("pase2.cpu")
    return pase


    #def CWWmain(path, userName, time_code, mode_of_processing):
def CWWmain(fname, mode_of_processing):
    #mode_of_processing=1 # for console
    #mode_of_processing=2 # for production
    
    path_in_fname='/home/ec2-user/SageMaker/VFO2/VFO2/sample_store/'
    lpf=len(path_in_fname)
    path = fname[:lpf]
    userName = fname[lpf:lpf+37]
    time_code=fname[92:104]
       
    t_code=datetime.datetime.strptime(time_code[:10],"%y%m%d%H%M")
    begin_time_range = datetime.datetime.strftime(t_code + datetime.timedelta(seconds=-30),"%y%m%d%H%M")
    end_time_range = datetime.datetime.strftime(t_code + datetime.timedelta(seconds=120),"%y%m%d%H%M")
                                           
    audio_type_list = ['vowel-i', 'vowel-a', 'alphabet-a-z', 'count-1-20', 'cough', 'vowel-u', 'VowelA2', 'VowelAh2', 'VowelAb2']
    feature_list=[]
    audio_types=[]
     
    t_0 = time.process_time() # Here start counting time
    et_0=time.time()
    too_noisy = False
    not_enough_features = True
    no_vowel_i = True
    vocal_fold = False
    vocalCNN_prediction = 0
    color='b'
    version=4.0
    
    res = {
        'alpha':float(0),
        'beta':float(0),
        'delta':float(0),
        'Rk':float(0),
        'Rk_s':float(2),
        'min_distance':float(0.1),
        'distanceRatio':float(0),
        'eigenreal1':float(0),
        'eigenreal2':float(0),
        'eigensign':int(0),
        'timestamp': datetime.datetime.now().isoformat(),
        'dae_time': 0,
        'ode_time': 0,
        'noise':float(0),
        'process_result':'',
        'vocalCNN_prediction':float(0),
        'version':float(version),
    }

    
    pase = load_model()
    
    #path='VFO2/sample_store/'
    #userName='0320CE73-DA15-8DD8-79EC-FA934F3010DD'
    #time_code='21071001'
    
    directory_list=os.listdir(path+userName)

    for file_name in directory_list:
                                         
        audio_type='none'
                                         
        v=file_name.find('VowelAt')
        if v > -1:
            audio_type = 'vowel-i'
            time_stamp = file_name[v+7:v+17]
        
        v=file_name.find('Vowelo')
        if v > -1:
            audio_type = 'vowel-a'
            time_stamp = file_name[v+6:v+16]
            
        v=file_name.find('Vowelu')
        if v > -1:
            audio_type = 'vowel-u'
            time_stamp = file_name[v+6:v+16]
        
        v=file_name.find('cough')
        if v > -1:
            audio_type = 'cough'
            time_stamp = file_name[v+5:v+15]
            
        v=file_name.find('count')
        if v > -1:
            audio_type = 'count-1-20'
            time_stamp = file_name[v+5:v+15]
                                                     
        if ((audio_type != 'none') and
            (time_stamp >= begin_time_range) and 
            (time_stamp <= end_time_range) and
            (
            (os.path.splitext(file_name)[1] == '.wav') or
            (os.path.splitext(file_name)[1] == '.3gp') or
            (os.path.splitext(file_name)[1] == '.mp4') or
            (os.path.splitext(file_name)[1] == '.caf')
            )):
                                         
            #we have a file_name that meets criteria to process

            file_type=os.path.splitext(file_name)[1]
        
            if file_type==".3gp":
                f3gpname=file_name
                file_name=os.path.splitext(file_name)[0]+".wav"
                os.system("ffmpeg -i "+path+f3gpname+" "+path+file_name) #ffmpeg to wav
            
            r_signal, sr = librosa.load(path+userName+file_name, sr=None, mono=True)
            signal = r_signal
            #signal = r_signal / np.linalg.norm(r_signal)
            selected_signal = signal
       
            if audio_type == 'vowel-a' or audio_type == 'vowel-i' or audio_type == 'vowel-u':

                noise_clip = selected_signal[int(sr*.3):int(sr*0.5)]
                mean_noise = np.mean(np.abs(noise_clip))
                #print("mean_noise*10000=",mean_noise*10000)
                if mean_noise*10000>1200.0: 
                    too_noisy = True
                
                
                if audio_type == 'vowel-i':  
                
                    no_vowel_i = False

                    #trim signal
                    abs_signal = np.abs(selected_signal)
                    chunk = int(sr*.02)
                    avg_signal=[]
                    r_sum=sum(abs_signal[:(chunk-1)])
                    for index, value in enumerate(abs_signal[: len(abs_signal)-chunk]):
                        r_sum=r_sum+abs_signal[index+chunk]
                        avg_signal.append(r_sum)
                        r_sum=r_sum-abs_signal[index]
        
                    threshold = max(avg_signal)
                    index=0
                    while avg_signal[index] < (threshold*.6):
                        start_sample = index
                        index=index+1

                    while (avg_signal[index] > (threshold*.2)) and (index<len(avg_signal)-3):
                        end_sample=index
                        index=index+1

                    trim_start=start_sample+int(sr*.1)
                    trim_end=end_sample-int(sr*.1)
                    selected_signal = selected_signal[trim_start:trim_end]
                    r_signal_i=r_signal
                    #fig, ax = plt.subplots(figsize=(20,3))
                    #plt.title('Signal trimmed')
                    #ax.plot(selected_signal)
                    #plt.show()

                slices=[]
                uttMean = np.mean(np.abs(selected_signal))
                uttStd = np.std(selected_signal)
                for sl in range(0, len(selected_signal)-int(sr*1.0), int(sr/10)):
                    slice=[np.mean(np.abs(selected_signal[sl:sl+int(sr*1.0)])),np.std((selected_signal[sl:sl+int(sr*1.0)])),sl]
                    #mSlice=np.mean(np.abs(x[sl:sl+sr]))
                    #sSlice=np.std(np.abs(x[sl:sl+sr]))
                    slices.append(slice)
                    #print("slice sl",slice)
        
                sorted_slices=sorted(slices, key=lambda slice: slice[1])
        
                for sl in sorted_slices: 
                    sliceStart=sl[2]
                    if sl[0]>uttMean*1.5:
                        sliceStart=sl[2]
                        break
            
                sliceEnd=sliceStart + sr
                selected_signal=selected_signal[sliceStart:sliceEnd]
  
          
                if audio_type == 'vowel-i' and too_noisy == False: 
            
                    g_order=2 * int(np.round(sr / 4000))
                    t_order=2 * int(np.round(sr / 2000))+4
                    gl_audio, dg, vt, gf = iaif_ola(selected_signal, Fs=sr, tract_order=t_order, glottal_order=g_order)
                    gl_audio = gl_audio[int(sr*0.1):len(gl_audio)-int(sr*0.1)]
                    selected_signal=selected_signal[int(sr*0.1):len(selected_signal)-int(sr*0.1)]
                    selected_signal = selected_signal / np.linalg.norm(selected_signal)
                    gl_audio = gl_audio / np.linalg.norm(gl_audio)        
                    res=vfo_vocal_fold_estimator(gl_audio,selected_signal,sr)
                    r_signal_i=r_signal
                    sr_i=sr
                    selected_signal_i=selected_signal
                    start_i=sliceStart
                    end_i=sliceEnd

                
            #extract pase features 
            a_signal = librosa.resample(selected_signal, sr, 8000)
            #a_signal=selected_signal
            a_signal = torch.Tensor(a_signal)
            a_signal = torch.unsqueeze(a_signal, 0)
            a_signal = torch.unsqueeze(a_signal, 0)
            #y = pase(a_signal.cuda())
            y = pase(a_signal.cpu())        
            feature = y.detach().cpu().numpy()
            #print("feature", feature)
            
            #collect for TestDataset
            feature_list.append(feature)
            audio_types.append(audio_type)
        
    if len(feature_list)>2:
        not_enough_features = False
    
    
    if not_enough_features == True:
        #not enough features
        print("not enough features")
        
    elif too_noisy == True:
        #too noisy           
        t_1 = time.process_time() # Here end counting time
        et_1=time.time()
        res.update({'processingTime':float((et_1-et_0)/60)})
        res.update({'cpuTime':float(t_1-t_0)})
        res.update({'noise':float(mean_noise)})
        res.update({'process_result':'failed; too noisy'})

        #plt.subplots_adjust(hspace = -1.0)
        fig = plt.figure(figsize=(8, 18))
        ax1= fig.add_subplot(9,9,1,frameon=False)
        ax1.axis('off') 
        ax1.plot([0,1,0,1,0],[0,1,1,0,1], color,linewidth=3)
        ax2= fig.add_subplot(9,9,10,frameon=False)
        ax2.axis('off')
        ax2.plot([0,1,0,1,0],[0,1,1,0,1], color, linewidth=3)
        ax3= fig.add_subplot(3,2,3,frameon=True)
        ax4= fig.add_subplot(3,2,4,frameon=True)
        ax5= fig.add_subplot(6,1,5,frameon=False)
        ax3.plot([0,1,0,1,0],[0,1,1,0,1], color,linewidth=6)
        ax3.axes.yaxis.set_ticks([])
        ax3.set_ylabel('Left Vocal Fold, λ = {:.9f}'.format(res['eigenreal1']), fontsize=10)
        ax3.yaxis.label.set_color(color)
        ax3.xaxis.label.set_color(color)
        ax3.axes.xaxis.set_ticks([])
        ax3.set_xlabel("Too noisy to process", wrap=True, fontsize=10)
        ax4.plot([0,1,0,1,0],[0,1,1,0,1], color,linewidth=6)
        ax4.axes.yaxis.set_ticks([])
        ax4.set_ylabel('Right  Vocal Fold, λ = {:.9f}'.format(res['eigenreal2']), fontsize=10)
        ax4.xaxis.label.set_color(color)
        ax4.yaxis.label.set_color(color)
        ax4.axes.xaxis.set_ticks([])
        ax4.set_xlabel("{} \nNoise = {:.2f} (< 1.00)".format(res['timestamp'],res['noise']*10000), wrap=True, fontsize=10)
        ax5.axes.yaxis.set_ticks([])
        ax5.axes.xaxis.set_ticks([])
        ax5.plot(r_signal_i, color, linewidth=0.1,markersize=0.1)
        #ax5.axvspan(trim_start+start_i, trim_start+end_i, facecolor='#91CC29')
        #ax5.set_xlabel("Vocal Clip Duration = {:.2f} seconds (1 second) \nProcessing time = {:.2f} minutes".format((len(gl_audio)/sr_i),((et_1-et_0)/60)),fontsize=10)
        ax5.set_xlabel("VocalCNN ={:.2f}".format(vocalCNN_prediction),fontsize=10)
        ax5.xaxis.label.set_color(color)
        fname=path+userName+"plot_"+time_code+".png"
        results_name=path+userName+"results_"+time_code+".json"
        print("fname=",fname,results_name)
        plt.savefig(fname, bbox_inches='tight',pad_inches = 0.05, transparent=True, edgecolor='none')
        results_file = open(results_name, "w")
        json.dump(res, results_file)
        results_file.close()
        
    elif no_vowel_i == True:
        #no vowel_i
        print("no vowel_i")
    
    else:
        #succeeded
        #perform CNN classification
        test_dataset = TestDataset(feature_list, audio_types)
        
        class_model = get_model()
        #class_model.cpu() #test
        class_model.load_state_dict(torch.load('/home/ec2-user/SageMaker/VFO2/version_1/models/trained_pase_model_model0.pth',map_location=torch.device('cpu'))) 
        #if CUDA:
        #    class_model.cuda()
        test_loader = dataloader.DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        prediction0, true, all_spks, spk_wise, all_features = test_epoch(class_model, test_loader, None, None)

        class_model = get_model()
        class_model.load_state_dict(torch.load('/home/ec2-user/SageMaker/VFO2/version_1/models/trained_pase_model_model0.pth',map_location=torch.device('cpu'))) 
        #if CUDA:
        #    class_model.cuda()
        test_loader = dataloader.DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        prediction1, true, all_spks, spk_wise, all_features = test_epoch(class_model, test_loader, None, None)
        
        vocalCNN_prediction=np.mean([np.mean(prediction0),np.mean(prediction1)])
        print("vocalCNN_prediction =",vocalCNN_prediction,prediction0,prediction1)
        t_max = 500
        vdp_init_t = 0.0
        vdp_init_state = [0.0, 0.1, 0.0, 0.1]  # (xr, dxr, xl, dxl), xl=xr=0
        vdp_params = [res['alpha'], res['beta'], res['delta']]
    
        # Solve vocal fold displacement model
        sol = ode_solver(
            vdp_coupled,
            vdp_jacobian,
            vdp_params,
            vdp_init_state,
            vdp_init_t,
            solver="lsoda",
            ixpr=0,
            dt=1,
            tmax=t_max,
            )

        # Get steady state
        Sr = sol[int(t_max / 2) :, [1, 2]]  # right states, (xr, dxr)
        Sl = sol[int(t_max / 2) :, [3, 4]]  # left states, (xl, dxl)
        min_distance=100
        max_distance=0
        i=0
        while i<len(Sr):
            distance=sqrt((Sr[i,0]*Sr[i,0])+(Sr[i,1]*Sr[i,1]))
            if distance<min_distance:
                min_distance=distance
            if distance>max_distance:
                max_distance=distance
            distance=sqrt((Sl[i,0]*Sl[i,0])+(Sl[i,1]*Sl[i,1]))
            if distance<min_distance:
                min_distance=distance
            if distance>max_distance:
                max_distance=distance
            i=i+1
            
        t_1 = time.process_time() # Here end counting time
        et_1=time.time()
        res.update({'processingTime':float((et_1-et_0)/60)})
        res.update({'cpuTime':float(t_1-t_0)})
        res.update({'noise':float(mean_noise)})
        res.update({'process_result':'success'})
        res.update({'vocalCNN_prediction':float(vocalCNN_prediction)})
        res.update({'version':float(version)})

        #plt.subplots_adjust(hspace = -1.0)
        fig = plt.figure(figsize=(8, 18))
        ax1= fig.add_subplot(9,9,1,frameon=False)
        ax1.axis('off') 
        ax2= fig.add_subplot(9,9,10,frameon=False)
        ax2.axis('off')
        ax3= fig.add_subplot(3,2,3,frameon=True)
        ax4= fig.add_subplot(3,2,4,frameon=True)
        ax5= fig.add_subplot(6,1,5,frameon=False)
        ax1.plot(Sl[:, 0], Sl[:, 1], color, linewidth=0.5,markersize=0.5)
        ax2.plot(Sr[:, 0], Sr[:, 1], color, linewidth=0.5,markersize=0.5)
        ax3.plot(Sl[:, 0], Sl[:, 1], color)
        ax3.axes.yaxis.set_ticks([])
        ax3.set_ylabel('Left Vocal Fold, λ = {:.9f}'.format(res['eigenreal1']), fontsize=10)
        ax3.yaxis.label.set_color(color)
        ax3.xaxis.label.set_color(color)
        ax3.axes.xaxis.set_ticks([])
        ax3.set_xlabel("α = {:.3f} , β = {:.3f} , δ = {:.3f} \nFit = {:.2f} (< 1.00), Min Distance = {:.2f} (>1.0)".format(res['alpha'], res['beta'], res['delta'],res['Rk_s'],res['min_distance']), wrap=True, fontsize=10)
        ax4.axes.yaxis.set_ticks([])
        ax4.set_ylabel('Right  Vocal Fold, λ = {:.9f}'.format(res['eigenreal2']), fontsize=10)
        ax4.xaxis.label.set_color(color)
        ax4.yaxis.label.set_color(color)
        ax4.axes.xaxis.set_ticks([])
        ax4.plot(Sr[:, 0], Sr[:, 1], color)
        ax4.set_xlabel("{} \nNoise = {:.2f} (< 1.00)".format(res['timestamp'],res['noise']*10000), wrap=True, fontsize=10)
        ax5.axes.yaxis.set_ticks([])
        ax5.axes.xaxis.set_ticks([])
        ax5.plot(r_signal_i, color, linewidth=0.1,markersize=0.1)
        ax5.axvspan(trim_start+start_i, trim_start+end_i, facecolor='#91CC29')
        ax5.set_xlabel("Vocal Clip Duration = {:.2f} seconds (1 second) \nProcessing time = {:.2f} minutes".format((len(gl_audio)/sr_i),((et_1-et_0)/60)),fontsize=10)
        ax5.set_xlabel("VocalCNN ={:.2f}".format(vocalCNN_prediction),fontsize=10)
        ax5.xaxis.label.set_color(color)
        fname=path+userName+"plot_"+time_code+".png"
        results_name=path+userName+"results_"+time_code+".json"
        plt.savefig(fname, bbox_inches='tight',pad_inches = 0.05, transparent=True, edgecolor='none')
        results_file = open(results_name, "w")
        json.dump(res, results_file)
        results_file.close()
    
    return
    

 
  
if __name__ == '__main__':
    CWWmain('/home/ec2-user/SageMaker/VFO2/VFO2/sample_store/493183CB-1934-3BBD-7E38-D058849E5421/VowelAt220124142952.caf',1)

