import matplotlib.pyplot as plt
import numpy as np
import random
from warnings import filterwarnings
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline
from constants import *
from time import time
from BaryInterp import bary_interpolation


filterwarnings("ignore")

def glue_cEFT(e, p, mu, a):
    if a == 1:
        cEFT_eos = np.genfromtxt("cEFT_soft.dat")
    else:
        cEFT_eos = np.genfromtxt("cEFT_hard.dat")
    p = np.append(cEFT_eos[:, 0], p)
    e = np.append(cEFT_eos[:,1], e)
    mu = np.append(cEFT_eos[:,2], mu)
    return e, p, mu

def glue_pQCD(e, p, mu):
    mu_i = np.linspace(2601, 3000, num = 10)
    for x in np.linspace(1,4,10000):
        p_i = 3/(4*np.pi**2*hbar3_c3)*(mu_i*MeV/3)**4 * (0.9008-(0.5034*x**(-0.3553))/(mu_i/1000-1.452*x**(-0.9101)))/MeV_fm3_to_pa
        if np.abs(p_i[0] - p[-1])<1:
            e = np.append(e, np.linspace(e[-1], e[-1]+30000, num = 10))
            mu = np.append(mu, mu_i)
            p = np.append(p, p_i)
            return e, p, mu
    return np.array([0]), np.array([0]), np.array([0])


def tov(y, r):
    P, m = y
    eden = e(P)
    dPdr = -G * (eden + P/c**2) * (m + 4.0 * np.pi * r ** 3 * P/c**2) / (r * (r - 2.0 * G * m/c**2))
    dmdr = 4.0 * np.pi * r ** 2 * eden


    return [dPdr, dmdr]

def dedp(r, R_dep):
    e_R, p_R, m_R = R_dep

    p_R = p_R(r)
    dp = p_R * 0.005

    el_3 = e(p_R - 3 * dp)
    el_2 = e(p_R - 2 * dp)
    el_1 = e(p_R - 1 * dp)
    er_3 = e(p_R + 3 * dp)
    er_2 = e(p_R + 2 * dp)
    er_1 = e(p_R + 1 * dp)

    
    de_dp = (-1 / 60 * el_3 + 3 / 20 * el_2 - 3 / 4 * el_1 + 3 / 4 * er_1 - 3 / 20 * er_2 + 1 / 60 * er_3) / dp

    return de_dp

def love(param, r, R_dep):
    beta, H = param
    e_R, p_R, m_R = R_dep

    if bool(np.isnan(p_R(r))) is True:
        return [100000, 100000]

    try:
      dummy = p_R(r)
    except ValueError:
      return [100000, 100000]

    de_dp = dedp(r, R_dep)

    dbetadr = H * (-2 * pi * G / c ** 2 * (
        5 * e_R(r) + 9 * p_R(r) / c ** 2 + de_dp * c ** 2 * (e_R(r) + p_R(r) / c ** 2)) \
                   + 3 / r ** 2 \
                   + 2 * (1 - 2 * m_R(r) / r * km_to_mSun) ** (-1) * (
                       m_R(r) / r ** 2 * km_to_mSun + G / c ** 4 * 4 * pi * r * p_R(r)) ** 2) \
              + beta / r * (
                  -1 + m_R(r) / r * km_to_mSun + 2 * pi * r ** 2 * G / c ** 2 * (e_R(r) - p_R(r) / c ** 2))
    dbetadr *= 2 * (1 - 2 * m_R(r) / r * km_to_mSun) ** (-1)

    dHdr = beta
    return [dbetadr, dHdr]



def solve_tov(c_dens):

    c_dens *= MeV_fm3_to_pa_cgs / c ** 2
    r = np.linspace(1, 2e6, 1000)
    P = p(c_dens)
    eden = e(P)
    m = 4.0 * np.pi * r[0] ** 3 * eden

    psol = odeint(tov, [P, m], r, rtol=0.00001)
    p_R, m_R = psol[:,0], psol[:,1]


    diff = (m_R[1:] - m_R[:-1])/m_R[1:]
    ind = -1

    for i, dm in enumerate(diff):
      if dm < 10e-12 and m_R[i] != 0:
        ind = i
        break


    M = m_R[ind - 1]
    R = r[ind - 1]

    r   = r[:ind]
    p_R = p_R[:ind]
    m_R = m_R[:ind]
    e_R = e(p_R)
    

    e_R = interp1d(r, e_R, kind='cubic', bounds_error=False)
    p_R = interp1d(r, p_R, kind='cubic', bounds_error=False)
    m_R = interp1d(r, m_R, kind='cubic', bounds_error=False)

    beta0 = 2 * r[0]
    H0 = r[0] ** 2

    solution = odeint(love, [beta0, H0], r, args=([e_R, p_R, m_R],), rtol=0.1)
    beta = solution[-1, 0]
    H = solution[-1, 1]

    y = R * beta / H

    C = compactness = M / R * km_to_mSun

    k2 = 8 / 5 * C ** 5 * (1 - 2 * C) ** 2 * (2 + 2 * C * (y - 1) - y) * (
          2 * C * (6 - 3 * y + 3 * C * (5 * y - 8)) + 4 * C ** 3 * (
            13 - 11 * y + C * (3 * y - 2) + 2 * C ** 2 * (1 + y)) + 3 * (1 - 2 * C) ** 2 * (2 - y + 2 * C * (y - 1)) * (
            np.log(1 - 2 * C))) ** (-1)

    # Tidal deformability:
    L = 2/3 * k2 * C**(-5)

    return R / 1e5, M / Msun, L


def polytrope_interpolation(number_of_polytropes):
    while 1:

        a = random.randint(1,2)

        if a == 1:
            mu_0, p_0, n_0, e_0, c_0 = 965.6988636, 2.163, 0.16 * 1.1, 167.8, 0.039525
        else:
            mu_0, p_0, n_0, e_0, c_0 = 977.5113636, 3.542, 0.16 * 1.1, 168.5, 0.053470

        pQCD_limit = random.uniform(2579.5,4242)
        mu = np.array([])
        p, n, e, c_s, g_arr = np.array([p_0]), np.array([n_0]), np.array([e_0]), np.array([c_0]), np.array([])
        mu_matching, n_matching, p_matching, e_matching = np.array([]), np.array([]), np.array([]), np.array([])


        mu_matching = np.append(mu_matching, mu_0)
        j = 1
        while j < number_of_polytropes:
            mu_matching = np.append(mu_matching, random.uniform(mu_0, 2600))
            j+=1

        mu_matching = np.append(mu_matching, 2600)
        mu_matching = np.sort(mu_matching)

        j = 0
        while j <= number_of_polytropes:

            if j == 0:
                n_matching = np.append(n_matching, n_0)
                e_matching = np.append(e_matching, e_0)
                p_matching = np.append(p_matching, p_0)
                mu_i = np.linspace(mu_matching[j], mu_matching[j+1], 50)


            elif j == number_of_polytropes-1:

                n_matching = np.append(n_matching, n[-1])
                e_matching = np.append(e_matching, e[-1])
                p_matching = np.append(p_matching, p[-1])
                mu_i = np.linspace(mu_matching[j], 2600, 50)

            elif j==number_of_polytropes:
                n_matching = np.append(n_matching, 40 * 0.16)
                g_i = random.uniform(0, 10)
                k_i = p_matching[-1] * n_matching[-1] ** (-g_i)
                e_matching = np.append(e_matching, ((mu_i * n_matching[-1] + g_i * e_matching[-1] - p_matching[-1] - e_matching[-1]) * p_i / (mu_i * (g_i - 1) * n_matching[-1] - g_i * e_matching[-1] + p_matching[-1] + e_matching[-1]))[-1])
                p_matching = np.append(p_matching, p[-1])
                break
            else:
                n_matching = np.append(n_matching, n[-1])
                e_matching = np.append(e_matching, e[-1])
                p_matching = np.append(p_matching, p[-1])
                mu_i = np.linspace(mu_matching[j], mu_matching[j+1], 50)

            mu = np.append(mu, mu_i)

            g_i = random.uniform(0, 10)
            g_arr = np.append(g_arr, g_i)
            k_i = p_matching[-1] * n_matching[-1] ** (-g_i)

            p_i = k_i * (n_matching[-1]**(g_i - 1) + (g_i - 1)/(k_i * g_i)*(mu_i-mu_matching[j]))**((g_i)/(g_i-1))
            p = np.append(p, p_i)

            e_i = (mu_i * n_matching[-1] + g_i * e_matching[-1] - p_matching[-1] - e_matching[-1]) * p_i / (mu_i * (g_i - 1) * n_matching[-1] - g_i * e_matching[-1] + p_matching[-1] + e_matching[-1])
            n_i = (g_i * n_matching[-1] * p_i) / (mu_i * (g_i - 1) * n_matching[-1] - g_i * e_matching[-1] + p_matching[-1] + e_matching[-1])

            e = np.append(e, e_i)
            n = np.append(n, n_i)

            c_i = g_i/(e_i/p_i + 1)
            c_s = np.append(c_s, c_i)

            j+=1



        if p[-1] > 2579.5 and p[-1] < 4242 and np.max(c_s) < 1 and e[-1]<24000:

            e, p, mu = glue_cEFT(e,p,mu,a)
            e, p, mu = glue_pQCD(e,p,mu)
            if e.all() == [0]:
                continue
            e_cgs = e * MeV_fm3_to_pa_cgs / c**2
            p_cgs = p * MeV_fm3_to_pa_cgs

            def e_fun(x):
                interp = interp1d(p_cgs, e_cgs, bounds_error=False, fill_value=(e_cgs[0], e_cgs[-1]))
                return interp.__call__(x)
            def p_fun(x):
                interp = interp1d(e_cgs, p_cgs, bounds_error=False, fill_value=(p_cgs[0], p_cgs[-1]))
                return interp.__call__(x)
            
            return e_fun, p_fun, mu, mu_matching, p_matching, e, p

# Prepare plots
fig1, axs1 = plt.subplots(1,2)
fig2, axs2 = plt.subplots(1,2)
fig3 = plt.figure(3)


#Compute EoSs
k = 0
number_of_EOS = 1
poly_passable_EoS = 0
pchip_passable_EoS = 0
poly_gamma_small_arr = np.array([])
poly_gamma_max_arr = np.array([])
pchip_gamma_small_arr = np.array([])
pchip_gamma_max_arr = np.array([])

while k < number_of_EOS:


    if k <= number_of_EOS/2:
        number_of_polytropes = random.randint(2,4)
        e, p, mu, mu_matching, p_matching, e_list, p_list = polytrope_interpolation(number_of_polytropes)
    else:
        e, p, n, n_matching, p_matching, e_list, p_list = bary_interpolation()
    # mu = np.linspace(1,3000,1000)
    # plt.plot(mu, p_list[1:])
    # plt.plot(mu_matching, p_matching, "o")
    # print(e(p(mu)))
    # plt.loglog(e_list, p_list, "g")

    m_arr = np.array([])
    R_arr = np.array([])
    L_arr = np.array([])
    central_e_arr = np.logspace(-0,3.7,100)

    for dens_c in central_e_arr:
        R, M, L = solve_tov(dens_c)
        m_arr = np.append(m_arr, M)
        R_arr = np.append(R_arr, R)
        L_arr = np.append(L_arr, L)

    L = interp1d(m_arr, L_arr, bounds_error = False)

    # Create plots:
    
    #Mass and tidal deformation constraints:
    if k <= number_of_EOS/2:
        fig1
        if np.max(m_arr) <= 2.01:
            axs1[1].plot(R_arr, m_arr, "c", zorder = 2)
            axs1[0].plot(e_list, p_list, "c", zorder = 2)
        elif L(1.4)<70 or L(1.4)>580:
            axs1[1].plot(R_arr, m_arr, "m", zorder = 1)
            axs1[0].plot(e_list, p_list, "m", zorder = 1)
        else:
            axs1[1].plot(R_arr, m_arr, "g", zorder = 3)
            axs1[0].plot(e_list, p_list, "g", zorder = 3)
            fig3
            plt.loglog(e_list, p_list, "b", zorder = 2)
            poly_passable_EoS += 1

            #Compute polytropic index:
            # print(np.log(e_list))
            # print(np.log(p_list))
            # gamma = InterpolatedUnivariateSpline(np.log(e_list), np.log(p_list))
            # gamma = gamma.derivative()
            gamma = np.diff(np.log(p_list))/np.diff(np.log(e_list))

            idx_1 = (np.abs(m_arr - 1.4)).argmin()
            idx_2 = (np.abs(e_list - central_e_arr[idx_1])).argmin()
            if bool(np.isnan(gamma[idx_2])) is False:
                poly_gamma_small_arr = np.append(poly_gamma_small_arr, gamma[idx_2])

            idx_1 = np.where(m_arr==np.max(m_arr))[0][0]
            idx_2 = (np.abs(e_list - central_e_arr[idx_1])).argmin()
            if bool(np.isnan(gamma[idx_2])) is False:
                poly_gamma_max_arr = np.append(poly_gamma_max_arr, gamma[idx_2])



    else:
        fig2
        if np.max(m_arr) <= 2.01:
            axs2[1].plot(R_arr, m_arr, "c", zorder = 2)
            axs2[0].plot(e_list, p_list[1:], "c", zorder = 2)
        elif L(1.4)<70 or L(1.4)>580:
            if np.max(m_arr) < 100:
                axs2[1].plot(R_arr, m_arr, "m", zorder = 1)
            axs2[0].plot(e_list, p_list[1:], "m", zorder = 1)
        else:
            if np.max(m_arr) < 100:
                axs2[1].plot(R_arr, m_arr, "g", zorder = 3)
            axs2[0].plot(e_list, p_list[1:], "g", zorder = 3)
            fig3
            plt.loglog(e_list, p_list[1:], "r", zorder = 1)
            pchip_passable_EoS += 1

            #Compute polytropic index:
            # print(np.log(e_list))
            # print(np.log(p_list))
            # gamma = InterpolatedUnivariateSpline(np.log(e_list), np.log(p_list))
            # gamma = gamma.derivative()
            gamma = np.diff(np.log(p_list[1:]))/np.diff(np.log(e_list))

            idx_1 = (np.abs(m_arr - 1.4)).argmin()
            idx_2 = (np.abs(e_list - central_e_arr[idx_1])).argmin()
            if bool(np.isnan(gamma[idx_2])) is False:
                pchip_gamma_small_arr = np.append(pchip_gamma_small_arr, gamma[idx_2])

            idx_1 = np.where(m_arr==np.max(m_arr))[0][0]
            idx_2 = (np.abs(e_list - central_e_arr[idx_1])).argmin()
            if bool(np.isnan(gamma[idx_2])) is False:
                pchip_gamma_max_arr = np.append(pchip_gamma_max_arr, gamma[idx_2])
            


    k += 1
    print(k)


#Print number of passable EoSs:
print(f"Polytrope: Number of passable EoS: {poly_passable_EoS}")
print(f"Pchip: Number of passable EoS: {pchip_passable_EoS}")
if poly_passable_EoS>0 and pchip_passable_EoS>0:
    print(f"Polytrope: 1.4M star adiabatic index in range {np.min(poly_gamma_small_arr), np.max(poly_gamma_small_arr)}, with average {np.average(poly_gamma_small_arr)}")
    print(f"Polytrope: Max mass star adiabatic index in range {np.min(poly_gamma_max_arr), np.max(poly_gamma_max_arr)}, with average {np.average(poly_gamma_max_arr)}")
    print(f"Pchip: 1.4M star adiabatic index in range {np.min(pchip_gamma_small_arr), np.max(pchip_gamma_small_arr)}, with average {np.average(pchip_gamma_small_arr)}")
    print(f"Pchip: Max mass star adiabatic index in range {np.min(pchip_gamma_max_arr), np.max(pchip_gamma_max_arr)}, with average {np.average(pchip_gamma_max_arr)}")


#Plotting
axs1[0].set_yscale("log")
axs1[0].set_xscale("log")
axs1[0].set_xlabel(r"Energy density [MeV/fm$^3$]")
axs1[0].set_ylabel(r"Pressure [MeV/fm$^3$]")
axs1[1].set_ylabel(r'${\rm M~[M_\odot]}$')
axs1[1].set_xlabel(r'${\rm R~[km]}$')
axs1[0].grid(which = "both", ls = ":", lw = 0.3)
axs1[1].grid(which = "both", ls = ":", lw = 0.3)
axs2[0].set_yscale("log")
axs2[0].set_xscale("log")
axs2[0].set_xlabel(r"Energy density [MeV/fm$^3$]")
axs2[0].set_ylabel(r"Pressure [MeV/fm$^3$]")
axs2[1].set_ylabel(r'${\rm M~[M_\odot]}$')
axs2[1].set_xlabel(r'${\rm R~[km]}$')
axs2[0].grid(which = "both", ls = ":", lw = 0.3)
axs2[1].grid(which = "both", ls = ":", lw = 0.3)
# fig2
# plt.xlabel(r"Energy density [MeV/fm$^3$]")
# plt.ylabel(r"Pressure [MeV/fm$^3$]")

plt.show()

