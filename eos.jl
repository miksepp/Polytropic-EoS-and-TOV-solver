using PyCall
@pyimport matplotlib.pyplot as mplt
using DifferentialEquations
using Interpolations
using Cubature
using Distributions
using DelimitedFiles
using ForwardDiff
using Statistics
using Trapz
using LaTeXStrings

const G = 6.6730831e-8
const c = 2.99792458e10
const MeV_fm3_to_pa = 1.6021766e32
const Msun = 1.988435e33
const MeV_fm3_to_pa_cgs = 1.6021766e33
const km_to_mSun = G/c^2
const pi = MathConstants.pi
const hbar3_c3=3.16002470519e-77


function glue_cEFT(e, p, mu, a)
    if a == 1
        cEFT_eos = readdlm("cEFT_soft.dat")
    else
        cEFT_eos = readdlm("cEFT_hard.dat")
    end
    p = vcat(cEFT_eos[:, 1], p)
    e = vcat(cEFT_eos[:, 2], e)
    mu = vcat(cEFT_eos[:, 3], mu)
    return e, p, mu
end

function glue_pQCD(e, p, mu)
    mu_i = range(2601, stop=3000, length=10)
    for x in range(1, stop=4, length=10000)
        p_i = 3/(4*pi^2*hbar3_c3) .* (mu_i*MeV/3).^4 .* (0.9008 .- (0.5034*x^(-0.3553))./(mu_i/1000 .- 1.452*x^(-0.9101)))./MeV_fm3_to_pa
        if abs(p_i[1] - p[end]) < 1
            e = vcat(e, range(e[end], stop=e[end]+30000, length=10))
            mu = vcat(mu, mu_i)
            p = vcat(p, p_i)
            return e, p, mu
        end
    end
    return [0], [0], [0]
end

function tov!(du, u, e, t)
    du[1] = -G * (e(u[1]) + u[1]/c^2) * (u[2] + 4.0 * pi * t^3 * u[1] / c^2) / (t * (t - 2.0 * G * u[2]/c^2))
    du[2] = 4.0 * pi * t^2 * e(u[1])
end

function love!(du, u, params, r)
    e_R, p_R, m_R, e = params[1], params[2], params[3], params[4]

    de_dp = ForwardDiff.derivative(e, p_R(r))

    beta = u[1]
    H = u[2]
    du[1] = H .* (-2 .* pi .* G ./ c .^ 2 .* (5 .* e_R(r) .+ 9 .* p_R(r) ./ c ^ 2 .+ de_dp .* c .^ 2 .* (e_R(r) .+ p_R(r) ./ c ^ 2)) .+ 3 ./ r .^ 2 .+ 2 .* (1 .- 2 .* m_R(r) ./ r .* km_to_mSun) .^ (-1) .* (m_R(r) ./ r .^ 2 .* km_to_mSun .+ G ./ c .^ 4 .* 4 .* pi .* r .* p_R(r)) .^ 2).+ beta ./ r .* (-1 .+ m_R(r) ./ r .* km_to_mSun .+ 2 .* pi .* r .^ 2 .* G ./ c .^ 2 .* (e_R(r) .- p_R(r) ./ c .^ 2)) .* 2 .* (1 .- 2 .* m_R(r) ./ r .* km_to_mSun) .^ (-1)
    du[2] = beta
end

function solve_tov(c_dens, p, e)
    c_dens *= MeV_fm3_to_pa_cgs / c^2
    r = LinRange(1, 2e6, 1000)
    P = p(c_dens)
    m = 4.0 * pi * r[1]^3 * c_dens

    #Solve mass-radius relationship
    u0 = [P, m]
    tspan = (1, 2.0e6)    
    psol = solve(ODEProblem(tov!, u0, tspan, e), AutoTsit5(Rosenbrock23()), reltol = 0.00001)
    p_R, m_R, r = psol[1,:], psol[2,:], psol.t

    diff = (m_R[2:end] .- m_R[1:end-1]) ./ m_R[2:end]
    ind = 0

    for (i, dm) in enumerate(diff)
        if dm < 10e-10 && m_R[i] != 0
            ind = i
            break
        end
    end

    if ind == 0
        M = m_R[end - 1]
        R = r[end - 1]
        
        r   = r[1:end]
        p_R = p_R[1:end]
        m_R = m_R[1:end]
    else
        M = m_R[ind - 1]
        R = r[ind - 1]
        
        r   = r[1:ind]
        p_R = p_R[1:ind]
        m_R = m_R[1:ind]
    end 

    # Solve tidal deformability:
    e_R = e(p_R)
    
    e_R = LinearInterpolation(r, e_R, extrapolation_bc=Flat())
    p_R = LinearInterpolation(r, p_R, extrapolation_bc=Flat())
    m_R = LinearInterpolation(r, m_R, extrapolation_bc=Flat())
    
    beta0 = 2 * r[1]
    H0 = r[1]^2
    
    
    u0_love = [beta0, H0]
    params = e_R, p_R, m_R, e
    love_sol = solve(ODEProblem(love!, u0_love, (1,2e6), params), AutoTsit5(Rosenbrock23()), reltol = 0.00001)
    beta = love_sol[1,end]
    H = love_sol[2,end]
    
    y = R * beta / H
    
    # Compactness
    C = M / R * km_to_mSun
    
    # Love number
    k2 = 8 / 5 * C^5 * (1 - 2 * C)^2 * (2 + 2 * C * (y - 1) - y) * (
        2 * C * (6 - 3 * y + 3 * C * (5 * y - 8)) + 4 * C^3 * (
            13 - 11 * y + C * (3 * y - 2) + 2 * C^2 * (1 + y)) + 3 * (1 - 2 * C)^2 * (2 - y + 2 * C * (y - 1)) * (
            log(1 - 2 * C)))^(-1)
    
    # Tidal deformability:
    L = 2/3 * k2 * C^(-5)
    
    return R / 1e5, M / Msun, L
end


function polytrope_interpolation(number_of_polytropes)
    while true
        a = rand(1:2)
        
        if a == 1
            mu_0, p_0, n_0, e_0, c_0 = 965.6988636, 2.163, 0.16 * 1.1, 167.8, 0.039525
        else
            mu_0, p_0, n_0, e_0, c_0 = 977.5113636, 3.542, 0.16 * 1.1, 168.5, 0.053470
        end
        
        mu = []
        p, n, e, c_s, g_arr = [p_0], [n_0], [e_0], [c_0], []
        mu_matching, n_matching, p_matching, e_matching = [], [], [], []
        
        push!(mu_matching, mu_0)
        j = 1
        while j < number_of_polytropes
            push!(mu_matching, rand(mu_0:2600))
            j += 1
        end
        
        push!(mu_matching, 2600)
        sort!(mu_matching)
        
        j = 1
        while j <= number_of_polytropes
            if j == 1
                push!(n_matching, n_0)
                push!(e_matching, e_0)
                push!(p_matching, p_0)
                mu_i = LinRange(mu_matching[j], mu_matching[j+1], 100)
            elseif j == number_of_polytropes
                push!(n_matching, n[end])
                push!(e_matching, e[end])
                push!(p_matching, p[end])
                mu_i = LinRange(mu_matching[j], 2600, 100)
            elseif j == number_of_polytropes + 1
                push!(n_matching, 40 * 0.16)
                g_i = rand(Uniform(0, 10))
                k_i = p_matching[end] * n_matching[end] ^ (-g_i)
                push!(e_matching, ((mu_i .* n_matching[end] .+ g_i * e_matching[end] .- p_matching[end] .- e_matching[end]) .* p_i ./ (mu_i .* (g_i - 1) .* n_matching[end] .- g_i * e_matching[end] .+ p_matching[end] .+ e_matching[end]))[end])
                push!(p_matching, p[end])
                break
            else
                push!(n_matching, n[end])
                push!(e_matching, e[end])
                push!(p_matching, p[end])
                mu_i = LinRange(mu_matching[j], mu_matching[j+1], 100)
            end
            
            push!(mu, mu_i...)
            
            g_i = rand(Uniform(0, 10))
            push!(g_arr, g_i)
            k_i = p_matching[end] * n_matching[end] ^ (-g_i)
            
            if minimum(n_matching[end]^(g_i - 1) .+ (g_i - 1) / (k_i * g_i) .* (mu_i .- mu_matching[j])) < 0 
                continue
            end
            p_i = k_i * (n_matching[end]^(g_i - 1) .+ (g_i - 1) / (k_i * g_i) .* (mu_i .- mu_matching[j])).^((g_i) / (g_i - 1))
            append!(p, p_i)
            
            e_i = (mu_i .* n_matching[end] .+ g_i * e_matching[end] .- p_matching[end] .- e_matching[end]) .* p_i ./ (mu_i .* (g_i - 1) .* n_matching[end] .- g_i * e_matching[end] .+ p_matching[end] .+ e_matching[end])
            n_i = (g_i * n_matching[end] .* p_i) ./ (mu_i .* (g_i - 1) .* n_matching[end] .- g_i * e_matching[end] .+ p_matching[end] .+ e_matching[end])
            
            append!(e, e_i)
            append!(n, n_i)
            
            c_i = g_i ./ (e_i ./ p_i .+ 1)
            append!(c_s, c_i)
            
            j += 1
        end

        if p[end] > 2579.5 && p[end] < 4242 && maximum(c_s) < 1 && minimum(c_s)>0 && e[end] < 24000 && length(mu) == length(p[2:end])
            
            e, p, mu = glue_cEFT(e, p, mu, a)
            e, p, mu = glue_pQCD(e, p, mu)


            if all(e .== 0)
                continue
            end
            
            e_cgs = sort(e .* MeV_fm3_to_pa_cgs ./ c.^2)
            p_cgs = sort(p .* MeV_fm3_to_pa_cgs)

            Interpolations.deduplicate_knots!(e_cgs, move_knots = true)
            Interpolations.deduplicate_knots!(p_cgs, move_knots = true)

            p_fun = LinearInterpolation(e_cgs, p_cgs, extrapolation_bc=Flat())
            e_fun = LinearInterpolation(p_cgs, e_cgs, extrapolation_bc=Flat())

            return e_fun, p_fun, mu, mu_matching, p_matching, e, p
        end
    end
end



# Prepare plots
fig1, axs1 = mplt.subplots(1,2, figsize=(10,5))
fig2, axs2 = mplt.subplots(1,1, figsize=(5, 5))
fig3, axs3 = mplt.subplots(1,1, figsize=(5, 5))


#Prepare EoSs
global k = 0
global poly_passable_EoS = 0
global poly_gamma_small_arr = []
global poly_gamma_max_arr = []


#Set number of EoSs
global number_of_EoS = 100000


while k < number_of_EoS

    # generate random polytropes with 2, 3 or 4 segments
    number_of_polytropes = rand(2:4)
    e, p, mu, mu_matching, p_matching, e_list, p_list = polytrope_interpolation(number_of_polytropes)


    # Optional plots of pressure as a function of chemical potential:
    # mplt.plot(mu_matching, p_matching,"o")
    # mplt.plot(mu, p_list[1:end])
    # e, p, n, n_matching, p_matching, e_list, p_list = bary_interpolation()
    # mplt.loglog(e_list, p_list[2:end], "g")


    m_arr = []
    R_arr = []
    L_arr = []
    central_e_arr = 10 .^ range(0, stop=3.45, length=200)
    

    for dens_c in central_e_arr
        R, M, L = solve_tov(dens_c, p, e)
        if R > 0.1
            m_arr = push!(m_arr, M)
            R_arr = push!(R_arr, R)
            L_arr = push!(L_arr, L)
        end
    end


    global L = linear_interpolation(sort(m_arr), L_arr)

    fig2
    axs2.loglog(e_list, p_list, "g", zorder = 2)
    fig1
    if maximum(m_arr) <= 2.01
        axs1[2].plot(R_arr, m_arr, "c", zorder = 1)
        axs1[1].loglog(e_list, p_list, "c", zorder = 1)
    elseif L(1.4) <= 70
        axs1[2].plot(R_arr, m_arr, "m", zorder = 2)
        axs1[1].loglog(e_list, p_list, "m", zorder = 2)
    elseif L(1.4) >= 580
        axs1[2].plot(R_arr, m_arr, "m", zorder = 2)
        axs1[1].loglog(e_list, p_list, "m", zorder = 2)
    else
        axs1[2].plot(R_arr, m_arr, "g", zorder = 3)
        axs1[1].loglog(e_list, p_list, "g", zorder = 3)
        axs3.loglog(e_list, p_list, color = "g", zorder = 2)

        global poly_passable_EoS += 1

        # Compute and analyse polytropic index:
        gamma = diff(log.(p_list))./diff(log.(e_list))
        idx_1 = argmin((abs.(m_arr .- 1.4)))
        idx_2 = argmin((abs.(e_list .- central_e_arr[idx_1 + (length(central_e_arr)- length(m_arr))])))
        if isnan(gamma[idx_2]) == false && gamma[idx_2]>0 && gamma[idx_2]<1000
            global poly_gamma_small_arr = push!(poly_gamma_small_arr, gamma[idx_2])
        end
        idx_1 = argmax(m_arr)
        idx_2 = argmin((abs.(e_list .- central_e_arr[idx_1 + (length(central_e_arr)- length(m_arr))])))
        if isnan(gamma[idx_2]) == false && gamma[idx_2]>0 && gamma[idx_2]<1000
            global poly_gamma_max_arr = push!(poly_gamma_max_arr, gamma[idx_2])
        end
    end
    global k += 1
    if k%100 == 0
        println(k)
    end
end

println("Polytrope: Number of passable EoS: $poly_passable_EoS")
if poly_passable_EoS>0 #&& pchip_passable_EoS>0
    println("Polytrope: 1.4M star adiabatic index in range $(minimum(poly_gamma_small_arr)), $(maximum(poly_gamma_small_arr)), with average $(mean(poly_gamma_small_arr))")
    println("Polytrope: Max mass star adiabatic index in range $(minimum(poly_gamma_max_arr)), $(maximum(poly_gamma_max_arr)), with average $(mean(poly_gamma_max_arr))")
end

mplt.rc("text", usetex=true)
mplt.rc("font", family="serif")
axs1[1].set_yscale("log")
axs1[1].set_xscale("log")
axs1[1].set_xlabel(L"Energy density [MeV/fm$^3$]", fontsize=12)
axs1[1].set_ylabel(L"Pressure [MeV/fm$^3$]", fontsize=12)
axs1[2].set_xlim(5, 17)
axs1[1].set_xlim(100,40000)
axs1[1].set_ylim(1,10000)
axs1[2].set_ylabel(L"${\rm M~[M_\odot]}$", fontsize=12)
axs1[2].set_xlabel(L"${\rm R~[km]}$", fontsize=12)
axs1[1].grid(which = "both", ls = ":", lw = 0.3)
axs1[2].grid(which = "both", ls = ":", lw = 0.3)

fig3
axs3.set_xlabel(L"Energy density [MeV/fm$^3$]", fontsize=12)
axs3.set_ylabel(L"Pressure [MeV/fm$^3$]", fontsize=12)
axs3.grid(which = "both", ls = ":", lw = 0.3)
axs3.set_xlim(100,40000)
axs3.set_ylim(1,10000)
axs2.set_xlabel(L"Energy density [MeV/fm$^3$]", fontsize=12)
axs2.set_ylabel(L"Pressure [MeV/fm$^3$]", fontsize=12)
axs2.grid(which = "both", ls = ":", lw = 0.3)
axs2.set_xlim(100,40000)
axs2.set_ylim(1,10000)


mplt.show()
