using Plots 
using Pkg;
Pkg.activate("/Users/pavelzhelnin/Documents/physics/LJ_Project_Test/test_LJ")
using test_LJ
using QuadGK
using BenchmarkTools

includet("plotting_scripts.jl")
sim = SimulationConfig() 
sim.qubits = 100
sim.ϕmin = 0
sim.ϕmax = 0
sim.θmin = 0
sim.θmax = 0
sim.t_i = -50 
sim.t_f = 50
sim.Δt = 0.05
#sim.α = 1/10
sim.α = 0.3018
#sim.α = 1/6
sim.g = 1
sim.n = 7

#single qubit sim  
sim.qubits = 1 
sol = one_run_sim(sim)
super_s =  one_run_superadiabatic_sim(sim)
super_amplitudes_plot(super_s,sim)
amplitudes_plot(sol,sim)
#polar_animation = polar_plot(sol,sim)
#gif(polar_animation, "/Users/pavelzhelnin/Documents/physics/LJ_Project_Test/polar.gif", fps = 1)
#single_animation = single_spin_animation(sol,sim)
#gif(single_animation, "/Users/pavelzhelnin/Documents/physics/LJ_Project_Test/oscillation_rotation.gif", fps = 1)
amplitudes_plot(sol,sim)
energies_plot(sol,sim)
single_entropy_plot(sol)
cg_entropy_plot(sol,sim)
#many_cg_factor_entropy(sol,sim,[2,4,8,16,32,64,128])
many_cg_factor_entropy(sol,sim,[2,4])
projected_entropy(sol,sim)
projected_super_entropy(super_s,sim)
projected_superadiabatic_S(sol,sim)
calculation(sol,sim)
projected_z(sol,sim)
projected_y(sol,sim)
plot_frequencies(sol,sim)
plot_c(sim.t_i:sim.Δt:sim.t_f)
adiabatic_c_n([5],sim.t_i:sim.Δt:sim.t_f,sim.α;do_plot=true)
plot_s_adiabatic_stuff([5],sim.t_i:sim.Δt:sim.t_f,sim.α)
sim.n = 0
super_s =  one_run_superadiabatic_sim(sim)
plot_superA_zenith(super_s,sim)

#diabatic transition probability 
P = exp(-(π/2)*(sim.g^2)*(1/sim.α))
theta = 2 * asin(sqrt(P))
display((pi/2)*(1/sim.α))
super_s[end]' * super_s[end]
super_s[end][2]/0.0055 
super_s[end][2]*(sim.α)
display(rad2deg(theta))
display(normalize([sim.g,0,sim.α*sim.t_f]))
display(exp(-π*(sim.g^2)*(1/sim.α)))
display((sol[end][2])*(sol[end][2])')                     
display(1 - (sol[end][2])*(sol[end][2])')
#many qubit sim 
SX,SY,SZ,time = many_run_sim(sim)

animation = spin_animation(time[1],SX,SY,sim)
gif(animation, "/Users/pavelzhelnin/Documents/physics/LJ_Project_Test/rotation.gif", fps = 1000)

entropy_plot(SX,SY,SZ,time[1])

super_s[end][1]

#-0.9995015592501119 - 0.0311444681462919im
#-0.9999849297800067 - 5.9482479590891676e-5im
#-0.9999849040833195 + 0.00015819603057333045im