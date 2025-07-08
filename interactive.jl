#using Plots 
using Pkg;
Pkg.activate("/Users/pavelzhelnin/Documents/physics/LJ_Project_Test/test_LJ")
using test_LJ
using QuadGK
using BenchmarkTools
using StatsBase
using CairoMakie

includet("plotting_scripts.jl")
sim = SimulationConfig() 
sim.qubits = 1
sim.ϕmin = 0
sim.ϕmax = 2pi
sim.θmin = 0
sim.θmax = pi/2
sim.t_i = -10
sim.t_f = 10
sim.Δt = 0.01
#sim.α = 1/10
#sim.α = 0.314
sim.α = 5
#sim.α = 0.3
#sim.α = 1/6
sim.g = 1
sim.n = 0

#single qubit sim  
sim.qubits = 1
sol = one_run_sim(sim)
projected_entropy(sol,sim)
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
sim.α
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
f = Figure();
for (jdx,timedelta) in enumerate([9.98,10.0,10.02])
#for (jdx,timedelta) in enumerate([40,50,60])
    if jdx == 1
        f = Figure(size=(450,470))
    end 
    #label = "[-$(round(timedelta*10,sigdigits=4)):$(round(timedelta*10,sigdigits=4))]"
    if jdx == 1 
        label = L"[-(t_0 - δt), t_0 - δt]" 
    elseif jdx == 2 
        label = L"[-t_0,t_0]"
    else 
        label =L"[-(t_0 + δt), t_0 + δt]" 
    end 
    Label(f[0,jdx], label; fontsize=14, tellwidth=false)
    for (idx,delta) in enumerate([0.95,1.001,1.05])
        if delta == 1.001 
            Label(f[idx, 4], "1.00ϵ"; fontsize=14, width = 10)
        else 
            Label(f[idx, 4], "$(delta)ϵ"; fontsize=14, width = 10)
        end
       
        ax = Axis(f[idx, jdx],xticklabelsize=10,yticklabelsize=10)
        # if idx == 1
        #     ax = Axis(f[idx, jdx],aspect=1,xticklabelsize=10,yticklabelsize=10,xlabel="$(delta)ϵ")
        # elseif jdx == 1
        #     ax = Axis(f[idx, jdx],aspect=1,xticklabelsize=10,yticklabelsize=10,xlabel="t = [-$timedelta:$timedelta]")
        # else 
        #     ax = Axis(f[idx, jdx],aspect=1,xticklabelsize=10,yticklabelsize=10)
        # end
        xticks = idx == 3 ? false : true
        yticks = jdx == 1 ? false : true
        if xticks 
            hidexdecorations!(ax,grid=false,)
        end
        if yticks
            hideydecorations!(ax,grid=false)
        end
        sim.t_i = -timedelta*1
        sim.t_f = timedelta*1
        # sim.t_i = -timedelta
        # sim.t_f = timedelta
        sim.α = 5.0 * delta
        sim.qubits = 4000
        SX,SY,SZ,time = many_run_sim(sim)
        #sol = one_run_sim(sim)
        #println(timedelta)
        #println(delta)
        #projected_entropy(sol,sim)
        
        entropy_map(f,ax,SX,SY,SZ,sim)
    end 
    colsize!(f.layout, jdx, 100)
    rowsize!(f.layout, jdx, 100)
    # if jdx == 3 
    #     display(f)
    # end 
end 

#f = Figure();
Colorbar(f[4, 1:3],labelsize=12,ticklabelsize=12,vertical=false,limits=(-0.6,0.6), flipaxis=false,colormap=cgrad(:vik),ticks=-0.6:0.1:0.6,label="ΔS")
#resize_to_layout!(f)
#empty!(f[4,1:3])
display(f)
save("5HistoryMap.pdf",f)
using MathTeXEngine: FontFamily, texfont
FontFamily() = FontFamily("NewComputerModern")


update_theme!( 
    fonts=(; 
        regular=texfont(:regular),
        bold=texfont(:bold),
        italic=texfont(:italic),
        bolditalic=texfont(:bolditalic)
    )
)

for (idx,delta) in enumerate([0.34,0.89,5])
    if idx == 1
        f = Figure()
    end

    # if delta == 1.001 
    #     Label(f[idx, 4], "1.00ϵ"; fontsize=10, font=:bold, width = 10)
    # else 
    #     Label(f[idx, 4], "$(delta)ϵ"; fontsize=10, font=:bold, width = 10)
    # end
    
    ax = Axis3(f[1,idx],aspect=(1,1,1),xlabel="",ylabel="",zlabel="")
    entropy_ax = Axis(f[2,idx],aspect=1,xlabel="Time", ylabel=L"\Delta S")
    hidedecorations!(ax); hidespines!(ax);
    hidedecorations!(ax); hidespines!(ax);
  
    if idx == 1 
        sim.t_i = -500
        sim.t_f = 500
        sim.Δt = 0.1
    elseif idx == 2 
        sim.t_i = -50
        sim.t_f = 50
        sim.Δt = 0.001
    else 
        sim.t_i = -10
        sim.t_f = 10
        sim.Δt = 0.001
    end
    sim.α = delta
    sim.qubits = 1
    SX,SY,SZ,time = many_run_sim(sim)
    sphere_plot(ax,entropy_ax,SX,SY,SZ,sim)
    #entropy_map(f,ax,SX,SY,SZ,sim)
end 
colgap!(f.layout, 5)
rowgap!(f.layout, 1e-8)
display(f)
save("bloch_plot.png",f)
# rowgap!(f.layout, 1, -10)
# colgap!(f.layout, 1, Relative(-0.2))
# colgap!(f.layout, 2, Relative(-0.2))
# colgap!(f.layout, 3, 10)
animation = spin_animation(time[1],SX,SY,sim)
gif(animation, "/Users/pavelzhelnin/Documents/physics/LJ_Project_Test/rotation.gif", fps = 1000)
SX
entropy_plot(SX,SY,SZ,time[1])

super_s[end][1]

plot_xy(sol,sim;alpha=10)

y = 0.990:0.001:0.999

f = Figure();
ax = Axis(f[1,1],xlabel=L"x",ylabel=L"Q^2",aspect=1,xscale = log10, yscale = log10);
for idx in 1:1:5
    y = 0.00001:0.00001:0.9999
    E_ν = 10 .^ (idx)
    θ = 1*pi/180
    E_μ = (1 .- y).*E_ν
    Q_2 = (2 .*E_ν.*E_μ) .*  (1 .- cos(θ))
    x = Q_2./(2*(E_ν.-E_μ))
    lines!(ax,x,Q_2,label = L"E_\nu = 10^%$idx", linewidth=10)
    #axislegend(ax,position = :lb)
end 
Legend(f[1,2],ax)
xlims!(ax,1e-5,1)
ylims!(ax,1,1e5)
display(f)
d = 1 + (2*x)/E_ν
y = 1/d
Q_2 = (2*E_ν*x)/d
#-0.9995015592501119 - 0.0311444681462919im
#-0.9999849297800067 - 5.9482479590891676e-5im
#-0.9999849040833195 + 0.00015819603057333045im