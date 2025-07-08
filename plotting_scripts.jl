using Pkg;
Pkg.activate("/Users/pavelzhelnin/Documents/physics/LJ_Project_Test/test_LJ")
using test_LJ
using LinearAlgebra 
using Statistics
using FFTW 
using QuadGK
using StatsBase
using SpecialFunctions


function adiabatic_or_nah(sim::SimulationConfig)
    P = exp(-(π/2)*(sim.g^2)*(1/sim.α))
    if P > 0.9
        return "diabatic"
    elseif P < 0.1
        return "adiabatic"
    else 
        return "intermediate"
    end
end

function plot_xy(sol,s::SimulationConfig;alpha=alpha)
    sx,sy,sz = [],[],[]

    for (idx,t) in enumerate(sol.t)
        ψ = [sol[1,idx],sol[2,idx]]
        push!(sx, real(ψ'*σ[1]*ψ))
        push!(sy, real(ψ'*σ[2]*ψ))
        push!(sz, real(ψ'*σ[3]*ψ))
    end
    println(maximum(sy))
    println(maximum(sx.^2 .+ sy.^2 .+ sz.^2))
    title = adiabatic_or_nah(s)
    time = s.t_i:s.Δt:s.t_f
    scatter(
        sx[1:end],sy[1:end],marker_z=sz[1:end],
        # sx[1:length(time)-1:end], 
        # sy[1:length(time)-1:end], 
        # marker_z=rad2deg.(cos.(sz[1:length(time)-1:end])),
        title="$(title) transition",
        legend=:topleft,
        framestyle=:box,
        label="",
        xlimits=(-1,1),
        ylimits=(-1,1),
        aspect_ratio=:equal,
        # left_margin=5mm,
        # top_margin=5mm,
        # right_margin=5mm,
        # bottom_margin=5mm,
        fontfamily="Computer Modern",
        msw=0,
        # ms=10,
        xlabel="x",
        ylabel="y",
        tickfontsize=12,
        size=(600, 400)
    )
end
# function spin_animation(times,SX,SY,s::SimulationConfig)
#     q_anim = @animate for (idx,t) in enumerate(times) 
#         B = normalize([s.g,0,s.α*t])
#         #B = [0,s.g * -sin(t),s.α * cos(t)]
#         sxs = [sx[idx] for sx in SX]
#         sys = [sy[idx] for sy in SY]

#         scatter([sxs],[sys], 
#         label="time: $(t)",aspect_ratio=:equal, 
#         xlims=(-1,1), ylims=(-1,1),
#         legend=:topleft,
#         framestyle=:box,
#         # left_margin=5mm,
#         # top_margin=5mm,
#         # right_margin=5mm,
#         # bottom_margin=5mm,
#         fontfamily="Computer Modern",
#         tickfontsize=12,
#         size=(600, 400)
#         )
#         scatter!([B[1]], [B[2]], label="", xlims=(-1,1), ylims=(-1,1))
#         #scatter!([positions[:,1]], [sy[idx]], label="", xlims=(-1,1), ylims=(-1,1))
#     end
#     return q_anim 
# end 

function entropy_plot(SX,SY,SZ,time)
    entropy = [] 
    for (idx,_) in enumerate(time) 
        xs = [sx[idx] for sx in SX]
        ys = [sy[idx] for sy in SY]
        zs = [sz[idx] for sz in SZ]
        I = [1 0; 0 1]
        ρ = 1/2*(I + (mean(xs)*σ[1] + mean(ys)*σ[2] + mean(zs)*σ[3]))
        push!(entropy,-tr(ρ*log(ρ)))
    end 
    scatter(time, entropy, ylims=(0,log(2)+0.1),label="",title="entropy",
    legend=:topright,
    framestyle=:box,
    # left_margin=5mm,
    # top_margin=5mm,
    # right_margin=5mm,
    # bottom_margin=5mm,
    fontfamily="Computer Modern",
    tickfontsize=12,
    size=(600, 400)
    )
    hline!([log(2)], label="max entropy") 
end

function amplitudes_plot(sol,s::SimulationConfig)
    c1 = sol[1, :] .* conj(sol[1, :]) 
    c2 = sol[2, :] .* conj(sol[2, :])

    title = adiabatic_or_nah(s)

    τs = s.α .* sol.t
    wc = pi/2
    σ = zeros(length(τs))
    for (idx,τ) in enumerate(τs) 
        w = quadgk(x -> sqrt(x^2+s.g^2),0,τ)[1]
        σ[idx] = w/sqrt(2*sim.α*abs(wc))
    end 
    f = Figure() 
    ax = Axis(f[1, 1],aspect=1)
    xlims!(ax,(minimum(σ),maximum(σ)))
    lines!(ax,σ, 
        real.(sqrt.(c2)), 
        # title = "$(title) transition", 
        # label="|c1|^2",
        # legend=:right,
        # framestyle=:box,
        # xlimits=(-15,15),
        #ylimits=(0,1e-1),
        # left_margin=5mm,
        # top_margin=5mm,
        # right_margin=5mm,
        # bottom_margin=5mm,
        # fontfamily="Computer Modern",
        # xlabel="time",
        # ylabel="amplitude",
        # tickfontsize=12,
        # size=(600, 400)
        )
    
    #plot!(sol.t, real.(sqrt.(c2)), label="|c2|^2")
    hlines!(ax,[exp(-pi/(2*sim.α))], label="final c1 asymptote")
    display(f)
end 

function super_amplitudes_plot(sol,s::SimulationConfig)
    #sol[2,:] = sol[2,:] .* sim.α
    c1 = sol[1, :] .* conj(sol[1, :]) 
    c2 = sol[2, :] .* conj(sol[2, :])

    title = adiabatic_or_nah(s)

    τs = s.α .* sol.t
    wc = pi/2
    σ = zeros(length(τs))
    for (idx,τ) in enumerate(τs) 
        w = quadgk(x -> sqrt(x^2+s.g^2),0,τ)[1]
        σ[idx] = w/sqrt(2*sim.α*abs(wc))
    end 
    plot(σ, 
    real.(sqrt.(c1)),
        #real.(sqrt.(c2)), 
        title = "$(title) transition", 
        label="|c2|^2",
        legend=:right,
        framestyle=:box,
        # left_margin=5mm,
        # top_margin=5mm,
        # right_margin=5mm,
        # bottom_margin=5mm,
        fontfamily="Computer Modern",
        xlimit=(-5,5),
        xlabel="time",
        ylabel="amplitude",
        tickfontsize=12,
        size=(600, 400)
        )
    hline!([exp(-wc/sim.α)], label="final c2 asymptote")
    # plot!(sol.t, real.(sqrt.(c1)), label="|c2|^2")
end 

function energies_plot(sol,s::SimulationConfig)
    E=[]
    for t in sol.t
        H1 = H(s.α, s.g, t)
        #H1 = [s.α*cos(t) im*s.g*sin(t); -im*s.g*sin(t) -s.α*cos(t)]
        push!(E,eigvals(H1))
    end
    
    title = adiabatic_or_nah(s)
    e1 = [e[1] for e in E]
    plot(sol.t, e1./maximum(abs.(e1)), 
        title = "$(title) transition", label="E1",
        legend=:right,
        framestyle=:box,
        xlabel="time",
        ylabel="energy",
        # left_margin=5mm,
        # top_margin=5mm,
        # right_margin=5mm,
        # bottom_margin=5mm,
        fontfamily="Computer Modern",
        tickfontsize=12,
        size=(600, 400)
    )

    e2 = [e[2] for e in E] 
    plot!(sol.t, e2/maximum(abs.(e2)), label="E2")
end

# function single_spin_animation(sol,s::SimulationConfig)
#     sx,sy,sz = [],[],[]

#     for (idx,t) in enumerate(sol.t)
#         ψ = [sol[1,idx],sol[2,idx]]
#         push!(sx, real(ψ'*σ[1]*ψ))
#         push!(sy, real(ψ'*σ[2]*ψ))
#         push!(sz, real(ψ'*σ[3]*ψ))
#     end

#     q_anim = @animate for (idx,t) in enumerate(sol.t) 
#         #B = [0,s.g * -sin(t),s.α * cos(t)]
#         B = normalize([s.g,0,s.α*t])
#         scatter([B[1]],[B[2]], label="time: $(t)", xlims=(-1,1), ylims=(-1,1))
#         scatter!([sx[idx]], [sy[idx]], label="", xlims=(-1,1), ylims=(-1,1))
#     end 
#     return q_anim 
# end 

function single_entropy_plot(sol)
    sx,sy,sz = [],[],[]

    for (idx,t) in enumerate(sol.t)
        ψ = [sol[1,idx],sol[2,idx]]
        push!(sx, real(ψ'*σ[1]*ψ))
        push!(sy, real(ψ'*σ[2]*ψ))
        push!(sz, real(ψ'*σ[3]*ψ))
    end
   
    entropy = [] 
    for (idx,_) in enumerate(sol.t) 
        xs = sx[idx]
        ys = sy[idx]
        zs = sz[idx]
        ρ = 1/2*(I + (xs*σ[1] + ys*σ[2] + zs*σ[3]))

        if tr(ρ) == 1 
            push!(entropy,0)
        end

        push!(entropy,-tr(ρ*log(ρ)))
    end 
    f = Figure() 
    ax = Axis(f[1, 1], aspect=1)
    ylims!(ax,(0,log(2)+0.1))
    scatter!(ax, sol.t, real.(entropy), 
        # label="single qubit S",
        # title="entropy",
        # legend=:topleft,
        # framestyle=:box,
        # left_margin=5mm,
        # top_margin=5mm,
        # right_margin=5mm,
        # bottom_margin=5mm,
        # fontfamily="Computer Modern",
        # tickfontsize=12,
        # size=(600, 400)
        )
    hlines!(ax,[log(2)]) 
    display(f)
end 

function cg_entropy_plot(sol,s::SimulationConfig)
    sx,sy,sz = [],[],[]

    for (idx,t) in enumerate(sol.t)
        ψ = [sol[1,idx],sol[2,idx]]
        push!(sx, real(ψ'*σ[1]*ψ))
        push!(sy, real(ψ'*σ[2]*ψ))
        push!(sz, real(ψ'*σ[3]*ψ))
    end
    
    entropy = [] 
    cg_entropy = [] 
    for (idx,_) in enumerate(sol.t) 

        window_size = s.cg_factor 
        start_index = max(1, idx - window_size + 1)
        end_index = idx
        cg_sx = mean(sx[start_index:end_index])
        cg_sy = mean(sy[start_index:end_index])
        cg_sz = mean(sz[start_index:end_index])


        xs = sx[idx]
        ys = sy[idx]
        zs = sz[idx]

        ρ = 1/2*(I + (xs*σ[1] + ys*σ[2] + zs*σ[3]))
        cg_ρ = 1/2*(I + (cg_sx*σ[1] + cg_sy*σ[2] + cg_sz*σ[3]))

        if tr(ρ) == 1 
            push!(entropy,0)
        end

        push!(entropy,-tr(ρ*log(ρ)))
        push!(cg_entropy,-tr(cg_ρ*log(cg_ρ)))
    end 
    scatter(sol.t, real.(entropy), ylims=(0,log(2)+0.1),label="single qubit S",title="entropy",
        legend=:topleft,
        framestyle=:box,
        # left_margin=5mm,
        # top_margin=5mm,
        # right_margin=5mm,
        # bottom_margin=5mm,
        fontfamily="Computer Modern",
        tickfontsize=12,
        size=(600, 400)
        )
    scatter!(sol.t, cg_entropy,label="coarse grained,$(s.cg_factor), S")
    hline!([log(2)], label="max S") 
end 

function many_cg_factor_entropy(sol,s::SimulationConfig,cgs)

    sx,sy,sz = [],[],[]
    title = adiabatic_or_nah(s)
    for (idx,t) in enumerate(sol.t)
        ψ = [sol[1,idx],sol[2,idx]]
        push!(sx, real(ψ'*σ[1]*ψ))
        push!(sy, real(ψ'*σ[2]*ψ))
        push!(sz, real(ψ'*σ[3]*ψ))
    end

    entropy = [] 

    for (idx,_) in enumerate(sol.t) 

        xs = sx[idx]
        ys = sy[idx]
        zs = sz[idx]

        ρ = 1/2*(I + (xs*σ[1] + ys*σ[2] + zs*σ[3]))

        if tr(ρ) == 1 
            push!(entropy,0)
        end
        push!(entropy,-tr(ρ*log(ρ)))
    end 

    plt = scatter(sol.t, 
         real.(entropy), 
        ylims=(0,log(2)+0.1),
        label="single qubit S",
        title="$(title) transition",
        legend=:topleft,
        framestyle=:box,
        # left_margin=5mm,
        # top_margin=5mm,
        # right_margin=5mm,
        # bottom_margin=5mm,
        fontfamily="Computer Modern",
        msw=0,
        ms=2,
        xlabel="time",
        ylabel="entropy",
        tickfontsize=12,
        size=(600, 400)
        )
    hline!(plt,[log(2)], label="max S") 

    for cg in cgs 
        cg_entropy = []
        for (idx,_) in enumerate(sol.t)
            window_size = cg
            start_index = max(1, idx - window_size + 1)
            end_index = idx
            cg_sx = mean(sx[start_index:end_index])
            cg_sy = mean(sy[start_index:end_index])
            cg_sz = mean(sz[start_index:end_index])
            cg_ρ = 1/2*(I + (cg_sx*σ[1] + cg_sy*σ[2] + cg_sz*σ[3]))
            push!(cg_entropy,-tr(cg_ρ*log(cg_ρ)))
        
        end
        @show size(cg_entropy)
        @show size(sol.t)
        scatter!(plt, sol.t, ms=2, real.(cg_entropy),msw=0,label="coarse grained,$(cg), S")
    end 
    display(plt)
end 

function projected_z(sol,s::SimulationConfig)
    sx,sy,sz = [],[],[]

    for (idx,t) in enumerate(sol.t)
        ψ = [sol[1,idx],sol[2,idx]]
        push!(sx, real(ψ'*σ[1]*ψ))
        push!(sy, real(ψ'*σ[2]*ψ))
        push!(sz, real(ψ'*σ[3]*ψ))
    end

    title = adiabatic_or_nah(s)
    scatter(
        sol.t, 
        sz, 
        ylims=(-1.1,1.1),
        label="qubit",
        title="$(title) transition for z component",
        legend=:topleft,
        framestyle=:box,
        # left_margin=5mm,
        # top_margin=5mm,
        # right_margin=5mm,
        # bottom_margin=5mm,
        fontfamily="Computer Modern",
        msw=0,
        ms=2,
        xlabel="time",
        ylabel="z",
        tickfontsize=12,
        size=(600, 400)
        )

        B = [normalize([s.g,0,s.α*t]) for t in sol.t]
        #B = [[0,s.g * -sin(t),s.α * cos(t)] for t in sol.t]
        scatter!(sol.t, [b[3] for b in B], label="B")
end 

function projected_y(sol,s::SimulationConfig)
    sx,sy,sz = [],[],[]

    for (idx,t) in enumerate(sol.t)
        ψ = [sol[1,idx],sol[2,idx]]
        push!(sx, real(ψ'*σ[1]*ψ))
        push!(sy, real(ψ'*σ[2]*ψ))
        push!(sz, real(ψ'*σ[3]*ψ))
    end

    title = adiabatic_or_nah(s)
    scatter(
        sol.t, 
        sy, 
        ylims=(-1.1,1),
        label="qubit",
        title="$(title) transition for y component",
        legend=:topleft,
        framestyle=:box,
        # left_margin=5mm,
        # top_margin=5mm,
        # right_margin=5mm,
        # bottom_margin=5mm,
        fontfamily="Computer Modern",
        msw=0,
        ms=2,
        xlabel="time",
        ylabel="y",
        tickfontsize=12,
        size=(600, 400)
        )

        B = [normalize([s.g,0,s.α*t]) for t in sol.t]
        #B = [[0,s.g * -sin(t),s.α * cos(t)] for t in sol.t]
        scatter!(sol.t, [b[2] for b in B], label="B")
end 

function projected_super_entropy(sol,s)
    sx,sy,sz = [],[],[]

    for (idx,t) in enumerate(sol.t)
        ψ = [sol[1,idx],sol[2,idx]]
        push!(sx, real(ψ'*σ[1]*ψ))
        push!(sy, real(ψ'*σ[2]*ψ))
        push!(sz, real(ψ'*σ[3]*ψ))
    end

    #display("this is sz: $(sz)")
    entropy = [] 
    fidelity = []
    targets = []
    final_ρs = []
    αs = []
    Bs = []
    ε = sim.α
    n = sim.n
    g = sim.g

    for (idx,t) in enumerate(sol.t) 

        xs = sx[idx]
        ys = sy[idx]
        zs = sz[idx]

        
        
        τ = t*ε

        ψ = [sol[1,idx],sol[2,idx]]

        w_c = -im * pi/2

        function H(g,τ)
            return sqrt(g^2 + τ^2)
        end
        function a_θ(n,w_c,w)
            return (1/(2*im))*((factorial(n) * im^(n+1))/2π) * (1/(w - w_c)^(n+1) - 1/(w - adjoint(w_c))^(n+1))
        end
        function b(n,w_c,w)
            return -2 * a_θ(n,w_c,w)
        end 
        w = quadgk(x -> 2*H(g,x),0,τ)[1]
        z = -ε^(n+2) * 4 * H(g,τ) * a_θ(n,w_c,w) * adjoint(b(0,w_c,w))
        xy = ε^(n+1) * 4*H(g,τ) * adjoint(a_θ(n,w_c,w)) * exp((im*w)/ε)
        B = normalize([real(xy),imag(xy),real(z)])
        #display("this is B $(B)")
        #B = [0,s.g * -sin(t),s.α * cos(t)]
        xs_B,ys_B,zs_B = dot(B,normalize([xs,ys,zs]))*B

        push!(Bs,real(z))
        α = acos(dot([real(xy),imag(xy),real(z)],[xs,ys,zs])/(norm([xs,ys,zs])*norm([real(xy),imag(xy),real(z)])))
        push!(αs,α)
        #display(dot(B,[xs,ys,zs]))

        ρ = 1/2*(I + (xs_B*σ[1] + ys_B*σ[2] + zs_B*σ[3]))
        #display(ρ)
        new_ρ = ψ * ψ'

        target_ρ = sol[idx] * sol[idx]'
        push!(targets,target_ρ)

        # f = tr(sqrt(sqrt(ρ)*target_ρ*sqrt(ρ)))
        
        # push!(fidelity, f)

        final_target = [1 0; 0 0]
        #f = tr(sqrt(sqrt(ρ)*final_target*sqrt(ρ)))
        f = tr(sqrt(sqrt(new_ρ)*final_target*sqrt(new_ρ)))
        push!(final_ρs,f)

        # if tr(ρ) == 1 
        #     # display(ρ)
        #     # display(-tr(ρ*log(ρ)))
        #     push!(entropy,0)
        #     continue 
        # end
        
        if isnan(ρ[1][1])
            push!(entropy,0)
        else 
            push!(entropy,-tr(ρ*log(ρ)))
        end
    end 

    display("this is entropy: $entropy")

    c_n = exp(-(π/2)*(sim.g^2) /sim.α) 
    # final_vector = [0,0,cos(theta)-0.002]
    final_vector = [0,0,1-2*c_n^2]
    α = 1 - 2*c_n^2
    X,Z = s.g,s.α*sol.t[end]
    final_vector = zeros(3)
    final_vector[3] = (Z*α * sqrt((Z^2 * α^2) -(X^2 +Z^2)*(α^2 -X^2)))/(X^2 +Z^2)
    #final_vector[3] = 1

    final_ρ = 1/2*(I + (final_vector[1]*σ[1] + final_vector[2]*σ[2] + final_vector[3]*σ[3]))
    final_S = -tr(final_ρ*log(final_ρ))
    if c_n == 0 
        final_S = 0 
    end
  
    title = adiabatic_or_nah(s)
    plt = plot(sol.t, 
         real.(entropy), 
        ylims=(-0.05,log(2)+0.1),
        label="single qubit S",
        title="$(title) transition",
        legend=:topleft,
        framestyle=:box,
        # left_margin=5mm,
        # top_margin=5mm,
        # right_margin=5mm,
        # bottom_margin=5mm,
        fontfamily="Computer Modern",
        msw=0,
        ms=2,
        xlabel="time",
        ylabel="entropy",
        tickfontsize=12,
        size=(600, 400)
        )
        hline!(plt,[log(2)], label="max S") 
        hline!(plt,[real.(final_S)], label="final S")
    plt2 =  plot(sol.t, 
        (real.(final_ρs)).^2,
        #ylims=(-0.05,log(2)+0.1),
        ylabel="fidelity",
        xlabel="time",
        ylimits=(0,1),
        tickfontsize=12,)
    #scatter!(plt2, sol.t, (real.(final_ρs)).^2 , label="final ρ")
     #display("this is the entropy: $(entropy)")
    
    plt3 = plot(sol.t, 
        real.(αs), 
        ylims=(0,π),
        label="α",
        title="$(title) transition n = $(sim.n)",
        legend=:topleft,
        framestyle=:box,
        # left_margin=5mm,
        # top_margin=5mm,
        # right_margin=5mm,
        # bottom_margin=5mm,
        fontfamily="Computer Modern",
        msw=0,
        ms=2,
        xlabel="time",
        ylabel="α",
        tickfontsize=12,
        size=(600, 400)
        )

    plt2 = plot(sol.t,sx,xlims=(0,20),ylims=(-0.05,0))
    plt3 = plot(sol.t,sy)
    plt4 = plot(sol.t,1 .+sz)
    #plt4 = plot(sx,sy)
    display("this is B: $(Bs)")
    display("this is sz: $(sz)")
    display("this is sy: $(sy)")
    display("this is sx: $(sx)")

    display(plot(plt, plt2, plt3, plt4,layout = (2, 2), legend = false))
end 

function projected_entropy(sol,s::SimulationConfig)

    sx,sy,sz = [],[],[]

    for (idx,t) in enumerate(sol.t)
        ψ = [sol[1,idx],sol[2,idx]]
        push!(sx, real(ψ'*σ[1]*ψ))
        push!(sy, real(ψ'*σ[2]*ψ))
        push!(sz, real(ψ'*σ[3]*ψ))
    end

    #display("this is sz: $(sz)")
    entropy = [] 
    fidelity = []
    targets = []
    final_ρs = []

    for (idx,t) in enumerate(sol.t) 

        xs = sx[idx]
        ys = sy[idx]
        zs = sz[idx]

        B = normalize([s.g,0,s.α*t])
        #B = [0,s.g * -sin(t),s.α * cos(t)]
        xs_B,ys_B,zs_B = dot(B,[xs,ys,zs])*B
        #display(dot(B,[xs,ys,zs]))

        ρ = 1/2*(I + (xs_B*σ[1] + ys_B*σ[2] + zs_B*σ[3]))
        target_ρ = sol[idx] * sol[idx]'
        push!(targets,target_ρ)

        f = tr(sqrt(sqrt(ρ)*target_ρ*sqrt(ρ)))
        push!(fidelity, f)

        final_target = [1 0; 0 0]
        f = tr(sqrt(sqrt(ρ)*final_target*sqrt(ρ)))
        push!(final_ρs,f)

        # if tr(ρ) == 1 
        #     # display(ρ)
        #     # display(-tr(ρ*log(ρ)))
        #     push!(entropy,0)
        #     continue 
        # end
        if isnan(-tr(ρ*log(ρ)))
            push!(entropy,0)
        else 
            push!(entropy,-tr(ρ*log(ρ)))
        end
    end 
    
    c_n = exp(-(π/2)*(sim.g^2) /sim.α) 
    # final_vector = [0,0,cos(theta)-0.002]
    final_vector = [0,0,1-2*c_n^2]
    #display(final_vector)
    α = 1 - 2*c_n^2
    X,Z = s.g,s.α*sol.t[end]
    final_vector = zeros(3)
    final_vector[3] = (Z*α * sqrt((Z^2 * α^2) -(X^2 +Z^2)*(α^2 -X^2)))/(X^2 +Z^2)
    #final_vector[3] = 1
    # display(final_vector)
    final_ρ = 1/2*(I + (final_vector[1]*σ[1] + final_vector[2]*σ[2] + final_vector[3]*σ[3]))
    final_S = -tr(final_ρ*log(final_ρ))
    if c_n == 0 
        final_S = 0 
    end
    # display(final_S)
    # display(entropy[end])
    # display(real.(entropy[1]))
    # display(entropy[end-10])
    # display(targets[end])
    title = adiabatic_or_nah(s)
    f = Figure() 
    ax = Axis(f[1,1])
    lines!(ax, 
         real.(entropy), 
        # ylims=(-0.05,log(2)+0.1),
        # label="single qubit S",
        # title="$(title) transition",
        # legend=:topleft,
        # framestyle=:box,
        # # left_margin=5mm,
        # # top_margin=5mm,
        # # right_margin=5mm,
        # # bottom_margin=5mm,
        # fontfamily="Computer Modern",
        # msw=0,
        # ms=2,
        # xlabel="time",
        # ylabel="entropy",
        # tickfontsize=12,
        # size=(600, 400)
        )
        hlines!(ax,[log(2)], label="max S") 
        hlines!(ax,[real.(final_S)], label="final S")
        println(final_S)
        ax2 = Axis(f[1,2])
    lines!(ax2,sol.t, 
        (real.(final_ρs)).^2,
        #ylims=(-0.05,log(2)+0.1),
        # ylabel="fidelity",
        # xlabel="time",
        # ylimits=(0,1),
        # tickfontsize=12,)
    )
    display(f)
    #scatter!(plt2, sol.t, (real.(final_ρs)).^2 , label="final ρ")
    #display(f)
    #display(plot(plt, plt2, layout = (1, 2), legend = false))

    # for cg in cgs 
    #     cg_entropy = []
    #     for (idx,_) in enumerate(sol.t)
    #         window_size = cg
    #         start_index = max(1, idx - window_size + 1)
    #         end_index = idx
    #         cg_sx = mean(sx[start_index:end_index])
    #         cg_sy = mean(sy[start_index:end_index])
    #         cg_sz = mean(sz[start_index:end_index])
    #         cg_ρ = 1/2*(I + (cg_sx*σ[1] + cg_sy*σ[2] + cg_sz*σ[3]))
    #         push!(cg_entropy,-tr(cg_ρ*log(cg_ρ)))
        
    #     end
    #     @show size(cg_entropy)
    #     @show size(sol.t)
    #     scatter!(plt, sol.t, ms=2, real.(cg_entropy),msw=0,label="coarse grained,$(cg), S")
    # end 
    # display(plt)
end 

function projected_superadiabatic_S(sol,s::SimulationConfig)

    sx,sy,sz = [],[],[]
    σ = [[0 1; 1 0], [0 -im; im 0], [1 0; 0 -1]]
    for (idx,t) in enumerate(sol.t)
        ψ = [sol[1,idx],sol[2,idx]]
        push!(sx, real(ψ'*σ[1]*ψ))
        push!(sy, real(ψ'*σ[2]*ψ))
        push!(sz, real(ψ'*σ[3]*ψ))
    end

    δ = sim.α 
    Δ = sim.g
    τs = δ*sol.t
    new_xs = [] 
    new_ys = []
    new_zs = [] 
    Bx = [] 
    display(δ)
    for τ in τs 
        integral = quadgk(x -> sqrt(x^2+Δ^2),0,τ)[1]
        θdot = -Δ/(τ^2 + Δ^2)
        H = sqrt(τ^2 +Δ^2)
        a = -im/(2*H) * 1/4 * θdot^2
        result = -4*δ*(H/θdot)*a*exp(-2im/δ * integral)
        β = -2*a/θdot
        z_result = δ^(2) * β * 4 * H * conj(a)/conj(θdot)

        append!(Bx, [normalize([Δ,0,τ])][1][1])
        append!(new_xs,real(result))
        append!(new_ys,imag(result))
        append!(new_zs,-(1/2)+z_result)
        #other_result = conj(result)
        #println("τ: $τ, result: $result")
    end

    new_entropy=[]
    entropy=[]
   
    for (idx,t) in enumerate(sol.t) 

        xs = sx[idx]
        ys = sy[idx]
        zs = sz[idx]

        B = normalize([s.g,0,s.α*t])
        H_new = normalize([new_xs[idx],new_ys[idx],new_zs[idx]])
        #B = [0,s.g * -sin(t),s.α * cos(t)]
        xs_B,ys_B,zs_B = dot(B,[xs,ys,zs])*B
        new_xs_B,new_ys_B,new_zs_B = dot(H_new,[new_xs[idx],new_ys[idx],new_zs[idx]])*H_new
        ρ = 1/2*(I + (xs_B*σ[1] + ys_B*σ[2] + zs_B*σ[3]))
        new_ρ = 1/2*(I + (new_xs_B*σ[1] + new_ys_B*σ[2] + new_zs_B*σ[3]))
        if tr(ρ) == 1 
            push!(entropy,0)
            continue
        end

        if tr(new_ρ) == 1 
            @show "trace = 1"
            push!(new_entropy,0)
            continue
        end
        display("trace calc")
        display(-tr(new_ρ*log(new_ρ)))
        push!(new_entropy,-tr(new_ρ*log(new_ρ)))
        push!(entropy,-tr(ρ*log(ρ)))
    end 
    P = exp(-π*(sim.g^2)*(1/sim.α))
    #theta = 2 * asin(sqrt(P))
    #theta = P
    #final_vector = [0,0,cos(theta)]
    display("this is P: $P")
    final_vector = [0,0,1-2*P]
    final_ρ = 1/2*(I + (final_vector[1]*σ[1] + final_vector[2]*σ[2] + final_vector[3]*σ[3]))
    display(final_ρ)
    display(log(final_ρ))
    display("this is new S")
    @show new_entropy
    final_S = -tr(final_ρ*log(final_ρ))
    display(real(final_S))
    @show real(final_S)==NaN
    if isnan(final_S)
        display("true")
        final_S = 0 
    end 

    title = adiabatic_or_nah(s)
    plt = scatter(sol.t, 
         real.(entropy), 
        ylims=(-0.05,log(2)+0.1),
        label="single qubit S",
        title="$(title) transition",
        legend=:bottomright,
        framestyle=:box,
        # left_margin=5mm,
        # top_margin=5mm,
        # right_margin=5mm,
        # bottom_margin=5mm,
        fontfamily="Computer Modern",
        msw=0,
        ms=2,
        xlabel="time",
        ylabel="entropy",
        tickfontsize=12,
        size=(600, 400)
        )
        #scatter!(plt, sol.t, real.(new_entropy), label="superadiabatic S")
        display(final_S)
        hline!(plt,[log(2)], label="max S") 
        hline!(plt,[real.(final_S)], label="final S")
    display(plt)
end 

function calculation(sol,s::SimulationConfig)

    sx,sy,sz = [],[],[]

    for (idx,t) in enumerate(sol.t)
        ψ = [sol[1,idx],sol[2,idx]]
        push!(sx, real(ψ'*σ[1]*ψ))
        push!(sy, real(ψ'*σ[2]*ψ))
        push!(sz, real(ψ'*σ[3]*ψ))
    end

    display(filter(x -> x < 1, sx.^2 + sy.^2 + sz.^2))
    entropy=[]
    for (idx,t) in enumerate(sol.t) 

        xs = sx[idx]
        ys = sy[idx]
        zs = sz[idx]

        ρ = 1/2*(I + (xs*σ[1] + ys*σ[2] + zs*σ[3]))

        # if tr(ρ) == 1 
        #     display(ρ)
        #     display(-tr(ρ*log(ρ)))
        #     break 
        #     push!(entropy,0)
        # end
        push!(entropy,-tr(ρ*log(ρ)))
    end 
    display(entropy)
    display(filter(x -> x > 1e-2, real.(entropy)))
    
end 

function plot_frequencies(sol,s::SimulationConfig)
    
    sx,sy,sz = [],[],[]

    for (idx,t) in enumerate(sol.t)
        ψ = [sol[1,idx],sol[2,idx]]
        push!(sx, real(ψ'*σ[1]*ψ))
        push!(sy, real(ψ'*σ[2]*ψ))
        push!(sz, real(ψ'*σ[3]*ψ))
    end


    mask = sol.t .> 0.0
    F = fft(Float64.(sy[mask]))
    freqs = fftfreq(length(sol.t[mask]), 1/s.Δt)

    time_domain = plot(sol.t[mask], sy[mask], xlabel="time", label="",title = "Signal",legend=:top)
    freq_domain = plot(freqs, abs.(F), title = "Spectrum", xlabel="frequency", xlim=(-10, +10), xticks=-10:2:10, label="",legend=:top)
    plot(time_domain, freq_domain, layout = (2,1))
end

# function polar_plot(sol, s::SimulationConfig)

#     SX,SY,SZ = [],[],[]

#     for (idx,t) in enumerate(sol.t)
#         ψ = [sol[1,idx],sol[2,idx]]
#         push!(SX, real(ψ'*σ[1]*ψ))
#         push!(SY, real(ψ'*σ[2]*ψ))
#         push!(SZ, real(ψ'*σ[3]*ψ))
#     end

#     q_anim = @animate for (idx,t) in enumerate(sol.t) 
#         B = normalize([s.g,0,s.α*t])
#         #B = [0,s.g * -sin(t),s.α * cos(t)]
#         sxs = [SX[idx]]
#         sys = [SY[idx]]
#         szs = [SZ[idx]]

#         r = sqrt.(sxs.^2 + sys.^2 + szs.^2)
#         θ = acos.(szs ./ r) 

     
#         B_r = sqrt.(B[1]^2 + B[2]^2 + B[3]^2)
#         B_θ = acos(B[3]/B_r)

#         scatter([θ],[r],proj=:polar, 
#         label="time: $(t)",#aspect_ratio=:equal, 
#         xlims=(0,1), ylims=(0,1),
#         legend=:topleft,
#         framestyle=:box,
#         # left_margin=5mm,
#         # top_margin=5mm,
#         # right_margin=5mm,
#         # bottom_margin=5mm,
#         fontfamily="Computer Modern",
#         tickfontsize=12,
#         size=(600, 400)
#         )
#         scatter!([B_θ], [B_r], proj=:polar,label="B")
#         #scatter!([B[1]], [B[2]], label="", xlims=(-1,1), ylims=(-1,1))
#         #scatter!([positions[:,1]], [sy[idx]], label="", xlims=(-1,1), ylims=(-1,1))
#     end
#     return q_anim 
# end

function a_θ(n,w_c,w)
    return (1/2*im)*((factorial(n) * im^(n+1))/2π) * (1/(w - w_c)^(n+1) - 1/(w - adjoint(w_c))^(n+1))
end

function H_pp(n,w_c,w,H,ε)
    return ε^(n+2) -2*adjoint(a_θ(0,w_c,w)) * 4 * a_θ(n,w_c,w)*H
end

function H_pm(n,w_c,w,H,ε)
    return -ε^(n+1) * 4 * H * a_θ(n,w_c,w) * exp((im*w)/ε)
end

function plot_c(τs)
    wc = -im * pi/2 
    Δ = 1
    ε= 0.3018
    xs = []
    σs = []
    for τ in τs
        w = quadgk(x -> sqrt(x^2+Δ^2),0,τ)[1]
        σ = w/sqrt(2*ε*abs(wc))
        push!(σs,σ)
        x = 1/2 * (1 + erf(σ)) * exp(-abs(wc)/ε)
        push!(xs,x)
    end 
    plt = plot(σs,xs,xlimit=(-6,6),label="")
    display(plt)
end 

function adiabatic_c_n(ns,τs,ε; do_plot=true)
    wc = -im * pi/2 
    Δ = 1
    #5.2
    #ε= 0.3018*2
    #7.8
    #ε = 0.3018 /1.5
    #2.6
    #ε = 0.3018 *2
    c_n = zeros(length(τs))
    σ = zeros(length(τs))
    xs = []
    function integrand(w,ε,wc,n)
        return exp((-im * w)/ε) * (1/(w-wc)^(n+1) - 1/(w-adjoint(wc))^(n+1))
    end 
    if do_plot
        plt = plot(
            xlimit=(-10,10),
            #yscale=:log10,
            #ylimit=(1e-5,1e-0),
            legend=:bottomright,
            xlabel = "time", 
            title="transition histories",
            fontfamily="Computer Modern",
        )
    end
    for (j,n) in enumerate(ns)
        for (idx,τ) in enumerate(τs) 
            w = quadgk(x -> sqrt(x^2+Δ^2),0,τ)[1]
            σ[idx] = w/sqrt(2*ε*abs(wc))
            #integral = quadgk(x -> exp((-im * x)/ε) * (1/(x-wc)^(n+1) - 1/(x-adjoint(wc))^(n+1)),-Inf,w)[1]
            integral = quadgk(x -> integrand(x,ε,wc,n),-Inf,w)[1]
            display(integral)
            c = (im^(n+1) * ε^n *factorial(n))/2π *integral[1]
            c_n[idx] = c*adjoint(c)
            #c_n[idx] = (im^(n+1) * ε^n *factorial(n))/2π *integral[1]
            x = 1/2 * (1 + erf(σ[idx])) * exp(-abs(wc)/ε)
            if j == 1 
                push!(xs,x)
            end 
        end 
        if do_plot 
            plot!(plt,  
                σ,
                sqrt.(c_n),
                label="\$ c_{$(n)} \$", 
                ylabel="\$ |c_{n}_{-}| \$"
                )
        else
            return σ,c_n 
        end

    end 
    #display(σ)
    # plt = plot(
    #     σ,
    #     #τs,
    # #c_n,
    # sqrt.(c_n),
    # xlimit=(-10,10),
    # xlabel = "time", 
    # ylabel="\$ |c_{$(n)}| \$", 
    # title="transition histories",
    # fontfamily="Computer Modern", 
    # label="")
    #plot!(plt,σ,xs,xlimit=(minimum(τs),maximum(τs)),label="")
    if do_plot 
        plot!(plt,σ,xs,xlimit=(-5,5),label="")
    end
    
    display(plt)
end 

function plot_s_adiabatic_stuff(ns,τs,ε)
    wc = -im * pi/2 
    Δ = 1
    entropy = []
    σ = [[0 1; 1 0], [0 -im; im 0], [1 0; 0 -1]]
    plt = plot(xlabel="time",ylabel="fidelity",fontfamily="Computer Modern",
    label="",
    xlimits=(-5,5))
    for n in ns
        σs,c_n = adiabatic_c_n(n,τs;do_plot=false)
        ρs = []
        Fs = []
        for (idx,cn) in enumerate(c_n)
            
            ρ = 1/2 *(I + (0 *σ[1] + 0 *σ[2] +  (1 - (2*cn^2)) * σ[3])) 
            F = ρ[1][1]
            S = -tr(ρ*log(ρ))
            if isnan(S)
                S = 0
            end 
            push!(Fs,F)
            push!(ρs,S)
        end 
        plot!(plt,
            τs, 
            ρs,
            )
        # plot!(plt,
        #     τs, 
        #     Fs,
        #     )

    end
    # hline!(plt,
    # [log(2)],
    # label="max entropy"    
    # )
    display(plt)
    #angle between H and polarization vector
    #defined by cos(\theta) = c_n 
    
end 

function plot_s_adiabatic_stuff(ns,τs,ε)
    w_c = -im * pi/2 
    Δ = 1
    #5.2
    #ε= 0.3018
    #7.8
    #ε = 0.3018 
    #x = 1/2 * (1 + erf(σ[idx])) * exp(-abs(wc)/ε)
    entropy = []
    σ = [[0 1; 1 0], [0 -im; im 0], [1 0; 0 -1]]
    plt = plot(xlabel="time",ylabel="fidelity",fontfamily="Computer Modern",
    label="",
    xlimits=(-5,5))
    for n in ns
        σs,c_n = adiabatic_c_n(n,τs,ε;do_plot=false)

        ρs = []
        Fs = []
        for (idx,cn) in enumerate(c_n)
            if idx == 1 
                display(c_n[end])
            else
                break 
            end 
            w = quadgk(x -> sqrt(x^2+Δ^2),0,τs[idx])[1]
            H = sqrt(τs[idx]^2 +Δ^2)
            #H_pp(n,w_c,w,H,ε)
            z = real(H_pp(n,w_c,w,H,ε))
            xy = H_pm(n,w_c,w,H,ε)
            x,y = real(xy),imag(xy)
            H_s = normalize([x,y,z])

            ρ = 1/2 *(I + (0 *σ[1] + 0 *σ[2] +  (1 - (2*cn^2)) * σ[3])) 
            F = ρ[1][1]
            S = -tr(ρ*log(ρ))
            if isnan(S)
                S = 0
            end 
            push!(Fs,F)
            push!(ρs,S)
        end 
        break
        plot!(plt,
            τs, 
            ρs,
            )
        # plot!(plt,
        #     τs, 
        #     Fs,
        #     )

    end
    # hline!(plt,
    # [log(2)],
    # label="max entropy"    
    # )
    display(plt)
    #angle between H and polarization vector
    #defined by cos(\theta) = c_n 
    
end 

function plot_superA_zenith(sol,s)
    new_xs = [] 
    new_ys = []
    new_zs = [] 
    Bx = [] 
    By = [] 
    Bz = []
    diags = []
    δ = s.α
    ε = s.α
    τs = δ.*sol.t
    n = s.n 
    g = s.g
    Δ = s.g
    for τ in τs 

        w_c = -im * pi/2
    
        function H(g,τ)
            return sqrt(g^2 + τ^2)
        end
        function a_θ(n,w_c,w)
            return (1/(2*im))*((factorial(n) * im^(n+1))/2π) * (1/(w - w_c)^(n+1) - 1/(w - adjoint(w_c))^(n+1))
        end
        function b(n,w_c,w)
            return -2 * a_θ(n,w_c,w)
        end 
        w = quadgk(x -> 2*H(g,x),0,τ)[1]
        z_result = -ε^(n+2) * 4 * H(g,τ) * a_θ(n,w_c,w) * adjoint(b(0,w_c,w))
        xy_result = ε^(n+1) * 4*H(g,τ) * adjoint(a_θ(n,w_c,w)) * exp((im*w)/ε)

        diag = eigvals([z_result xy_result; conj(xy_result) -z_result])[1]
        push!(diags,abs.(diag))

        append!(Bx, [normalize([Δ,0,τ])][1][1])
        append!(By, [normalize([Δ,0,τ])][1][2])
        append!(Bz, [normalize([Δ,0,τ])][1][3])
        append!(new_xs,real(xy_result))
        append!(new_ys,imag(xy_result))
        append!(new_zs,z_result)
        #other_result = conj(result)
        #println("τ: $τ, result: $result")
    end
    polar_angle = zeros(length(sol.t))
    old_polar_angle = zeros(length(sol.t))
    for idx in eachindex(sol.t)
        if real(new_zs[idx]) == 0  
            polar_angle[idx] = π/2
            #polar_angle[idx] = π + real(atan(sqrt(new_xs[idx]^2 + new_ys[idx]^2)/(new_zs[idx]))) 
        else 
            polar_angle[idx] = real(atan(sqrt(new_xs[idx]^2 + new_ys[idx]^2)/(new_zs[idx])))
        end

        if real(Bz[idx]) < 0 
            old_polar_angle[idx] = π + real(atan(sqrt(Bx[idx]^2 + By[idx]^2)/(Bz[idx]))) 
        else 
            old_polar_angle[idx] = real(atan(sqrt(Bx[idx]^2 + By[idx]^2)/(Bz[idx])))
        end
    end
    # polar_angle = [real.(atan.(sqrt.(new_xs.^2 + new_ys.^2)./(new_z))) for new_z in new_zs] 
    # old_polar_angle = [real.(atan.(sqrt.(Bx.^2 + By.^2)./(Bz))) for Bz in Bz] 
    C = minimum(sqrt.(new_xs.^2 + new_ys.^2))
    D = minimum(real.(new_zs))
    # display(D/C)
    # display(new_ys)
    display("this is diags: $(diags)")
    display(maximum(diags))
    plt = plot(new_xs,new_ys)
    plt2 = plot(sol.t, real.(new_zs))
    display("this is new_zs: $new_zs")
    display("this is new_xs: $new_xs")
    plt3 = plot(sol.t, new_xs)
    plt4 = plot(sol.t, new_ys)
    # plt3 = plot(sol.t, rad2deg.(polar_angle), ylimits=(85,95))
    # plt4 = plot(sol.t,diags)
    #plt4 = plot(sol.t, rad2deg.(old_polar_angle), ylimits=(0,180))

    plot(plt, plt2, plt3, plt4, layout = (2, 2), legend = false)
end 

function decompose(v)
    x = 1/2 * tr([0 1; 1 0]*v)
    y = 1/2 * tr([0 -im; im 0]*v)
    z = 1/2 * tr([1 0; 0 -1]*v)

return real.([x,y,z])
end 

function entropy_calc(SX,SY,SZ,H)
    hdotp = dot(normalize(decompose(H)),[SX,SY,SZ])
    ρ = hdotp * normalize(decompose(H))
    S = 1/2*(I + (ρ[1]*σ[1] + ρ[2]*σ[2] + ρ[3]*σ[3]))

    if real(-tr(S*log(S))) > 0
        return real(-tr(S*log(S)))
    else 
        return 0 
    end
end

function entropy_map(f,ax,SX,SY,SZ, s::SimulationConfig)

    entropies = zeros(length(SX))
    for idx in eachindex(SX)
        first_sx,first_sy,first_sz,last_sx,last_sy,last_sz = SX[idx][1],SY[idx][1],SZ[idx][1],SX[idx][end],SY[idx][end],SZ[idx][end]
        first_H = [s.α*s.t_i 1; 1  -s.α*s.t_i]
        last_H = [s.α*s.t_f 1; 1  -s.α*s.t_f]
        entropies[idx] = entropy_calc(last_sx,last_sy,last_sz,last_H) - entropy_calc(first_sx,first_sy,first_sz,first_H) 
    end 
    xs = getindex.(SX,1)
    ys = getindex.(SY,1)

    h = fit(Histogram, (xs, ys), weights(entropies), nbins=(20,20))
    n = fit(Histogram, (xs, ys), nbins=(20,20)).weights
    #h = normalize(h, mode=:probability)
    # Modify the weights of the histogram
    edges_x = h.edges[1]  # x edges
    edges_y = h.edges[2]  # y edges

    # ws = reshape(h.weights, length(edges_x)-1, length(edges_y)-1)
    ws = h.weights

    # Calculate the bin centers
    x_centers = [(edges_x[i] + edges_x[i+1]) / 2 for i in 1:length(edges_x)-1]
    y_centers = [(edges_y[i] + edges_y[i+1]) / 2 for i in 1:length(edges_y)-1]

    # Iterate over bin centers to set specific conditions
    for i in 1:length(x_centers)
        for j in 1:length(y_centers)
            # Check if bin center lies outside the circle x^2 + y^2 = 1
            if x_centers[i]^2 + y_centers[j]^2 > 1
                # Set weight to NaN if the bin count is zero
                if ws[i, j] == 0
                    ws[i, j] = NaN
                end
            else 
                if ws[i,j] == 0
                    try 
                        ws[i, j] = ((ws[i+1,j]/n[i+1,j]) + (ws[i-1,j]/n[i-1,j]) + (ws[i,j+1]/n[i,j+1]) + (ws[i,j-1]/n[i,j-1])
                        + (ws[i+1,j+1]/n[i+1,j+1]) + (ws[i+1,j-1]/n[i+1,j-1]) + (ws[i-1,j+1]/n[i-1,j+1]) + (ws[i-1,j-1]/n[i-1,j-1])) / 8
                        continue 
                    catch 
                        ws[i, j] = NaN
                    end 
                end 
            end 
            ws[i,j] = ws[i,j]/n[i,j]
        end
    end

    weighs = vcat(ws...)
    weighs = weighs[.!isnan.(weighs)]
    weighs = weighs[.!isinf.(weighs)]
    cb_bins= minimum(weighs):0.001:maximum(weighs)
    colorbar_ticks=(cb_bins, string.(round.(cb_bins; digits=4)))
    colorbar_ticks = string.(-1:0.1:1)
    #ax = Axis(f[1, 1],aspect=1)
    xlims!(ax,(-1., 1.))
    ylims!(ax,(-1., 1.))
    println(edges_x)
    heatmap!(ax,edges_x,edges_y, transpose(h.weights),colorrange=(-0.6,0.6),colormap=cgrad(:vik))
            # aspect_ratio=:equal,
            # xlimits=(-1.,1.),
            # ylimits=(-1.,1.),
            # title="ϵ = $(sim.α)",
            # xlabel="x",
            # ylabel="y",
            # colorbar_ticks=colorbar_ticks,
            # colorbar_title="Entropy",
            # fontfamily="Computer Modern",
            # color=cgrad(:vik))
end 

function sphere_plot(ax,entropy_ax,SX,SY,SZ,s::SimulationConfig)
    n = 45
    u = range(0,stop=2*π,length=n);
    v = range(0,stop=π,length=n);
    sx = zeros(n,n); sy = zeros(n,n); sz = zeros(n,n)
    for i in 1:n
        for j in 1:n
            sx[i,j] = cos.(u[i]) * sin(v[j]);
            sy[i,j] = sin.(u[i]) * sin(v[j]);
            sz[i,j] = cos(v[j]);
        end
    end

    H_exp_x = zeros(length(s.t_i:s.Δt:s.t_f))
    H_exp_y = zeros(length(s.t_i:s.Δt:s.t_f))
    H_exp_z = zeros(length(s.t_i:s.Δt:s.t_f))
    entropy = zeros(length(s.t_i:s.Δt:s.t_f))
    for (idx,t) in enumerate(s.t_i:s.Δt:s.t_f)
        H_exp_x[idx],H_exp_y[idx],H_exp_z[idx] = normalize(decompose([s.α*t 1; 1 -s.α*t]))
        entropy[idx] = entropy_calc(SX[1][idx],SY[1][idx],SZ[1][idx],[s.α*t 1; 1 -s.α*t])
    end

    p2 = surface!(ax,
    sx,sy,sz,colormap=cgrad([:gray], 7), transparency = true, alpha = 0.25)
    scale = 1.5
    scale!(p2, scale,scale,scale) 
    #   colormap=cgrad([:purple], 7),
    #   shading=NoShading,
    # ) 
    #ax.show_axis = false
    #wireframe!(ax, sx, sy, sz)
    # p2 = scatter!(ax, exp_x,exp_y,exp_z, color=:red,markersize=3.5)
    # scale!(p2, s,s,s) 
    space = 1
    # if s.t_i == -50
    #     space = 100
    #     p2 = lines!(ax, SX[1][1:1:end],SY[1][1:1:1],SZ[1][1:1:end], color=:red,linewidth=2)
    #     scale!(p2, scale,scale,scale) 
    #     p2 = scatter!(ax, SX[1][5010:space:5010+400],SY[1][5010:space:5010+400],SZ[1][5010:space:5010+400], color=:red)
    #     scale!(p2, scale,scale,scale) 
    #     p2 = lines!(ax, H_exp_x[2:end],H_exp_y[2:end],H_exp_z[2:end], color=:blue,linewidth=2)
    #     scale!(p2, scale,scale,scale) 

    #     xlims!(-10,10)
    #     lines!(entropy_ax,s.t_i:s.Δt:s.t_f,entropy,color=:red)
    #     return 
    # end 
    p2 = lines!(ax, SX[1][1:space:end],SY[1][1:space:end],SZ[1][1:space:end], color=:red,linewidth=2)
    scale!(p2, scale,scale,scale) 
    # p2 = lines!(ax, SX[1][1:1:5010],SY[1][1:1:5010],SZ[1][1:1:5010], color=:red,linewidth=2)
    # scale!(p2, scale,scale,scale) 
    # p2 = lines!(ax, SX[1][5010:space:5010+400],SY[1][5010:space:5010+400],SZ[1][5010:space:5010+400], color=:red,linewidth=2)
    # scale!(p2, scale,scale,scale) 
    p2 = lines!(ax, H_exp_x[2:end],H_exp_y[2:end],H_exp_z[2:end], color=:blue,linewidth=2)
    scale!(p2, scale,scale,scale) 

    xlims!(-10,10)
    lines!(entropy_ax,s.t_i:s.Δt:s.t_f,entropy,color=:red)
end 
    # save("bloch_plot.png",f)
