module test_LJ 

#using DifferentialEquations
using LinearAlgebra
using Statistics
using DifferentialEquations
using QuadGK 
using ForwardDiff 
# using Groebner 
# using Symbolics

export SimulationConfig, Deschamps_H, lz!,uniform_initialize_sphere,H,σ,many_run_sim, one_run_sim, one_run_superadiabatic_sim

const σ = [[0 1; 1 0], [0 -im; im 0], [1 0; 0 -1]]
const σ_x = σ[1]
const σ_y = σ[2]
const σ_z = σ[3]

@Base.kwdef mutable struct SimulationConfig
    qubits::Int64 = 100
    t_f::Float64 = 10
    t_i::Float64 = -10
    Δt::Float64 = 0.01
    θmin::Float64 = 0.0
    θmax::Float64 = π/16
    ϕmin::Float64 = 0.0
    ϕmax::Float64 = 2π
    α::Float64 = 1
    g::Float64 = 1
    n::Int64 = 1 
    cg_factor::Int64 = 16
end

function Base.show(io::IO, s::SimulationConfig)
    print(io, 
        """ 
        Plotting Config: 
        qubits: $(s.qubits),
        t_f: $(s.t_f),
        t_i: $(s.t_i),
        Δt: $(s.Δt),
        θmin: $(s.θmin),
        θmax: $(s.θmax),
        ϕmin: $(s.ϕmin),
        ϕmax: $(s.ϕmax),
        α: $(s.α),
        g: $(s.g),
        n: $(s.n),
        cg_factor: $(s.cg_factor)
        """
    )
end

function lz!(du, u, p, t)
    du[1] = -im * u[1] * p[1] * t  - im * u[2] * p[2]
    du[2] = -im * u[1] * p[2] + im * u[2] * p[1] * t 
end

function H(α,g,t)
    return [α*t g; g -α*t]
end 

function super_adiabatic_lz!(du, u, p, t)

    ε = p[1]
    g = p[2] 
    n = Int64(p[3])

    τ = t*ε

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

    du[1] = (1/(im))*(-ε^(n+2) * 4 * H(g,τ) * a_θ(n,w_c,w) * adjoint(b(0,w_c,w)) * u[1]) - (1/(im))*(ε^(n+1) * 4*H(g,τ) * adjoint(a_θ(n,w_c,w)) * exp((im*w)/ε) * u[2])
    du[2] = (1/(im))*(ε^(n+2) * 4 * H(g,τ) * adjoint(a_θ(n,w_c,w)) * b(0,w_c,w) * u[2]) - (1/(im))*(ε^(n+1) * 4*H(g,τ) * a_θ(n,w_c,w) * exp((-im*w)/ε) * u[1])
end

# function lz!(du, u, p, t)
#     du[1] = -im * u[1] * cos(t) * p[1]  + sin(t) * u[2] * p[2]
#     du[2] = -u[1] * sin(t) * p[2] + im * cos(t) * u[2] * p[1] 
# end

function random_initialize_sphere(θmin, θmax, ϕmin, ϕmax, qubits)

    n = qubits

    u_max = ϕmax/(2*pi)
    u_min = ϕmin/(2*pi)
    v_min = (cos(θmin)+1)/2
    v_max = (cos(θmax)+1)/2

    u = u_min .+ (u_max - u_min)*rand(n)
    v = v_min .+ (v_max - v_min)*rand(n)

    starting_ϕ = u.*2π
    starting_θ = acos.(1 .- (2).*v)

    c1 = cos.(starting_θ ./ 2)
    c2 = sin.(starting_θ ./ 2) .* exp.(im .* starting_ϕ)

    return c1,c2
end

function uniform_initialize_sphere(θmin, θmax, ϕmin, ϕmax, qubits)
    n = qubits

    u_max = ϕmax/(2*pi)
    u_min = ϕmin/(2*pi)
    v_min = (cos(θmin)+1)/2
    v_max = (cos(θmax)+1)/2

    function meshgrid_nbins(xmin, xmax, ymin, ymax, n::Int)

        xrange = xmax - xmin
        yrange = ymax - ymin
        aspect = xrange / yrange
    
        ny = Int(round(sqrt(n / aspect)))
        nx = Int(round(n / ny))
    
        x = range(xmin, xmax; length=nx)
        y = range(ymin, ymax; length=ny)
    
        X = repeat(x', ny, 1)
        Y = repeat(y, 1, nx)
    
        return X, Y, x, y
    end

    uns,vns,_,_ = meshgrid_nbins(0,1,0,1,n)
    u = u_min .+ (u_max - u_min)*uns
    v = v_min .+ (v_max - v_min)*vns
    starting_ϕ = u.*2π
    starting_θ = acos.(1 .- (2).*v)
    c1 = cos.(starting_θ ./ 2)
    c2 = sin.(starting_θ ./ 2) .* exp.(im .* starting_ϕ)

    return c1,c2
end


function many_run_sim(s::SimulationConfig)
    n = s.qubits
    c1s,c2s = random_initialize_sphere(s.θmin, s.θmax, s.ϕmin, s.ϕmax, n)
    SX, SY, SZ = [], [], []
    times = [] 
    for (idx,(c1,c2)) in enumerate(zip(c1s,c2s))
        u0 = ComplexF64[c1; c2;]
        tspan = (s.t_i, s.t_f)
        p = [s.α, s.g]
        prob = ODEProblem(lz!, u0, tspan, p; reltol=1e-8)
        sol = solve(prob, saveat = s.t_i:s.Δt:s.t_f, maxiters=1e8)
        sx = []
        sy = []
        sz = []

        if idx == 1
            push!(times,sol.t)
        end

        for (idx,t) in enumerate(sol.t)
            ψ = [sol[1,idx],sol[2,idx]]
            push!(sx, real(ψ'*σ_x*ψ))
            push!(sy, real(ψ'*σ_y*ψ))
            push!(sz, real(ψ'*σ_z*ψ))
        end
        push!(SX,sx)
        push!(SY,sy)
        push!(SZ,sz)
    end
    return SX,SY,SZ,times
end 

function one_run_sim(s::SimulationConfig)
    if s.qubits != 1
        println("Please set qubits to 1 for this function")
        return
    end
    c1,c2 = random_initialize_sphere(s.θmin, s.θmax, s.ϕmin, s.ϕmax, 1)
    u0 = ComplexF64[c1; c2;]
    tspan = (s.t_i, s.t_f)
    p = [s.α, s.g]
    prob = ODEProblem(lz!, u0, tspan, p; reltol=1e-8)
    sol = solve(prob, saveat = s.t_i:s.Δt:s.t_f, maxiters=1e8)
    return sol
end

function one_run_superadiabatic_sim(s::SimulationConfig)
    u0 = ComplexF64[0;-1;]
    tspan = (s.t_i, s.t_f)
    p = [s.α, s.g, s.n]
    prob = ODEProblem(super_adiabatic_lz!, u0, tspan, p; reltol=1e-9)
    sol = solve(prob, saveat = s.t_i:s.Δt:s.t_f)
    return sol
end # module test_LJ

# function super_a_wavefunction(α,g,n,t)
 
#     function H(τ)
#         return [τ 1; 1 -τ]
#     end
#     # function eigenvectors_function(τ)
#     #     eigenvalues,_ = eigen(H(τ))
#     #     return eigenvalues 
#     # end

#     function U(θ)
#         λs = [[-sin.(θ/2); cos.(θ/2);],[-sin.(0/2) cos.(0/2)],[cos.(θ/2) ;sin.(θ/2)],[cos.(0/2) sin.(0/2);]]
#         sum = zeros((2,2))
#         for λ in 1:2:length(λs)
#             λ1 = λs[λ]
#             λ2 = λs[λ+1]
#             sum += λ1 * λ2
#         end
#         return sum 
#     end

#     @variables a d λ
#     A = [a 1; 1 d]
#     char_poly = det(A-λ*I) .~ 0 
#     eigenvals = symbolic_solve(char_poly,[λ])
#     vecs = [] 
#     display(Symbolics.simplify(eigenvals[1]))
#     display((typeof(eigenvals[1])))
#     for ev in eigenvals 
#         B = A - ev*[1 0; 0 1]

#         B = Symbolics.simplify(B)

#         @variables x y 

#         Eq1 = B[1,1]*x + B[1,2]*y ~0 
#         Eq2 = B[2,1]*x + B[2,2]*y ~0

#         solution = symbolic_solve([Eq1,Eq2],[x,y])
#         push!(vecs,solution)
#     end 


#     display(vecs)
#     #  [-sin.(θ/2) cos.(θ/2)]
#     #[cos.(θ/2) sin.(θ/2)]
#     #return [-sin.(θ/2) cos.(θ/2); cos.(θ/2) sin.(θ/2)]
#     θ = acos(τ/sqrt(τ^2 + 1))


#     # V = U(θ)

    

#     # evs = reshape(ForwardDiff.jacobian(x -> U(x), [θ]),(2,2))

#     # D = adjoint(V) * H(τ) * V
#     # C = -im * adjoint(V) * evs 
#     # new_H = D + C
#     # display(new_H)
#     # eigenvalues,eigenvectors = eigen(new_H)


#     return 0 
# end
# end
end 