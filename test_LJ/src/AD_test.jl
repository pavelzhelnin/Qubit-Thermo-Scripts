using ForwardDiff 
using LinearAlgebra 
using Plots

const σ = [[0 1; 1 0], [0 -im; im 0], [1 0; 0 -1]]
const σ_x = σ[1]
const σ_y = σ[2]
const σ_z = σ[3]

function get_V( v)
    m = (v[1]+v[4]) / 2
    p = v[1]*v[4] - v[2]*v[3] 
    lambda = [m+sqrt(m^2 - p),m-sqrt(m^2 -p)]
    eigvecs = [v[2] lambda[1]-v[1]],[lambda[2]-v[4] v[3]]
    eigvecs = eigvecs[1]/norm(eigvecs[1]), eigvecs[2]/norm(eigvecs[2])
    return vec(hcat(eigvecs[1],eigvecs[2]))
end

function H0(r::Real)
    #v = [0.52*r 1; 1 -0.52*r];#2
    #v = [0.314*r 1; 1 -0.314*r]; #5
    v = [0.89*r 1; 1 -0.89*r]; #1
end

function V0(r::Real)
    v = H0(r::Real)
    return get_V(v)
end

function H1(r::Real)

    V = Transpose( reshape(V0(r),(2,2)) )
    Hdiag = V* H0(r) * V'
    vdot = Transpose(reshape(ForwardDiff.derivative(V0,r),(2,2)))

    Hn = Hdiag + im*vdot*V'
    return reshape(Hn, (4,1))
end

function H2(r::Real)
   H1_ = reshape(H1(r),(2,2))
   function V1(r)
        get_V(H1(r))
   end 
   V1_ = reshape(V1(r),(2,2))
   H1diag = V1_ * H1_ * V1_' 
   v1dot = Transpose(reshape(ForwardDiff.derivative(V1,r),(2,2)))
   Hn = H1diag + im*v1dot*V1_'
end 

function decompose(v)
    x = 1/2 * tr([0 1; 1 0]*v)
    y = 1/2 * tr([0 -im; im 0]*v)
    z = 1/2 * tr([1 0; 0 -1]*v)

return real.([x,y,z])
end 

function Hn(r::Real,n)
    n == 0 && return H0(r)
    n == 1 && return Transpose(reshape(H1(r),(2,2)))

    Vn(r) = get_V(Hn(r,n-1))
    Vn_ = Transpose(reshape(Vn(r),(2,2)))
    VnDot = Transpose(reshape(ForwardDiff.derivative(Vn,r),(2,2)))
    Hdiag = Vn_*Hn(r,n-1)*Vn_'
    Hdiag + im*VnDot*Vn_'
end

function initialize_sphere(θmin, θmax, ϕmin, ϕmax, qubits)

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

#Q = norm(decompose(Hdiag))/norm(decompose(VnDot))

function Q_Hn(r::Real,n)

    function Q_return(r::Real, H)
        Vn(r) = get_V(H(r))
        Vn_ = Transpose(reshape(Vn(r),(2,2)))
        VnDot = Transpose(reshape(ForwardDiff.derivative(Vn,r),(2,2)))
        C = im*VnDot*Vn_'
        Hdiag = Vn_*H(r)*Vn_'
        println(C)
        norm(decompose(Hdiag))/norm(decompose(C))
    end 
    if n == 0 
        return "Not possible"
    elseif n == 1
        H(r) = H0(r)
        return Q_return(r,H)
    elseif n == 2
        function H1_clean(r::Real)
            return Transpose(reshape(H1(r),(2,2)))
        end 
        return Q_return(r,H1_clean)
    else 
        function Hn_clean(r::Real)
            return Hn(r,n-1)
        end
        return Q_return(r,Hn_clean)
    end
end

trange = -9.995:0.01:10.005
nn = 3
Hn_history = Hn.(trange,nn)
cs = initialize_sphere(0,π,0,2π,1000)
cs = [[cs[1][i];cs[2][i]] for i in eachindex(cs[1])]
plt = plot(aspect_ratio=:equal, title = "n = $nn", fontfamily="Computer Modern")
for c in cs 
    #U = exp(-im*Hn_history[1]*0.01)*[1;0]
    U = exp(-im*Hn_history[1]*0.01)*c
    exp_x,exp_y,exp_z = zeros(length(trange)),zeros(length(trange)),zeros(length(trange))
    exp_z[1] = real(U'*[1 0; 0 -1]*U)
    exp_y[1] = real(U'*[0 -im; im 0]*U)
    exp_x[1] = real(U'*[0 1; 1 0]*U)
    entropy = zeros(length(trange))
    for i in 2:length(trange)
        U = exp(-im*Hn_history[i]*0.01)*U
        exp_z[i] = real(U'*[1 0; 0 -1]*U)
        exp_y[i] = real(U'*[0 -im; im 0]*U)
        exp_x[i] = real(U'*[0 1; 1 0]*U)
        hdotp = dot(normalize(decompose(Hn_history[i])),[exp_x[i],exp_y[i],exp_z[i]])
        ρ = hdotp * normalize(decompose(Hn_history[i]))
        S = 1/2*(I + (ρ[1]*σ[1] + ρ[2]*σ[2] + ρ[3]*σ[3]))
        if real(-tr(S*log(S))) > 0
            entropy[i] = real(-tr(S*log(S)))
        else 
            entropy[i] = 0
        end
    end
    θ = 2*acos(c[1])
    ϕ = angle(c[2]) / sin(θ/2)
    x = real(sin(θ)*cos(ϕ))
    y = real(sin(θ)*sin(ϕ))
    z = cos(θ)
    scatter!(plt,[x],[y],zcolor=entropy[end],label="",color=:oslo)
end 
display(plt)
entropy
scatter(trange[2:end],entropy,ylimits=(0,0.05),title="\$H_$nn\$ Entropy History",fontfamily="Computer Modern", xlabel="time", ylabel="entropy", legend=false)
scatter(trange,exp_z,ylimits=(0.9995,1.001),legend=false)
U'*[1 0; 0 -1]*U
U
Qn_history = Q_Hn.(trange,2)

println(exp(-pi/2*(1/0.52)))
Qhn_mins = []
for n in 1:8
    push!(Qhn_mins, minimum(Q_Hn.(trange,n)))
end

Qhn_mins
plot(trange,log10.(Qn_history),ylimits=(1,7),marker=:circle,legend=false)

plt = scatter(;legend=false)
for (idx,hist) in enumerate(Hn_history)
    x,y,z = decompose(hist)
    println(norm(decompose(hist)))
    scatter!(plt, [trange[idx]],[z])
end
display(plt)
#------

function diag_matrix(r::Real)
    v = [r 1;1 -r]; m = (v[1]+v[4]) / 2
    p = v[1]*v[4] - v[2]*v[3] 
    lambda = [m+sqrt(m^2 - p),m-sqrt(m^2 -p)]
    eigvecs = [v[2] lambda[1]-v[1]],[lambda[2]-v[4] v[3]]
    eigvecs = eigvecs[1]/norm(eigvecs[1]), eigvecs[2]/norm(eigvecs[2])
    return vec(hcat(eigvecs[1],eigvecs[2]))
    end

V = Transpose(reshape(diag_matrix(1),(2,2)))
Hdiag = V*[1 1;1 -1]*V'
vdot = Transpose(reshape(ForwardDiff.derivative(diag_matrix,2),(2,2)))
Hn = Hdiag + im*vdot*V'

using ForwardDiff: derivative

function W(n, t, p)
    n == 0 && return W₀(t, p)
    n == 1 && return W₁(t, p)
    2t*W(n-1, t, p) - 2∂W(n-1, t, p) - W(n-2, t, p)
end

function Hn(n, t, p)
    n == 0 && return W₀(t, p)
    n == 1 && return W₁(t, p)
    V(n-1, t)*Hn(n-1, t)*adjoint(V(n-1, t)) + im*∂V(n-1, t)*adjoint(V(n-1, t, p))
end

function ∂W(n, t, p)
	f(q) = W(n, t, q)
	derivative(f, p)
end

function ∂V(n, t)
	f(q) = W(n, t, q)
	derivative(f, p)
end

function W₀(t, p) # Implements the example equation $W_0$ above
	
end

function W₁(t, p)
	t < -1 && return complex(0.)
	t ≤ 1 && return -(1+exp(p+p*t)*(p-1)+p*t)/p^2
	-2*exp(p*t)*(p*cosh(p)-sinh(p))/p^2
end

t = .5; p = -.1;
W(2, t, p), W(3, t, p), W(4, t, p), W(5, t, p)