using ForwardDiff 
using LinearAlgebra 
using CairoMakie
using StatsBase

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
    #v = [0.314*r 1; 1 -0.314*r];#2
    #v = [0.34*r 1; 1 -0.34*r]; #5
    #v = [0.314*r 1; 1 -0.314*r]; #5
    #v = [0.4*r 1; 1 -0.4*r]; #4
    v = [0.89*r 1; 1 -0.89*r]; #1
    #v = [20*r 1; 1 -20*r]; #3
    #v = [3.4*r 1; 1 -3.4*r]; #2
    #v = [1.5*r 1; 1 -1.5*r];
    #v = [5*r 1; 1 -5*r];
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
        #println(C)
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

δt = 0.1
trange = -10.005:δt:10.005
nn = 2
Hn_history = Hn.(trange,nn)
cs = initialize_sphere(15pi/16,15pi/16,0,2π,1000)
cs = [[cs[1][i];cs[2][i]] for i in eachindex(cs[1])]
#plt = plot(aspect_ratio=:equal, xlimits = (-1,1),ylimits=(-1,1),title = "n = $nn", fontfamily="Computer Modern",label="")
#plt2 = plot(title= "Entropy history", fontfamily="Computer Modern",label="")
cs = [[0;1]]
#plt = plot()
xs = Float64[] 
ys = Float64[]
zs = Float64[] 
toplot = true 
entropies = Float64[]
#f = Figure()
f = Figure(; size = (600, 400), padding = 0)
for (idxx,nn) in enumerate([1,2,3])
    jdx = 3  # if idxx == 2
    #     f = Figure(; size = (600, 400), padding = 0)
    # end 
    if idxx > 1
        break 
    end 
    δt = 0.01
    if jdx < 3
        trange= -500.005:δt:500.005
    elseif jdx < 4
        trange = -10.005:δt:10.005
    else 
        trange = -20.005:δt:20.005
    end
    #jdx = idxx
    #Hn_history = Hn.(trange,nn)
    Hn_history = Hn.(trange,0)
    for c in cs 
        #U = exp(-im*Hn_history[1]*0.01)*[1;0]
        U = exp(-im*Hn_history[1]*δt)*c
        exp_x,exp_y,exp_z = zeros(length(trange)),zeros(length(trange)),zeros(length(trange))
        exp_z[1] = real(U'*[1 0; 0 -1]*U)
        exp_y[1] = real(U'*[0 -im; im 0]*U)
        exp_x[1] = real(U'*[0 1; 1 0]*U)
        entropy = zeros(length(trange))

        hdotp = dot(normalize(decompose(Hn_history[1])),[exp_x[1],exp_y[1],exp_z[1]])
        ρ = hdotp * normalize(decompose(Hn_history[1]))
        S = 1/2*(I + (ρ[1]*σ[1] + ρ[2]*σ[2] + ρ[3]*σ[3]))
        if real(-tr(S*log(S))) > 0
            entropy[1] = real(-tr(S*log(S)))
        else 
            entropy[1] = 0
        end
        
        H_exp_x,H_exp_y,H_exp_z = zeros(length(trange)),zeros(length(trange)),zeros(length(trange))
        for i in 2:length(trange)
            H_exp_x[i], H_exp_y[i], H_exp_z[i]= normalize(decompose(Hn_history[i]))
            U = exp(-im*Hn_history[i]*δt)*U
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
        z = real(cos(θ))

        push!(xs,x)
        push!(ys,y)
        push!(zs,z)
        push!(entropies,entropy[end]-entropy[1])

        z = cos(θ)

        # if toplot 
        #     f = Figure()
        #     ax = Axis(f[1,1],aspect=1)
        #     xlims!(ax,(-1,1))
        #     ylims!(ax,(-1,1))

        #     scatter!(ax,exp_x,exp_y)
        #     ax = Axis(f[2,1])
        #     xlims!()
        #     scatter!(ax,trange,entropy)
        #     display(f)
        # end

        if toplot 
            n = 30
            # if idxx == 1
            if idxx > 1
                ticks = [-1.0,0.0,1.0]
                n = 20
                m = 15
                u = range(0,stop=2*π,length=m);
                v = range(0,stop=π,length=n);
                sx = zeros(m,n); sy = zeros(m,n); sz = zeros(m,n)
                for i in 1:m
                    for j in 1:n
                        sx[i,j] = cos.(u[i]) * sin(v[j]);
                        sy[i,j] = sin.(u[i]) * sin(v[j]);
                        sz[i,j] = cos(v[j]);
                    end
                end
            # elseif nn == 1
            elseif nn == 5
                u = range(0,stop=2*π,length=n);
                v = range(0,stop=π/2,length=n);
                #v = range(0,stop=π/32,length=n);
            else
                ticks = [-1.0,0.0,1.0]
                n = 20
                m = 15
                u = range(0,stop=2*π,length=m);
                v = range(0,stop=π,length=n);
                sx = zeros(m,n); sy = zeros(m,n); sz = zeros(m,n)
                for i in 1:m
                    for j in 1:n
                        sx[i,j] = cos.(u[i]) * sin(v[j]);
                        sy[i,j] = sin.(u[i]) * sin(v[j]);
                        sz[i,j] = cos(v[j]);
                    end
                end
            end
            ticks = [-1.0,0.0,1.0]
            #f = Figure()
            ax = Axis3(f[1,jdx],aspect=(1,1,1),xlabel="",ylabel="",zlabel="")
            hidedecorations!(ax); hidespines!(ax);
            #ax = Axis(f[1,idxx],aspect=1,xlabel="",ylabel="",xticks=ticks,yticks=ticks)
            ax.xticklabelsize = 18
            ax.yticklabelsize = 18   
            ax.xlabelsize = 20         
            ax.ylabelsize = 20 
            # ax.xticklabelsize = 16
            # ax.yticklabelsize = 16 
            # hidexdecorations!(ax) 
            # hideydecorations!(ax)
            # hidespines!(ax)
            # ax = LScene(f[1, 1],scenekw=(camera = cam3d!,scale = Vec3f(2),))
            # zoom!(ax.scene, cameracontrols(ax.scene), 10)
            # update_cam!(ax.scene, cameracontrols(ax.scene))
            #ax.show_axis=false
            # ax.xticklabelsize = 0
            # ax.yticklabelsize = 0
            # ax.zticklabelsize = 0
            # ax.show_axis = false

            #NOTES 
            #remove grid
            #send an email to Luke 
            #include z axis arrow? 
            #For figure 2 parallel the figures already there but with a 2d projection for higher order frames

            s = 1.6 
            #s = 1
            p2 = surface!(ax,
            sx,sy,sz,colormap=cgrad([:gray], 7), transparency = true, alpha = 0.2)
            scale!(p2, s,s,s) 
            #   colormap=cgrad([:purple], 7),
            #   shading=NoShading,
            # ) 
            #ax.show_axis = false
            #wireframe!(ax, sx, sy, sz)
            if jdx < 3 
            #if jdx > 4
                p2 = scatter!(ax, exp_x,exp_y,exp_z, color=:red,markersize=3.5)
                scale!(p2, s,s,s) 
            else 
                p2 = lines!(ax, exp_x,exp_y,exp_z,color=:red,linewidth=2)
                
                scale!(p2, s,s,s)
            end 
            # println(exp_x)
            # p2 = scatter!(ax, exp_x,exp_y, color=:red,markersize=2)
            p2 = lines!(ax, H_exp_x[2:end], H_exp_y[2:end], H_exp_z[2:end], color=:blue,linewidth=2)
            scale!(p2, s,s,s) 
            if jdx == 1
                ax = Axis(f[2,jdx],aspect=1,xlabel="Time", ylabel=L"\Delta S",)
            else 
                ax = Axis(f[2,jdx],aspect=1,xlabel="Time", ylabel="")
            end 

            ax.xticklabelsize = 18
            ax.yticklabelsize = 18   
            ax.xlabelsize = 20         
            ax.ylabelsize = 20 

            #ylims!(-0.0075,0.2)
            #Label(f[2,idxx, Top()], halign = :left, L"\times 10^{-1}")
            xlims!(-10,10)
            lines!(ax,trange,entropy,color=:red)
            c_n = exp(-(π/2)*(1^2) / H0(1)[1]) 
            # final_vector = [0,0,cos(theta)-0.002]
            final_vector = [0,0,1-2*c_n^2]
            #display(final_vector)
            α = 1 - 2*c_n^2
            X,Z = 1, H0(1)[1]*1e3
            final_vector = zeros(3)
            final_vector[3] = (Z*α * sqrt((Z^2 * α^2) -(X^2 +Z^2)*(α^2 -X^2)))/(X^2 +Z^2)
            final_ρ = 1/2*(I + (final_vector[1]*σ[1] + final_vector[2]*σ[2] + final_vector[3]*σ[3]))
            final_S = -tr(final_ρ*log(final_ρ))
    
            hlines!(ax,real.(final_S), color=:black, linestyle=:dash, label = "Final entropy", linewidth=2)
            # if jdx == 3 
            #     ax = Axis(f[2, jdx],aspect =1, yticks = [-1.0,0.0,1.0], ylabelcolor=:blue, ylabel= "Z", yticklabelcolor = :blue, yaxisposition = :right)
            # else
            #     ax = Axis(f[2, jdx],aspect =1, yticksvisible = false, yticklabelcolor = :blue, yaxisposition = :right)
            #     hideydecorations!(ax, grid = false)
            # end
            # ax.xticklabelsize = 16
            # ax.yticklabelsize = 16 
            # ax.xlabelsize = 16        
            # ax.ylabelsize = 16
            # xlims!(-10,10)
            # ylims!(-1.05,1.05)
            # lines!(ax,trange,H_exp_z,color=:blue)

            # save("bloch_plot.png",f)
            # display(f)
            end
        #scatter!(plt,[x],[y],zcolor=entropy[end]-entropy[1],label="",color=:oslo)
        #scatter!(plt,exp_x,exp_y,zcolor=trange,label="")
        #scatter!(plt2,trange,entropy,label="")
    end
end
rowgap!(f.layout, 1)
colgap!(f.layout, 0)
rowsize!(f.layout, 1, 160)
rowsize!(f.layout, 2, 160)
colsize!(f.layout, 1, 160)
colsize!(f.layout, 2, 160)
colsize!(f.layout, 3, 160)
resize_to_layout!(f)
display(f)
#save("superadiabatic_bloch_2.png",f)
save("3d_bloch.png",f)

for (idxx,nn) in enumerate([1,2,3])
    jdx = 3   # if idxx == 2
    #     f = Figure(; size = (600, 400), padding = 0)
    # end 
    if idxx > 3
        break 
    end 
    δt = 0.01
    if jdx < 3
        trange= -500.005:δt:500.005
    elseif jdx < 4
        trange = -10.005:δt:10.005
    else 
        trange = -20.005:δt:20.005
    end
    jdx = idxx
    Hn_history = Hn.(trange,nn)
    #Hn_history = Hn.(trange,0)
    for c in cs 
        #U = exp(-im*Hn_history[1]*0.01)*[1;0]
        U = exp(-im*Hn_history[1]*δt)*c
        exp_x,exp_y,exp_z = zeros(length(trange)),zeros(length(trange)),zeros(length(trange))
        exp_z[1] = real(U'*[1 0; 0 -1]*U)
        exp_y[1] = real(U'*[0 -im; im 0]*U)
        exp_x[1] = real(U'*[0 1; 1 0]*U)
        entropy = zeros(length(trange))

        hdotp = dot(normalize(decompose(Hn_history[1])),[exp_x[1],exp_y[1],exp_z[1]])
        ρ = hdotp * normalize(decompose(Hn_history[1]))
        S = 1/2*(I + (ρ[1]*σ[1] + ρ[2]*σ[2] + ρ[3]*σ[3]))
        if real(-tr(S*log(S))) > 0
            entropy[1] = real(-tr(S*log(S)))
        else 
            entropy[1] = 0
        end
        
        H_exp_x,H_exp_y,H_exp_z = zeros(length(trange)),zeros(length(trange)),zeros(length(trange))
        for i in 2:length(trange)
            H_exp_x[i], H_exp_y[i], H_exp_z[i]= normalize(decompose(Hn_history[i]))
            U = exp(-im*Hn_history[i]*δt)*U
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
        z = real(cos(θ))

        push!(xs,x)
        push!(ys,y)
        push!(zs,z)
        push!(entropies,entropy[end]-entropy[1])

        z = cos(θ)

        # if toplot 
        #     f = Figure()
        #     ax = Axis(f[1,1],aspect=1)
        #     xlims!(ax,(-1,1))
        #     ylims!(ax,(-1,1))

        #     scatter!(ax,exp_x,exp_y)
        #     ax = Axis(f[2,1])
        #     xlims!()
        #     scatter!(ax,trange,entropy)
        #     display(f)
        # end

        if toplot 
            n = 30
            # if idxx == 1
            if idxx > 1
                ticks = [-1.0,0.0,1.0]
                n = 5
                m = 30
                u = range(0,stop=2*π,length=m);
                v = range(0,stop=π/2,length=n);
                sx = zeros(m,n); sy = zeros(m,n); sz = zeros(m,n)
                for i in 1:m
                    for j in 1:n
                        sx[i,j] = cos.(u[i]) * sin(v[j]);
                        sy[i,j] = sin.(u[i]) * sin(v[j]);
                        sz[i,j] = cos(v[j]);
                    end
                end
            # elseif nn == 1
            elseif nn == 5
                u = range(0,stop=2*π,length=n);
                v = range(0,stop=π/2,length=n);
                #v = range(0,stop=π/32,length=n);
            else
                ticks = [-1.0,0.0,1.0]
                n = 5
                m = 30
                u = range(0,stop=2*π,length=m);
                v = range(0,stop=π/2,length=n);
                sx = zeros(m,n); sy = zeros(m,n); sz = zeros(m,n)
                for i in 1:m
                    for j in 1:n
                        sx[i,j] = cos.(u[i]) * sin(v[j]);
                        sy[i,j] = sin.(u[i]) * sin(v[j]);
                        sz[i,j] = cos(v[j]);
                    end
                end
            end
            ticks = [-1.0,0.0,1.0]
            #f = Figure()
            # ax = Axis3(f[1,jdx],aspect=(1,1,1),xlabel="",ylabel="",zlabel="")
            #hidedecorations!(ax); hidespines!(ax);
            ax = Axis(f[1,idxx],aspect=1,xlabel="",ylabel="",xticks=ticks,yticks=ticks)
            ax.xticklabelsize = 18
            ax.yticklabelsize = 18   
            ax.xlabelsize = 20         
            ax.ylabelsize = 20 
            # ax.xticklabelsize = 16
            # ax.yticklabelsize = 16 
            # hidexdecorations!(ax) 
            # hideydecorations!(ax)
            # hidespines!(ax)
            # ax = LScene(f[1, 1],scenekw=(camera = cam3d!,scale = Vec3f(2),))
            # zoom!(ax.scene, cameracontrols(ax.scene), 10)
            # update_cam!(ax.scene, cameracontrols(ax.scene))
            #ax.show_axis=false
            # ax.xticklabelsize = 0
            # ax.yticklabelsize = 0
            # ax.zticklabelsize = 0
            # ax.show_axis = false

            #NOTES 
            #remove grid
            #send an email to Luke 
            #include z axis arrow? 
            #For figure 2 parallel the figures already there but with a 2d projection for higher order frames

            s = 1.6 
            s = 1
            p2 = surface!(ax,
            sx,sy,sz,colormap=cgrad([:gray], 7), transparency = true, alpha = 0.2)
            scale!(p2, s,s,s) 
            #   colormap=cgrad([:purple], 7),
            #   shading=NoShading,
            # ) 
            #ax.show_axis = false
            #wireframe!(ax, sx, sy, sz)
            # if jdx < 3 
            if jdx > 4
                p2 = scatter!(ax, exp_x,exp_y,exp_z, color=:red,markersize=3.5)
                scale!(p2, s,s,s) 
            else 
                p2 = lines!(ax, exp_x,exp_y,exp_z,color=:red,linewidth=2)
                
                scale!(p2, s,s,s)
            end 
            # println(exp_x)
            # p2 = scatter!(ax, exp_x,exp_y, color=:red,markersize=2)
            p2 = lines!(ax, H_exp_x[2:end], H_exp_y[2:end], H_exp_z[2:end], color=:blue,linewidth=2)
            scale!(p2, s,s,s) 
            if jdx == 1
                ax = Axis(f[2,jdx],aspect=1,xlabel="Time", ylabel=L"\Delta S",)
            else 
                ax = Axis(f[2,jdx],aspect=1,xlabel="Time", ylabel="")
            end 

            ax.xticklabelsize = 18
            ax.yticklabelsize = 18   
            ax.xlabelsize = 20         
            ax.ylabelsize = 20 

            ylims!(-0.0075,0.2)
            #Label(f[2,idxx, Top()], halign = :left, L"\times 10^{-1}")
            xlims!(-10,10)
            lines!(ax,trange,entropy,color=:red)
            c_n = exp(-(π/2)*(1^2) / H0(1)[1]) 
            # final_vector = [0,0,cos(theta)-0.002]
            final_vector = [0,0,1-2*c_n^2]
            #display(final_vector)
            α = 1 - 2*c_n^2
            X,Z = 1, H0(1)[1]*1e3
            final_vector = zeros(3)
            final_vector[3] = (Z*α * sqrt((Z^2 * α^2) -(X^2 +Z^2)*(α^2 -X^2)))/(X^2 +Z^2)
            final_ρ = 1/2*(I + (final_vector[1]*σ[1] + final_vector[2]*σ[2] + final_vector[3]*σ[3]))
            final_S = -tr(final_ρ*log(final_ρ))
    
            hlines!(ax,real.(final_S), color=:black, linestyle=:dash, label = "Final entropy", linewidth=2)
            if jdx == 3 
                ax = Axis(f[2, jdx],aspect =1, yticks = [-1.0,0.0,1.0], ylabelcolor=:blue, ylabel= "Z", yticklabelcolor = :blue, yaxisposition = :right)
            else
                ax = Axis(f[2, jdx],aspect =1, yticksvisible = false, yticklabelcolor = :blue, yaxisposition = :right)
                hideydecorations!(ax, grid = false)
            end
            ax.xticklabelsize = 16
            ax.yticklabelsize = 16 
            ax.xlabelsize = 16        
            ax.ylabelsize = 16
            xlims!(-10,10)
            ylims!(-1.05,1.05)
            lines!(ax,trange,H_exp_z,color=:blue)

            # save("bloch_plot.png",f)
            # display(f)
            end
        #scatter!(plt,[x],[y],zcolor=entropy[end]-entropy[1],label="",color=:oslo)
        #scatter!(plt,exp_x,exp_y,zcolor=trange,label="")
        #scatter!(plt2,trange,entropy,label="")
    end
end
# rowgap!(f.layout, 1)
# colgap!(f.layout, 0)
rowsize!(f.layout, 1, 160)
rowsize!(f.layout, 2, 160)
colsize!(f.layout, 1, 160)
colsize!(f.layout, 2, 160)
colsize!(f.layout, 3, 160)
resize_to_layout!(f)
display(f)
save("superadiabatic_bloch_2.png",f)

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
#save("bloch_plot.pdf",f)
# rowsize!(f.layout, 2, 100)
# rowsize!(f.layout, 1, 100)
display(f)
plot(plt,xlimit=(-1,1),ylimit=(-1,1),aspect_ratio=:equal) 
h = fit(Histogram, (xs, ys), weights(entropies), nbins=(40))
n = fit(Histogram, (xs, ys), nbins=(40)).weights
#h = normalize(h, mode=:probability)
# Modify the weights of the histogram
edges_x = h.edges[1]  # x edges
edges_y = h.edges[2]  # y edges
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
        end
        ws[i,j] = ws[i,j]/n[i,j]
    end
end

weighs = vcat(ws...)
weighs = weighs[.!isnan.(weighs)]

cb_bins= minimum(weighs):0.001:maximum(weighs)
colorbar_ticks=(cb_bins, string.(round.(cb_bins; digits=4)))
heatmap(h.edges[1], h.edges[2], ws,
        aspect_ratio=:equal,
        xlimits=(-1,1),
        ylimits=(-1,1),
        title="ϵ = 5, superadiabatic frame n = $nn",
        xlabel="x",
        ylabel="y",
        colorbar_ticks=colorbar_ticks,
        colorbar_title="Entropy",
        fontfamily="Computer Modern",
        color=cgrad(:vik))
H_exp_x,H_exp_y,H_exp_z = zeros(length(trange)),zeros(length(trange)),zeros(length(trange))
for i in 1:length(trange)
    H_exp_x[i],H_exp_y[i],H_exp_z[i] = normalize(decompose(Hn_history[i]))
end
scatter(H_exp_x,H_exp_y,label="",color=:red)
scatter(trange,H_exp_z,label="")

scatter!(plt,H_exp_x,H_exp_y,label="",color=:red)
#hline!(plt2,[log(2)],label="max entropy")
#display(plot(plt,plt2,layout=(1,2),size=(800,400)))
display(plot(plt,layout=(1,1),size=(800,400)))
exp(-pi/2*(1/100))
entropy
scatter(trange[2:end],entropy,ylimits=(0,0.05),title="\$H_$nn\$ Entropy History",fontfamily="Computer Modern", xlabel="time", ylabel="entropy", legend=false)
scatter(trange,exp_z,ylimits=(0.9995,1.001),legend=false)
U'*[1 0; 0 -1]*U
U

δt = 0.001
trange = -500.005:δt:500.005

nnn = 6
Qn_history = Q_Hn.(trange,nnn)
Qn_history

argmin(Qn_history)
trange[argmin(Qn_history)]
println(exp(-pi/2*(1/0.314)))
Qhn_mins = []
for n in 1:1:8
    trange= -10.005:δt:10.005
    if n > 6
        trange= -1.00:δt:-0.001
    end
    push!(Qhn_mins, minimum(Q_Hn.(trange,n)))
end
Qhn_mins
δt
println(Qhn_mins)

#bcolors = cgrad(:Blues,16,categorical=true,rev=true)
#rcolors = cgrad(:Reds,16,categorical=true,rev=true) 
for nnn in 1:1:5

    if nnn == 1
        f = Figure() 
        ax = Axis(f[1,1], xscale=log10, yscale=log10, ylabel="Q", xlabel=L"\text{log}(|\textit{t}|)", xlabelsize=20, ylabelsize=20,xticklabelsize=15,yticklabelsize=15)
        ylims!(ax,(1e0,1e5))
        xlims!(ax,(1e-2,10))
        ax.xticklabelsize = 22
        ax.yticklabelsize = 22
        ax.xlabelsize = 24  
        ax.ylabelsize = 22
        # δt = 0.035
        δt = 0.005
        trange= -10.0401:δt:10.005
    end
    if nnn == 4
        linewidth = 3.0 
        alpha = 1.0
        color = rcolors[7]
        Qn_history = Q_Hn.(trange,nnn)
        # hlines!(ax,[7.3],linewidth=2.0, linestyle=:dash,label="")
    elseif nnn > 4
        linewidth = 2.0
        alpha = 0.75
        color = bcolors[nnn*2]
        trange= -10.071:δt:10.071
        Qn_history = Q_Hn.(trange,nnn)
    elseif nnn == 8 
        linewidth = 2.0
        alpha = 0.75
        color = bcolors[nnn/2]
        trange= -1.005:δt:1.005
        Qn_history = Q_Hn.(trange,nnn)
    else 
        linewidth = 2.0
        alpha = 0.75
        color = bcolors[nnn*2]
        trange= -10.071:δt:10.071
        Qn_history = Q_Hn.(trange,nnn)
    end
    Qn_history = Q_Hn.(trange,nnn)
    #lines!(ax,trange, Qn_history, label="n = $nnn",alpha=alpha,linewidth=linewidth,color=nnn*15,colormap=:vik100,colorrange=(1,100))

    lines!(ax,(trange[div(length(trange), 2)+3:end]), Qn_history[div(length(trange), 2)+3:end], label="n = $nnn",  alpha=alpha,linewidth=linewidth,color=color)
end 
axislegend(position = :rb)
resize_to_layout!(f)
colsize!(f.layout, 1, 500)
display(f)
save("AdiabaticQHistory.png", f)
plt = plot()
plot!(plt,
    trange,
    ylabel="log(Q)",
    xlabel="Time",
    title="Q history for ϵ = 0.5",
    fontfamily="Computer Modern",
    xtickfontsize=12,
    ytickfontsize=12,
    yticks=(-1:1:7),
    log10.(Qn_history),
    ylimits=(-1,7),
    lw=3,
    label = "n = $nnn",
    # marker=:circle,
    )
plot!(plt,xlimits=(-5,5),ylimits=(1,6))
hline!(plt,[log10(35)],lw=1, label="")
plt = scatter(;legend=false)
for (idx,hist) in enumerate(Hn_history)
    x,y,z = decompose(hist)
    #println(norm(decompose(hist)))
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

c_n = exp(-(π) / (2*0.34))
c_n^2 * 100
# final_vector = [0,0,cos(theta)-0.002]
final_vector = [0,0,1-2*c_n^2]
#display(final_vector)
α = 1 - 2*c_n^2
#X,Z = 1, H0(1)[1]*1e2
X,Z = 1, 0.34*1e4
final_vector = zeros(3)
final_vector[3] = (Z*α * sqrt((Z^2 * α^2) -(X^2 +Z^2)*(α^2 -X^2)))/(X^2 +Z^2)
final_ρ = 1/2*(I + (final_vector[1]*σ[1] + final_vector[2]*σ[2] + final_vector[3]*σ[3]))
final_S = -tr(final_ρ*log(final_ρ))

t = .5; p = -.1;
W(2, t, p), W(3, t, p), W(4, t, p), W(5, t, p)