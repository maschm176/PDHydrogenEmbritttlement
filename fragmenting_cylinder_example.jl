using Peridynamics

using DelimitedFiles
function fragmenting_cylinder_geometry(input_mesh_file::AbstractString)
    input_raw = readdlm(input_mesh_file)
    position = copy(input_raw[:, 1:3]')
    volume = copy(input_raw[:, 5])
    return position, volume
end

input_mesh_file = joinpath(@__DIR__, "fragmenting_cylinder.txt")
position, volume = fragmenting_cylinder_geometry(input_mesh_file)

# using bond to bond material model
#body = Body(BBMaterial(), position, volume)
#material!(body, horizon=0.00417462, rho=7800, E=195e9, epsilon_c=0.02)

# using Cmaterial model
body = Body(CMaterial(), position, volume)
material!(body, horizon=0.00417462, rho=7800, E=195e9, nu=0.3, Gc=40.0)


velocity_ic!(p -> (200-50*((p[3]/0.05)-1)^2)*cos(atan(p[2],p[1])), body, :all_points, :x)
velocity_ic!(p -> (200-50*((p[3]/0.05)-1)^2)*sin(atan(p[2],p[1])), body, :all_points, :y)
velocity_ic!(p -> 100*((p[3]/0.05)-1), body, :all_points, :z)

vv = VelocityVerlet(time=2.5e-4)

job = Job(body, vv; path="results_cmaterial/fragmenting_cylinder", freq=10)
submit(job)