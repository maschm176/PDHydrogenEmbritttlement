using JSON
using Peridynamics

# ─────────────────────────────────────────────────────────────────────────────
# Material parameters
# ─────────────────────────────────────────────────────────────────────────────

E     = 195e9       # Young's modulus (Pa)
nu    = 0.3         # Poisson's ratio — not needed for BBMaterial but good to keep
rho   = 7800.0      # Density (kg/m³)
delta = 0.00417462  # Horizon size (m)

# ─────────────────────────────────────────────────────────────────────────────
# Reading in critical stretch values calculated from exp data
# ─────────────────────────────────────────────────────────────────────────────
epsilon_c_data = JSON.parsefile("epsilon_c_values.json")


# Pick the concentration you want to simulate
concentration = "60"   # ppm
epsilon_c = epsilon_c_data[concentration]
println("Using epsilon_c = $epsilon_c for [H] = $concentration ppm")

# ─────────────────────────────────────────────────────────────────────────────
# Cylinder Geometry
# ─────────────────────────────────────────────────────────────────────────────
using DelimitedFiles
function fragmenting_cylinder_geometry(input_mesh_file::AbstractString)
    input_raw = readdlm(input_mesh_file)
    position = copy(input_raw[:, 1:3]')
    volume = copy(input_raw[:, 5])
    return position, volume
end

input_mesh_file = joinpath(@__DIR__, "fragmenting_cylinder.txt")
position, volume = fragmenting_cylinder_geometry(input_mesh_file)

# ─────────────────────────────────────────────────────────────────────────────
# Running PD simulation
# ─────────────────────────────────────────────────────────────────────────────
body = Body(BBMaterial(), position, volume)
material!(body, horizon=delta, rho=rho, E=E, epsilon_c=epsilon_c)
#material!(body, horizon=delta, rho=rho, E=E, epsilon_c=0.02)

print("")

# ─────────────────────────────────────────────────────────────────────────────
# Initial velocity field/BC conditions
# ─────────────────────────────────────────────────────────────────────────────
velocity_ic!(p -> (200-50*((p[3]/0.05)-1)^2)*cos(atan(p[2],p[1])), body, :all_points, :x)
velocity_ic!(p -> (200-50*((p[3]/0.05)-1)^2)*sin(atan(p[2],p[1])), body, :all_points, :y)
velocity_ic!(p -> 100*((p[3]/0.05)-1), body, :all_points, :z)

vv = VelocityVerlet(time=2.5e-4)

job = Job(body, vv; path="results_cmaterial/hydrogen_embrittlement_$(concentration)ppm_fragmentingCylinder", freq=10)
submit(job)



