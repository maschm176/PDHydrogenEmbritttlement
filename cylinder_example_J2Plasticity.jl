using Peridynamics
using StaticArrays
using LinearAlgebra
using DelimitedFiles



# ─────────────────────────────────────────────────────────────────────────────
# J2 Plastic constitutive model
# 
# Implements plasticity with von Mises yield criterion and linear isotropic hardening.
#
# Plugs into CMaterial via the AbstractConstitutiveModel interface.
# State (plastic strain, accumulated equiv. plastic strain) is stored
# inside the struct itself — pre-allocated arrays indexed by point number.
# ─────────────────────────────────────────────────────────────────────────────

# structure to store history variables
# J2 plasticity requires history variables to track plastic strain and accumulated plastic strain at each point

mutable struct J2Plastic <: Peridynamics.AbstractConstitutiveModel
    sigma_y0::Float64
    H::Float64
    eps_p::Vector{SMatrix{3,3,Float64,9}}
    p_acc::Vector{Float64}
    # Thread-local slot: one entry per thread, stores the current point index.
    # Written by our calc_first_piola_kirchhoff! override before calling
    # first_piola_kirchhoff, read inside first_piola_kirchhoff.
    current_point::Vector{Int}

    function J2Plastic(sigma_y0::Real, H::Real, n_points::Int)
        eps_p = fill(zero(SMatrix{3,3,Float64,9}), n_points)
        p_acc = zeros(Float64, n_points)
        current_point = zeros(Int, 1024)    # ← hard ceiling, not nthreads()
        return new(Float64(sigma_y0), Float64(H), eps_p, p_acc, current_point)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# first_piola_kirchhoff — the required interface method
#
# Called by CMaterial every timestep for every local point i.
# F is the nonlocal deformation gradient already computed by the framework.
# params.λ and params.μ are the Lamé constants for this point.
# ─────────────────────────────────────────────────────────────────────────────

function Peridynamics.calc_first_piola_kirchhoff!(
        storage::Peridynamics.CStorage,
        mat::Peridynamics.CMaterial{J2Plastic},
        params::Peridynamics.CPointParameters,
        defgrad_res,
        Δt,
        i)

    tid = Threads.threadid()
    model = mat.constitutive_model

    # Safety check — tells you clearly if the ceiling needs raising
    if tid > length(model.current_point)
        error("Thread ID $tid exceeds current_point buffer size $(length(model.current_point)). " *
              "Increase the buffer size in the J2Plastic constructor.")
    end

    model.current_point[tid] = i

    (; F, Kinv) = defgrad_res
    P = Peridynamics.first_piola_kirchhoff(model, storage, params, F)
    PKinv = P * Kinv
    σ = Peridynamics.cauchy_stress(P, F)
    Peridynamics.update_tensor!(storage.cauchy_stress, i, σ)
    return PKinv
end
# ─────────────────────────────────────────────────────────────────────────────
# first_piola_kirchhoff — reads i from the thread-local slot
# ─────────────────────────────────────────────────────────────────────────────

function Peridynamics.first_piola_kirchhoff(
        model::J2Plastic,
        storage::Peridynamics.AbstractStorage,
        params::Peridynamics.AbstractPointParameters,
        F::SMatrix{3,3,T,9}) where T

    # Retrieve point index from thread-local slot
    i  = model.current_point[Threads.threadid()]
    λ  = params.λ
    μ  = params.μ

    I3 = SMatrix{3,3,Float64,9}(I)

    # Green-Lagrange strain
    E = 0.5 .* (F' * F - I3)

    # Read history for this point
    E_p_old = model.eps_p[i]
    p_old   = model.p_acc[i]

    # Trial elastic strain and stress
    E_e_trial = E - E_p_old
    Evoigt = SVector{6,Float64}(
        E_e_trial[1,1], E_e_trial[2,2], E_e_trial[3,3],
        2*E_e_trial[2,3], 2*E_e_trial[3,1], 2*E_e_trial[1,2]
    )
    Cvoigt  = Peridynamics.get_hooke_matrix_voigt(params.nu, λ, μ)
    Svoigt  = Cvoigt * Evoigt
    S_trial = SMatrix{3,3,Float64,9}(
        Svoigt[1], Svoigt[6], Svoigt[5],
        Svoigt[6], Svoigt[2], Svoigt[4],
        Svoigt[5], Svoigt[4], Svoigt[3]
    )

    # Deviatoric trial stress and Von Mises yield check
    tr_S    = S_trial[1,1] + S_trial[2,2] + S_trial[3,3]
    s_trial = S_trial - (tr_S / 3) * I3
    s_norm  = sqrt(sum(s_trial .* s_trial))
    sigma_y = model.sigma_y0 + model.H * p_old
    f_trial = s_norm - sqrt(2.0/3.0) * sigma_y

    if f_trial <= 0.0
        S = S_trial
    else
        Δγ     = f_trial / (2*μ + (2.0/3.0)*model.H)
        n_flow = s_trial / s_norm
        s_new  = s_trial - 2*μ * Δγ * n_flow
        S      = s_new + (tr_S / 3) * I3
        model.eps_p[i] = E_p_old + Δγ * n_flow
        model.p_acc[i] = p_old   + sqrt(2.0/3.0) * Δγ
    end

    P = F * S
    return P
end

# ─────────────────────────────────────────────────────────────────────────────
# strain_energy_density — also required by the interface
# Use elastic trial strain energy as approximation (standard approach)
# ─────────────────────────────────────────────────────────────────────────────

function Peridynamics.strain_energy_density(
        model::J2Plastic,
        storage::Peridynamics.AbstractStorage,
        params::Peridynamics.AbstractPointParameters,
        F::SMatrix{3,3,T,9}) where T

    i   = model.current_point[Threads.threadid()]
    I3  = SMatrix{3,3,Float64,9}(I)
    E   = 0.5 .* (F' * F - I3)
    E_e = E - model.eps_p[i]
    Ψ   = 0.5 * params.λ * tr(E_e)^2 + params.μ * tr(E_e * E_e)
    return Ψ
end

# ─────────────────────────────────────────────────────────────────────────────
# Simulation setup — fragmenting cylinder with J2 plasticity
# ─────────────────────────────────────────────────────────────────────────────

function fragmenting_cylinder_geometry(input_mesh_file::AbstractString)
    input_raw = readdlm(input_mesh_file)
    position  = copy(input_raw[:, 1:3]')
    volume    = copy(input_raw[:, 5])
    return position, volume
end

input_mesh_file = joinpath(@__DIR__, "fragmenting_cylinder.txt")
position, volume = fragmenting_cylinder_geometry(input_mesh_file)

n_points = size(position, 2)

E       = 195e9
nu      = 0.3
sigma_y = 350e6
H       = 1e9
Gc      = 40.0

plastic_model = J2Plastic(sigma_y, H, n_points)

body = Body(CMaterial(model=plastic_model), position, volume)

material!(body, horizon=0.00417462, rho=7800, E=E, nu=nu, Gc=Gc)

# ── Fix 4: no-fail zones at top and bottom rims ──────────────────────────────
# The cylinder runs from z=0 to z=0.1 m based on the Peridigm point cloud.
# We protect one horizon-width at each end to prevent the surface truncation
# artifact from stopping the rim points unnaturally.
# Adjust the z thresholds if your cylinder has different end coordinates.
δ = 0.00417462                                    # horizon size

point_set!(p -> p[3] <= δ,       body, :bottom_rim)
point_set!(p -> p[3] >= 0.1 - δ, body, :top_rim)

no_failure!(body, :bottom_rim)   # ← was no_fail_region!
no_failure!(body, :top_rim)  

velocity_ic!(p -> (200-50*((p[3]/0.05)-1)^2)*cos(atan(p[2],p[1])), body, :all_points, :x)
velocity_ic!(p -> (200-50*((p[3]/0.05)-1)^2)*sin(atan(p[2],p[1])), body, :all_points, :y)
velocity_ic!(p -> 100*((p[3]/0.05)-1), body, :all_points, :z)

vv  = VelocityVerlet(time=2.5e-4)
job = Job(body, vv;
          path="results/fragmenting_cylinder_j2",
          freq=10,
          fields=(:displacement, :damage, :von_mises_stress, :cauchy_stress))
submit(job)

