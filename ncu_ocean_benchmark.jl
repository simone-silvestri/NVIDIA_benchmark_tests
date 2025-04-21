using Oceananigans
using Oceananigans.Architectures: on_architecture
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: CATKEVerticalDiffusivity
using SeawaterPolynomials.TEOS10
using Random
using NVTX
using CUDA

include("ocean_benchmark_function.jl")

arch = GPU() 
Nx   = 1080
Ny   = 900
Nz   = 60

# WENO 7 tracer advection
model_periodic = ocean_benchmark(arch, Nx, Ny, Nz, (Periodic, Periodic, Bounded), false)

# Warmup
for _ in 1:5
    time_step!(model_periodic, 0.1)
end

# The actual profile
CUDA.@profile begin
    Oceananigans.TimeSteppers.compute_tendencies!(model_periodic, [])
end