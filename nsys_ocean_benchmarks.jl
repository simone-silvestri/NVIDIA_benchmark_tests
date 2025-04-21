using Oceananigans
using Oceananigans.Architectures: on_architecture
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: CATKEVerticalDiffusivity
using SeawaterPolynomials.TEOS10
using Random
using NVTX

include("ocean_benchmark_function.jl")

arch = GPU() 
Nx   = 2160
Ny   = 1080
Nz   = 60

# WENO 7 tracer advection
model_periodic = ocean_benchmark(arch, Nx, Ny, Nz, (Periodic, Periodic, Bounded), false)
model_bounded  = ocean_benchmark(arch, Nx, Ny, Nz, (Bounded,   Bounded, Bounded), false)

# Warmup
for _ in 1:5
    time_step!(model_periodic, 0.1)
    time_step!(model_bounded,  0.1)
end

NVTX.@range "Periodic topology, expensive vertical advection" begin
    for _ in 1:10
        time_step!(model_periodic, 0.1)
    end
end
NVTX.@range "Bounded topology, expensive vertical advection" begin
    for _ in 1:10
        time_step!(model_bounded, 0.1)
    end
end

cheap_vertical_advection = FluxFormAdvection(WENO(order=7), WENO(order=7), Centered())

model_periodic = ocean_benchmark(arch, Nx, Ny, Nz, (Periodic, Periodic, Bounded), false, cheap_vertical_advection)
model_bounded  = ocean_benchmark(arch, Nx, Ny, Nz, (Bounded,   Bounded, Bounded), false, cheap_vertical_advection)

# Warmup
for _ in 1:5
    time_step!(model_periodic, 0.1)
    time_step!(model_bounded,  0.1)
end

NVTX.@range "Periodic topology, cheap vertical advection" begin
    for _ in 1:10
        time_step!(model_periodic, 0.1)
    end
end
NVTX.@range "Bounded topology, cheap vertical advection" begin
    for _ in 1:10
        time_step!(model_bounded, 0.1)
    end
end

model_immersed = ocean_benchmark(arch, Nx, Ny, Nz, (Periodic, Periodic, Bounded), true)

# Warmup
for _ in 1:5
    time_step!(model_immersed, 0.1)
end

NVTX.@range "Immersed grid" begin
    for _ in 1:10
        time_step!(model_immersed, 0.1)
    end
end
