using Oceananigans
using Oceananigans.Architectures: on_architecture
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: CATKEVerticalDiffusivity
using SeawaterPolynomials.TEOS10
using Random
using NVTX

arch = GPU() 
Nx   = 2160
Ny   = 1080
Nz   = 60

function ocean_benchmark(arch, Nx, Ny, Nz, topology, immersed, tracer_advection=WENO(order=7))    
    
    z_faces = collect(range(-6000, 0, length=Nz+1))

    grid = RectilinearGrid(arch; size=(Nx, Ny, Nz), 
                                 halo=(7, 7, 7), 
                                    z=z_faces, 
                                    x=(-1000000, 1000000), 
                                    y=(-1000000, 1000000), 
                                    topology)

    grid = if immersed
        Random.seed!(1234)
        bottom = Oceananigans.Architectures.on_architecture(arch, - 5000 .* rand(Nx, Ny) .- 1000)
        ImmersedBoundaryGrid(grid, GridFittedBottom(bottom); active_cells_map=true)
    else
        grid
    end
    
    @info "Grid is built"
    momentum_advection = WENOVectorInvariant()
    buoyancy = SeawaterBuoyancy(equation_of_state=TEOS10EquationOfState())
    coriolis = HydrostaticSphericalCoriolis()
    free_surface = SplitExplicitFreeSurface(grid; substeps=70)
    
    model = HydrostaticFreeSurfaceModel(; grid,
                                          momentum_advection,
                                          tracer_advection,
                                          buoyancy,
                                          coriolis,
                                          closure,
                                          free_surface,
                                          tracers = (:T, :S, :e))

    @info "Model is built"

    R = rand(size(model.grid))

    # initialize variables with randomish values
    Tᵢ = 0.0001 .* R .+ 20
    Sᵢ = 0.0001 .* R .+ 35
    uᵢ = 0.0001 .* R
    vᵢ = 0.0001 .* R
    
    set!(model, T=Tᵢ, S=Sᵢ, e=1e-6, u=uᵢ, v=vᵢ)

    return model
end

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

cheap_vertical_advection = (WENO(order=7), WENO(order=7), Centered())

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
