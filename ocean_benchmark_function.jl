
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
    free_surface = SplitExplicitFreeSurface(grid; substeps=70)
    closure = CATKEVerticalDiffusivity()

    model = HydrostaticFreeSurfaceModel(; grid,
                                          momentum_advection,
                                          tracer_advection,
                                          buoyancy,
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
