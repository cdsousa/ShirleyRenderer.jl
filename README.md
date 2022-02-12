Experiments over https://github.com/lawless-m/ShirleyRenderer.jl

Seamless CUDA support through Tullio.jl:

```julia
# RandomScene.jl
julia> main(samples_per_pixel=50)
 59.563979 seconds

julia> main(samples_per_pixel=50, use_cuda=true)
 12.267839 seconds
```