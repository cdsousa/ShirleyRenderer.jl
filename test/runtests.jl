using ShirleyRayTracer
using Test

function single_scancol()
    scene = Scene(Camera(Point3(13.,2.,3.), zero(Point3), Vec3(0,1,0), 20, 16/9, 0.1, 10.0))
    add!(scene, Sphere(Point3(0, 1, 0), 1.0, Dielectric(1.5)))
	add!(scene, Sphere(Point3(-4, 1, 0), 1.0, Lambertian(0.4,0.2,0.1)))
	add!(scene, Sphere(Point3(4, 1, 0), 1.0, Metal(0.7,0.6,0.5, 0.0)))
    width = 1200
    height = 100
    nsamples = 1
    max_depth = 5
    x = 600
	image = Vector{RGB{N0f8}}(undef, height)
	for y in 1:height
        image[y] = render_pixel(x, height-y+1, scene, height, width, nsamples, max_depth)
    end

    image
end

@testset "ShirleyRayTracer.jl" begin
    @test single_scancol()
end
