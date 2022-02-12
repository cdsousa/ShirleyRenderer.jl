using Revise
using Images
using CUDA, CUDAKernels, KernelAbstractions
using Tullio
CUDA.allowscalar(false)
using ShirleyRayTracer


function add_random_scene!(scene)

	push!(scene, Sphere(Point3(0,-1000,0), 1000.0, Lambertian(0.5, 0.5, 0.5)))

	rand_material(p) = if p < 0.8
				Lambertian()
			elseif p < 0.95
				rf = randf(0.5, 1)
				Metal(rf, rf, rf, 0.5rand())
			else
				Dielectric(1.5)
			end

	for a in -11:10, b in -11:10
		center = Point3(a + 0.9rand(), 0.2, b + 0.9rand())
		if ShirleyRayTracer.magnitude(center - Point3(4, 0.2, 0)) > 0.9
			push!(scene, Sphere(center, 0.2, rand_material(rand())))
		end
	end

	push!(scene, Sphere(Point3(0, 1, 0), 1.0, Dielectric(1.5)))
	push!(scene, Sphere(Point3(-4, 1, 0), 1.0, Lambertian(0.4,0.2,0.1)))
	push!(scene, Sphere(Point3(4, 1, 0), 1.0, Metal(0.7,0.6,0.5, 0.0)))
end

function add_random_scene_lambertian!(S, scene)

	# S = Sphere
	# S = Sphere{ShirleyRayTracer.Material}
	# S = Sphere{ShirleyRayTracer.Lambertian}

	push!(scene, S(Point3(0,-1000,0), 1000.0, Lambertian(0.5, 0.5, 0.5)))

	rand_material(p) = if p < 0.8
				Lambertian(0.5,0.3,0.5)
			elseif p < 0.95
				rf = randf(0.5, 1)
				Lambertian(rf, rf, 0.5rand())
			else
				Lambertian(0.1,0.1,0.1)
			end

	function rand_hitable(center, p)
		# if p < 0.5
			S(center, 0.2, rand_material(rand()))
		# else
		# 	ShirleyRayTracer.Sphere2(center, 0.2, rand_material(rand()))
		# end
	end

	for a in -11:10, b in -11:10
		center = Point3(a + 0.9rand(), 0.2, b + 0.9rand())
		if ShirleyRayTracer.magnitude(center - Point3(4, 0.2, 0)) > 0.9
			push!(scene, rand_hitable(center, rand()))
		end
	end

	push!(scene, S(Point3(0, 1, 0), 1.0, Lambertian(0.5,0.3,0.1)))
	push!(scene, S(Point3(-4, 1, 0), 1.0, Lambertian(0.4,0.2,0.1)))
	push!(scene, S(Point3(4, 1, 0), 1.0, Lambertian(0.7,0.6,0.5)))
end


struct DummyHitable{T} <: ShirleyRayTracer.Hitable; end


filename="render.png"
image_width=640
aspect_ratio=16/9
samples_per_pixel=5
max_depth=10
image_height = round(Int, image_width / aspect_ratio)
# function main(;filename="render.png", image_width=640, aspect_ratio=16/9, samples_per_pixel=5, max_depth=5)

image_height = round(Int, image_width / aspect_ratio)

# S = Sphere
# S = Sphere{ShirleyRayTracer.Material}
# S = Sphere{ShirleyRayTracer.Lambertian}

# H = ShirleyRayTracer.Hitable
# H = Sphere
# H = Sphere{ShirleyRayTracer.Material}
H = Union{Sphere{Lambertian}, Sphere{Metal}, Sphere{Dielectric}}
# H = Sphere{Lambertian}

# H = Union{H, Sphere{ShirleyRayTracer.Material}}
# H = Union{H, DummyHitable}
# H = Union{H, DummyHitable{Int}}

hitables = Vector{H}()

add_random_scene!(hitables)
# add_random_scene_lambertian!(S, hitables);
# push!(hitables, S(Point3(0,-1000,0), 1000.0, Lambertian(0.5, 0.5, 0.5)))

cam = Camera(Point3(13.,2.,3.), zero(Point3), Vec3(0,1,0), 20, aspect_ratio, 0.1, 10.0)

scene = Scene(cam, hitables)

image = Array{RGB{Float32}, 2}(undef, image_height, image_width);

scene = scene |> cu;
image = image |> cu;

nothing

##

@time begin
	render!(image, scene, image_width, image_height, samples_per_pixel, max_depth)
	isa(image, CuArray) && CUDA.synchronize()
end

@time image_out = image |> Array;
save(filename, image_out)

