using Revise
using ShirleyRayTracer
using Images

#==

# launch using 40 threads
 julia -t 40 --project=.. -L RandomScene.jl  -e "main()" > random_scene.ppm

# launch force using 1 thread
 julia -t 1 --project=.. -L RandomScene.jl  -e "main()" > random_scene.ppm

# launch using thread count from environment
 julia --project=.. -L RandomScene.jl  -e "main()" > random_scene.ppm

==#

function add_random_scene!(scene::Scene)

	add!(scene, Sphere(Point3(0,-1000,0), 1000.0, Lambertian(0.5, 0.5, 0.5)))

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
			add!(scene, Sphere(center, 0.2, rand_material(rand())))
		end
	end

	add!(scene, Sphere(Point3(0, 1, 0), 1.0, Dielectric(1.5)))
	add!(scene, Sphere(Point3(-4, 1, 0), 1.0, Lambertian(0.4,0.2,0.1)))
	add!(scene, Sphere(Point3(4, 1, 0), 1.0, Metal(0.7,0.6,0.5, 0.0)))
end

function add_random_scene_lambertian!(S, scene::Scene)

	# S = Sphere
	# S = Sphere{ShirleyRayTracer.Material}
	# S = Sphere{ShirleyRayTracer.Lambertian}

	add!(scene, S(Point3(0,-1000,0), 1000.0, Lambertian(0.5, 0.5, 0.5)))

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
			add!(scene, rand_hitable(center, rand()))
		end
	end

	add!(scene, S(Point3(0, 1, 0), 1.0, Lambertian(0.5,0.3,0.1)))
	add!(scene, S(Point3(-4, 1, 0), 1.0, Lambertian(0.4,0.2,0.1)))
	add!(scene, S(Point3(4, 1, 0), 1.0, Lambertian(0.7,0.6,0.5)))
end

##


struct DummyHitable{T} <: ShirleyRayTracer.Hitable; end

function main(;filename="render.png", image_width=640, aspect_ratio=16/9, samples_per_pixel=5, max_depth=5)
	image_height = round(Int, image_width / aspect_ratio)

	# S = Sphere
	S = Sphere{ShirleyRayTracer.Material}
	# S = Sphere{ShirleyRayTracer.Lambertian}


	# H = ShirleyRayTracer.Hitable
	# H = Sphere
	H = Sphere{ShirleyRayTracer.Material}
	# H = Union{Sphere{Lambertian}, Sphere{Metal}, Sphere{Dielectric}}
	# H = Sphere{Lambertian}

	# H = Union{H, Sphere{ShirleyRayTracer.Material}}
	# H = Union{H, DummyHitable}
	# H = Union{H, DummyHitable{Int}}


	world = Scene{H}(Camera(Point3(13.,2.,3.), zero(Point3), Vec3(0,1,0), 20, aspect_ratio, 0.1, 10.0))

	# add_random_scene!(world)
	add_random_scene_lambertian!(S, world)

	@time image = render(world, image_width, image_height, samples_per_pixel, max_depth)
	save(filename, image)
end

main()

##


filename="render.png"
image_width=64
aspect_ratio=16/9
samples_per_pixel=5
max_depth=5
image_height = round(Int, image_width / aspect_ratio)

H = ShirleyRayTracer.Hitable
# H = Sphere{ShirleyRayTracer.Material}
# H = Union{Sphere{Lambertian}, Sphere{Metal}, Sphere{Dielectric}}
# H = Sphere{Lambertian}
# H = Sphere

world = Scene{H}(Camera(Point3(13.,2.,3.), zero(Point3), Vec3(0,1,0), 20, aspect_ratio, 0.1, 10.0))

# add_random_scene!(world)
add_random_scene_lambertian!(world)

image = Array{RGB{N0f8}, 2}(undef, image_height, image_width)
@time image = render!(image, world, image_width, image_height, samples_per_pixel, max_depth)
save(filename, image)

##


# @code_warntype render!(image, world, image_width, image_height, samples_per_pixel, max_depth)
# @code_warntype render_pixel(1,1, world, image_width, image_height, samples_per_pixel, max_depth)


using Cthulhu

# @descend_code_warntype render_pixel(1,1, world, image_width, image_height, samples_per_pixel, max_depth)


# using JET

# @report_call render_pixel(1,1, world, image_width, image_height, samples_per_pixel, max_depth)
