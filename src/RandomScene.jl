using ShirleyRayTracer
using Images
using CUDA

CUDA.allowscalar(false)


function add_random_scene!(hitables)

	push!(hitables, Sphere(Point3(0,-1000,0), 1000.0, Lambertian(0.5, 0.5, 0.5)))

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
			push!(hitables, Sphere(center, 0.2, rand_material(rand())))
		end
	end

	push!(hitables, Sphere(Point3(0, 1, 0), 1.0, Dielectric(1.5)))
	push!(hitables, Sphere(Point3(-4, 1, 0), 1.0, Lambertian(0.4,0.2,0.1)))
	push!(hitables, Sphere(Point3(4, 1, 0), 1.0, Metal(0.7,0.6,0.5, 0.0)))
end

function main(;filename="render.png", image_width=640, aspect_ratio=16/9, samples_per_pixel=5, max_depth=5, use_cuda=false)

	image_height = round(Int, image_width / aspect_ratio)

	hitables = Union{Sphere{Lambertian}, Sphere{Metal}, Sphere{Dielectric}}[]
	add_random_scene!(hitables)

	cam = Camera(Point3(13.,2.,3.), zero(Point3), Vec3(0,1,0), 20, aspect_ratio, 0.1, 10.0)

	if !use_cuda
		world = Scene(cam, hitables)
		arrtype = Array
	else
		world = Scene(cam, hitables) |> cu
		arrtype = CuArray
	end

	@time image = render(arrtype, world, image_width, image_height, samples_per_pixel, max_depth) |> Array

	save(filename, image)
end

main()
main(use_cuda=true)

main(samples_per_pixel=50)
main(samples_per_pixel=50, use_cuda=true)
