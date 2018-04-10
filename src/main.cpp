#include <common.h>
#include <device/cudadevice.h>
#include <scene.h>
#include <window.h>

int
main(int argc, char* argv[])
{
    // mat_t { { r, g, b }, emit, n, type }
    std::vector<mat_t> materials;
    materials.push_back( { { 0, 0, 0 }, { 1, 1, 1 }, DIFFUSE }); // 0: WHITE LIGHTING
    materials.push_back( { { 0, 0, 0 }, { 0, 0, 0 }, MIRROR }); // 1: REFLECTIVE
    materials.push_back( { { .6, .6, .6 }, { 0, 0, 0 }, DIFFUSE }); // 2: DIFFUSE WHITE
    materials.push_back( { { .2, .2, 1 }, { 0, 0, 0 }, DIFFUSE }); // 3: DIFFUSE BLUE
    materials.push_back( { { 1, .2, .2 }, { 0, 0, 0 }, DIFFUSE }); // 4: DIFFUSE RED
    materials.push_back( { { .2, 1, .2 }, { 0, 0, 0 }, DIFFUSE }); // 5: DIFFUSE GREEN

    // Initialize shapes here.
    // shape_t { { origin, radius }, matidx, type }
    float e = 1e5 + 1;
    std::vector<shape_t> shapes;
    shapes.push_back( (shape_t) { { Vector(-.5, 0, -1.5).gpu_type(), .3 }, 0, SPHERE });
    shapes.push_back( (shape_t) { { Vector(+.5, 0, -1.5).gpu_type(), .3 }, 1, SPHERE });
    shapes.push_back( (shape_t) { { Vector(0, +e, 0).gpu_type(), 1e5 }, 2, SPHERE }); // CEILING: WHITE
    shapes.push_back( (shape_t) { { Vector(0, -e, 0).gpu_type(), 1e5 }, 2, SPHERE }); // FLOOR: WHITE
    shapes.push_back( (shape_t) { { Vector(+e, 0, 0).gpu_type(), 1e5 }, 3, SPHERE }); // RIGHT: BLUE
    shapes.push_back( (shape_t) { { Vector(-e, 0, 0).gpu_type(), 1e5 }, 4, SPHERE }); // LEFT:  RED
    shapes.push_back( (shape_t) { { Vector(0, 0, -e - 1).gpu_type(), 1e5 }, 1, SPHERE }); // BACK: GREEN
    shapes.push_back( (shape_t) { { Vector(0, 0, +e - 1).gpu_type(), 1e5 }, 5, SPHERE }); // BEHIND: WHITE

    Window window;
    Camera cam(window.width(), window.height());
    Scene scene(cam, materials, shapes);

    CudaDevice device(cam.size(), materials.size(), shapes.size());

    // Set the timer to zero
    glfwSetTime(0.0);
    double cur = 0;
    double prev = 0;
    // Do the render loop
    do
    {
        window.render(device, scene);
        cur = glfwGetTime();
        window.handle_key(&cam, cur - prev);
        prev = cur;
    }
    while (!window.should_close());
}
