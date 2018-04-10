#include <device/device.h>
#include <scene.h>

static int sample;
std::chrono::time_point<std::chrono::system_clock> start;

void
start_timer()
{
    start = std::chrono::system_clock::now();
}

void
end_timer()
{
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double, std::milli> delta_time = end - start;
    std::cout << "\e[7mFrame duration:\t" << std::setw(5) << delta_time.count()
              << " ms" << "\tFramerate:\t" << std::setw(5)
              << 1000 / delta_time.count() << " fps\r";
    std::flush(std::cout);
}

double
Device::render(Scene& scene, color_t* buffer)
{
    this->copy(scene.materials(), scene.shapes());

    scene_t gpu_scene = d_scene(scene);
    this->producerays(gpu_scene, scene.camera().size(), sample);
    this->pathtrace(gpu_scene);
    this->rgbtoint(gpu_scene, sample);

    this->write(buffer, scene.camera().size());

    printf("%d\n", sample++);
    return 0.f;
}
