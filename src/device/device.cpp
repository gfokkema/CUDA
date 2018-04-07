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
Device::render(color_t * buffer, Scene * scene)
{
    int camsize = scene->camera()->size();
    camera_t camera = scene->camera()->gpu_type();

    this->producerays(camera, camsize, sample);
    this->copy(scene->materials(), scene->shapes());
    this->pathtrace(camera);
    this->rgbtoint(camera, sample);
    this->write(buffer, camsize);

    printf("%d\n", sample++);
    return 0.f;
}
