#include "window.h"

int main(int argc, char* argv[])
{
    Window window;
    Camera cam(window.width(), window.height());
    Scene scene(&cam);

    // Set the timer to zero
    glfwSetTime(0.0);
    double cur = 0;
    double prev = 0;
    // Do the render loop
    do
    {
        window.render(scene);
        cur = glfwGetTime();
        window.handle_key(&cam, cur - prev);
        prev = cur;
    }
    while (!window.should_close());
}
