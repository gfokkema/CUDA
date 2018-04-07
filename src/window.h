#ifndef WINDOW_H_
#define WINDOW_H_

#include <common.h>
#include <device/device.h>
#include <scene.h>

#define TITLE "Raytracer"
#define WIDTH 1280
#define HEIGHT 800

class Window
{
public:
    Window(int width = WIDTH, int height = HEIGHT, std::string title = TITLE);
    ~Window();
private:
    void init_input();
    void init_buffers();

public:
    std::string title();
    int width();
    int height();

    void render(Device * device, Scene * scene);
    bool should_close();

    void cb_focus(int focused);
    void cb_key(int key, int scancode, int action, int mods);
    void cb_size(int width, int height);

    void handle_key(Camera* cam, float dt);
    void handle_mouse(Camera* cam);
private:
    int m_width;
    int m_height;
    std::string m_title;

    GLFWwindow* p_window;
    GLuint m_vbo;
};

#endif /* WINDOW_H_ */
