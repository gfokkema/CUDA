#include "common.h"
#include "scene.h"

#ifndef WINDOW_H_
#define WINDOW_H_

#define WIDTH 640
#define HEIGHT 480
#define TITLE "Raytracer"

class Window
{
public:
    Window(int width = WIDTH, int height = HEIGHT, std::string title = TITLE);
    ~Window();

    int width()
    {
        return m_width;
    }
    int height()
    {
        return m_height;
    }
    std::string title()
    {
        return m_title;
    }

    void render(Scene& scene);
    bool should_close();

    void cb_focus(int focused);
    void cb_key(int key, int scancode, int action, int mods);
    void cb_size(int width, int height);
    void handle_key(Camera* cam, float dt);
    void handle_mouse(Camera* cam);
private:
    void init_input();
    void init_buffers();
private:
    int m_width;
    int m_height;
    std::string m_title;

    GLFWwindow* p_window;
    GLuint m_vbo;
};

#endif /* WINDOW_H_ */
