#include <window.h>

void
focus_callback(GLFWwindow * glfwwindow, int focused)
{
    Window* window = (Window*) glfwGetWindowUserPointer(glfwwindow);
    window->cb_focus(focused);
}

void
key_callback(GLFWwindow * glfwwindow, int key, int scancode, int action,
                  int mods)
{
    Window* window = (Window*) glfwGetWindowUserPointer(glfwwindow);
    window->cb_key(key, scancode, action, mods);
}

void
size_callback(GLFWwindow * glfwwindow, int width, int height)
{
    Window* window = (Window*) glfwGetWindowUserPointer(glfwwindow);
    window->cb_size(width, height);
}

Window::Window(int width, int height, std::string title)
: p_window(nullptr),
  m_width(width),
  m_height(height),
  m_title(title)
{
    // Initialise GLFW
    if (!glfwInit())
    {
        throw std::runtime_error("Failed to initialize GLFW.");
    }

    // Open a window and create its OpenGL context
    p_window = glfwCreateWindow(width, height, title.c_str(), NULL, NULL);
    // FIXME: glfwCreateWindow causes a race condition!
    //        Find a way to block until succesfull window creation...
    usleep(10000);
    if (p_window == NULL)
    {
        glfwTerminate();
        throw std::runtime_error("Failed to open GLFW window.");
    }
    glfwMakeContextCurrent(p_window);
    glfwSetWindowUserPointer(p_window, this);

    // Initialize GLEW
    if (glewInit() != GLEW_OK)
    {
        glfwTerminate();
        throw std::runtime_error("Failed to initialize GLEW.");
    }

    init_input();
    init_buffers();
}

Window::~Window()
{
    // Close OpenGL window and terminate GLFW
    glfwTerminate();
}

void
Window::init_input()
{
    // Ensure we can capture the escape key being pressed below
    glfwSetInputMode(p_window, GLFW_STICKY_KEYS, GL_TRUE);
//    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // Set the keyboard callback for time independent keyboard handling
    glfwSetKeyCallback(p_window, &key_callback);
    glfwSetWindowFocusCallback(p_window, &focus_callback);
    glfwSetWindowSizeCallback(p_window, &size_callback);
}

void
Window::init_buffers()
{
    // Initialize our vertex buffer
    glGenBuffers(1, &m_vbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_vbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, m_width * m_height * 3, 0, GL_DYNAMIC_DRAW);
}

std::string
Window::title()
{
    return m_title;
}

int
Window::width()
{
    return m_width;
}

int
Window::height()
{
    return m_height;
}

void
Window::render(Device * device, Scene * scene)
{
    glClear(GL_COLOR_BUFFER_BIT);

    // Map the buffer and render the scene
    color_t* buffer = (color_t*) glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
    device->render(buffer, scene);
    glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);

    // Draw the buffer onto the off screen buffer
    glDrawPixels(WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, 0); // FIXME: fixed width and height

    // Swap buffers
    glfwSwapBuffers(p_window);
    glfwPollEvents();
}

bool
Window::should_close()
{
    return glfwWindowShouldClose(p_window);
}

/**
 * Focus callback function
 */
void
Window::cb_focus(int focused)
{
//    if (focused) {
//        double middle_x = WIDTH/2.0;
//        double middle_y = HEIGHT/2.0;
//        glfwSetCursorPos(window, middle_x, middle_y);
//        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
//    }
}

/**
 * Time independent keyboard function
 */
void
Window::cb_key(int key, int scancode, int action, int mods)
{
    switch (key)
    {
    case GLFW_KEY_ESCAPE:
        glfwSetWindowShouldClose(p_window, 1);
        break;
    }
}

void
Window::cb_size(int width, int height)
{
    std::cout << "New size: (" << width << "," << height << ")" << std::endl;

    m_width = width;
    m_height = height;
}

/**
 * Time dependent keyboard function
 */
void
Window::handle_key(Camera* cam, float dt)
{
    if (glfwGetKey(p_window, GLFW_KEY_LEFT) == GLFW_PRESS)
        cam->strafe(-1.f, dt);
    if (glfwGetKey(p_window, GLFW_KEY_RIGHT) == GLFW_PRESS)
        cam->strafe(1.f, dt);
    if (glfwGetKey(p_window, GLFW_KEY_UP) == GLFW_PRESS)
        cam->move(1.f, dt);
    if (glfwGetKey(p_window, GLFW_KEY_DOWN) == GLFW_PRESS)
        cam->move(-1.f, dt);
}

void
Window::handle_mouse(Camera* cam)
{
    double middle_x = m_width / 2.0;
    double middle_y = m_height / 2.0;

    double x, y;
    glfwGetCursorPos(p_window, &x, &y);
    glfwSetCursorPos(p_window, middle_x, middle_y);
    if (x < m_width && y < m_height)
    {
        double dx = x - middle_x;
        double dy = y - middle_y;
        if (dx == 0.f && dy == 0.f)
            return;

        cam->lookAt(x, m_height - y);
    }
}
