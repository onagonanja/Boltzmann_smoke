#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <memory>

class Visualizer {
public:
    Visualizer(int width, int height);
    ~Visualizer();

    void update(const float* density_data, const float* temperature_data, int nx, int ny, int nz, bool show_temperature = true);
    bool shouldClose() const { return glfwWindowShouldClose(window); }
    void close();

private:
    void initShaders();
    void renderField(const float* data, int nx, int ny, int nz, bool isTemperature, int viewport_x, int viewport_y, int viewport_width, int viewport_height);
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void mouseCallback(GLFWwindow* window, double xpos, double ypos);
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);

    int window_width, window_height;
    GLFWwindow* window;
    unsigned int densityShaderProgram;
    unsigned int temperatureShaderProgram;

    // Camera-related variables
    glm::vec3 camera_pos;
    glm::vec3 camera_front;
    glm::vec3 camera_up;
    float yaw;
    float pitch;
    float last_x;
    float last_y;
    bool first_mouse;
}; 