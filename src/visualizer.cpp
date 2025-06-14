#include "visualizer.hpp"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

Visualizer::Visualizer(int width, int height)
    : window_width(width), window_height(height), window(nullptr), shaderProgram(0),
      camera_pos(0.0f, -5.0f, 20.0f), camera_front(0.0f, 0.0f, -1.0f), camera_up(0.0f, 1.0f, 0.0f),
      yaw(-90.0f), pitch(0.0f), last_x(width / 2.0f), last_y(height / 2.0f), first_mouse(true)
{
    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW");
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    window = glfwCreateWindow(width, height, "Boltzmann Smoke Simulation", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        throw std::runtime_error("Failed to create window");
    }
    glfwMakeContextCurrent(window);
    glfwSetWindowUserPointer(window, this);
    glfwSetKeyCallback(window, keyCallback);
    if (glewInit() != GLEW_OK) {
        throw std::runtime_error("Failed to initialize GLEW");
    }
    initShaders();
    // glfwSetCursorPosCallback(window, mouseCallback);
    // glfwSetScrollCallback(window, scrollCallback);
    // glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
}

Visualizer::~Visualizer() {
    if (window) {
        glfwDestroyWindow(window);
    }
    glfwTerminate();
}

void Visualizer::initShaders() {
    const char* vertexShaderSource = R"(
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in float aDensity;
        out float Density;
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        void main() {
            gl_Position = projection * view * model * vec4(aPos, 1.0);
            Density = aDensity;
        }
    )";
    const char* fragmentShaderSource = R"(
        #version 330 core
        in float Density;
        out vec4 FragColor;
        void main() {
            vec3 smokeColor = mix(vec3(0.1, 0.1, 0.1), vec3(0.9, 0.9, 0.9), Density);
            float alpha = min(Density * 5.0, 0.6);
            FragColor = vec4(smokeColor, alpha);
        }
    )";
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

void Visualizer::update(const float* density_data, int nx, int ny, int nz) {
    if (!window) return;

    // Calculate statistics
    float max_density = 0.0f;
    float avg_density = 0.0f;
    int active_voxels = 0;
    int total_voxels = nx * ny * nz;

    for (int z = 0; z < nz; z++) {
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                float density = density_data[z * nx * ny + y * nx + x];
                if (density > 0.00197f) {  // Adjust threshold to average density value
                    active_voxels++;
                    max_density = std::max(max_density, density);
                    avg_density += density;
                }
            }
        }
    }
    avg_density = active_voxels > 0 ? avg_density / active_voxels : 0.0f;

    // Existing rendering code
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glUseProgram(shaderProgram);
    glm::mat4 view = glm::lookAt(camera_pos, camera_pos + camera_front, camera_up);
    glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)window_width / (float)window_height, 0.1f, 100.0f);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
    for (int z = 0; z < nz; z++) {
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                float density = density_data[z * nx * ny + y * nx + x];
                if (density > 0.00197f) {
                    glm::mat4 model = glm::mat4(1.0f);
                    model = glm::translate(model, glm::vec3(x - nx/2, y - ny/2, z - nz/2) * 0.1f);
                    model = glm::scale(model, glm::vec3(0.1f));
                    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
                    float vertices[] = {
                        -0.5f, -0.5f,  0.5f, density,
                         0.5f, -0.5f,  0.5f, density,
                         0.5f,  0.5f,  0.5f, density,
                        -0.5f,  0.5f,  0.5f, density,
                        -0.5f, -0.5f, -0.5f, density,
                         0.5f, -0.5f, -0.5f, density,
                         0.5f,  0.5f, -0.5f, density,
                        -0.5f,  0.5f, -0.5f, density,
                    };
                    unsigned int indices[] = {
                        0, 1, 2, 2, 3, 0,
                        4, 5, 6, 6, 7, 4,
                        0, 4, 7, 7, 3, 0,
                        1, 5, 6, 6, 2, 1,
                        3, 2, 6, 6, 7, 3,
                        0, 1, 5, 5, 4, 0
                    };
                    unsigned int VBO, VAO, EBO;
                    glGenVertexArrays(1, &VAO);
                    glGenBuffers(1, &VBO);
                    glGenBuffers(1, &EBO);
                    glBindVertexArray(VAO);
                    glBindBuffer(GL_ARRAY_BUFFER, VBO);
                    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
                    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
                    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
                    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
                    glEnableVertexAttribArray(0);
                    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(3 * sizeof(float)));
                    glEnableVertexAttribArray(1);
                    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
                    glDeleteVertexArrays(1, &VAO);
                    glDeleteBuffers(1, &VBO);
                    glDeleteBuffers(1, &EBO);
                }
            }
        }
    }
    glfwSwapBuffers(window);
    glfwPollEvents();
}

void Visualizer::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    Visualizer* vis = static_cast<Visualizer*>(glfwGetWindowUserPointer(window));
    if (!vis) return;
    float cameraSpeed = 0.1f;
    if (key == GLFW_KEY_W && (action == GLFW_PRESS || action == GLFW_REPEAT))
        vis->camera_pos += cameraSpeed * vis->camera_front;
    if (key == GLFW_KEY_S && (action == GLFW_PRESS || action == GLFW_REPEAT))
        vis->camera_pos -= cameraSpeed * vis->camera_front;
    if (key == GLFW_KEY_A && (action == GLFW_PRESS || action == GLFW_REPEAT))
        vis->camera_pos -= glm::normalize(glm::cross(vis->camera_front, vis->camera_up)) * cameraSpeed;
    if (key == GLFW_KEY_D && (action == GLFW_PRESS || action == GLFW_REPEAT))
        vis->camera_pos += glm::normalize(glm::cross(vis->camera_front, vis->camera_up)) * cameraSpeed;
}

void Visualizer::mouseCallback(GLFWwindow* window, double xpos, double ypos) {
    Visualizer* vis = static_cast<Visualizer*>(glfwGetWindowUserPointer(window));
    if (!vis) return;
    if (vis->first_mouse) {
        vis->last_x = xpos;
        vis->last_y = ypos;
        vis->first_mouse = false;
    }
    float xoffset = xpos - vis->last_x;
    float yoffset = vis->last_y - ypos;
    vis->last_x = xpos;
    vis->last_y = ypos;
    float sensitivity = 0.1f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;
    vis->yaw += xoffset;
    vis->pitch += yoffset;
    if (vis->pitch > 89.0f)
        vis->pitch = 89.0f;
    if (vis->pitch < -89.0f)
        vis->pitch = -89.0f;
    glm::vec3 front;
    front.x = cos(glm::radians(vis->yaw)) * cos(glm::radians(vis->pitch));
    front.y = sin(glm::radians(vis->pitch));
    front.z = sin(glm::radians(vis->yaw)) * cos(glm::radians(vis->pitch));
    vis->camera_front = glm::normalize(front);
}

void Visualizer::scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    Visualizer* vis = static_cast<Visualizer*>(glfwGetWindowUserPointer(window));
    if (!vis) return;
    float sensitivity = 0.1f;
    vis->camera_pos += vis->camera_front * static_cast<float>(yoffset) * sensitivity;
}

void Visualizer::close() {
    if (window) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
} 