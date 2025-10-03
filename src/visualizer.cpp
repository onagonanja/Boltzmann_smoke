#include "visualizer.hpp"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

Visualizer::Visualizer(int width, int height, const float* cam_pos)
    : window_width(width * 2), window_height(height), window(nullptr), 
      densityShaderProgram(0), temperatureShaderProgram(0),
      camera_pos(cam_pos[0], cam_pos[1], cam_pos[2]), camera_front(0.0f, 0.0f, -1.0f), camera_up(0.0f, 1.0f, 0.0f),
      yaw(-90.0f), pitch(0.0f), last_x(width / 2.0f), last_y(height / 2.0f), first_mouse(true)
{
    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW");
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    window = glfwCreateWindow(window_width, window_height, "Boltzmann Smoke Simulation - Density | Temperature", nullptr, nullptr);
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
    initBuffers();
    // glfwSetCursorPosCallback(window, mouseCallback);
    // glfwSetScrollCallback(window, scrollCallback);
    // glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
}

Visualizer::Visualizer(int width, int height, const float* cam_pos, bool show_temperature)
    : window_width(show_temperature ? width * 2 : width), window_height(height), window(nullptr),
      densityShaderProgram(0), temperatureShaderProgram(0),
      camera_pos(cam_pos[0], cam_pos[1], cam_pos[2]), camera_front(0.0f, 0.0f, -1.0f), camera_up(0.0f, 1.0f, 0.0f),
      yaw(-90.0f), pitch(0.0f), last_x(width / 2.0f), last_y(height / 2.0f), first_mouse(true)
{
    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW");
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    const char* title = show_temperature ? "Boltzmann Smoke Simulation - Density | Temperature" : "Boltzmann Smoke Simulation - Density";
    window = glfwCreateWindow(window_width, window_height, title, nullptr, nullptr);
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
    initBuffers();
}

Visualizer::~Visualizer() {
    if (window) {
        glfwDestroyWindow(window);
    }
    glfwTerminate();
}

void Visualizer::initShaders() {
    // Density field shader
    const char* densityVertexShaderSource = R"(
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec4 aInstance; // xyz: offset, w: value
        out float Density;
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        void main() {
            vec3 worldPos = aPos + aInstance.xyz;
            gl_Position = projection * view * model * vec4(worldPos, 1.0);
            Density = aInstance.w;
        }
    )";
    const char* densityFragmentShaderSource = R"(
        #version 330 core
        in float Density;
        out vec4 FragColor;
        void main() {
            vec3 smokeColor = mix(vec3(0.1, 0.1, 0.1), vec3(0.9, 0.9, 0.9), Density);
            float alpha = min(Density * 5.0, 0.6);
            FragColor = vec4(smokeColor, alpha);
        }
    )";
    
    // Temperature field shader
    const char* temperatureVertexShaderSource = R"(
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec4 aInstance; // xyz: offset, w: value
        out float Temperature;
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        void main() {
            vec3 worldPos = aPos + aInstance.xyz;
            gl_Position = projection * view * model * vec4(worldPos, 1.0);
            Temperature = aInstance.w;
        }
    )";
    const char* temperatureFragmentShaderSource = R"(
        #version 330 core
        in float Temperature;
        out vec4 FragColor;
        void main() {
            // Color gradient from blue (cold) to red (hot)
            // Show temperatures from 280K to 500K
            float normalizedTemp = clamp((Temperature - 280.0) / 220.0, 0.0, 1.0);
            vec3 coldColor = vec3(0.0, 0.0, 1.0);  // Blue
            vec3 hotColor = vec3(1.0, 0.0, 0.0);   // Red
            vec3 tempColor = mix(coldColor, hotColor, normalizedTemp);
            float alpha = min(normalizedTemp * 2.0 + 0.2, 0.8);  // Minimum alpha of 0.2
            FragColor = vec4(tempColor, alpha);
        }
    )";
    
    // Create density shader program
    unsigned int densityVertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(densityVertexShader, 1, &densityVertexShaderSource, NULL);
    glCompileShader(densityVertexShader);
    unsigned int densityFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(densityFragmentShader, 1, &densityFragmentShaderSource, NULL);
    glCompileShader(densityFragmentShader);
    densityShaderProgram = glCreateProgram();
    glAttachShader(densityShaderProgram, densityVertexShader);
    glAttachShader(densityShaderProgram, densityFragmentShader);
    glLinkProgram(densityShaderProgram);
    glDeleteShader(densityVertexShader);
    glDeleteShader(densityFragmentShader);
    
    // Create temperature shader program
    unsigned int temperatureVertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(temperatureVertexShader, 1, &temperatureVertexShaderSource, NULL);
    glCompileShader(temperatureVertexShader);
    unsigned int temperatureFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(temperatureFragmentShader, 1, &temperatureFragmentShaderSource, NULL);
    glCompileShader(temperatureFragmentShader);
    temperatureShaderProgram = glCreateProgram();
    glAttachShader(temperatureShaderProgram, temperatureVertexShader);
    glAttachShader(temperatureShaderProgram, temperatureFragmentShader);
    glLinkProgram(temperatureShaderProgram);
    glDeleteShader(temperatureVertexShader);
    glDeleteShader(temperatureFragmentShader);
}

void Visualizer::initBuffers() {
    // Unit cube geometry (positions only), indexed
    float vertices[] = {
        -0.05f, -0.05f,  0.05f,
         0.05f, -0.05f,  0.05f,
         0.05f,  0.05f,  0.05f,
        -0.05f,  0.05f,  0.05f,
        -0.05f, -0.05f, -0.05f,
         0.05f, -0.05f, -0.05f,
         0.05f,  0.05f, -0.05f,
        -0.05f,  0.05f, -0.05f
    };
    unsigned int indices[] = {
        0, 1, 2, 2, 3, 0,
        4, 5, 6, 6, 7, 4,
        0, 4, 7, 7, 3, 0,
        1, 5, 6, 6, 2, 1,
        3, 2, 6, 6, 7, 3,
        0, 1, 5, 5, 4, 0
    };

    glGenVertexArrays(1, &cubeVAO);
    glGenBuffers(1, &cubeVBO);
    glGenBuffers(1, &cubeEBO);
    glGenBuffers(1, &instanceVBO);

    glBindVertexArray(cubeVAO);
    glBindBuffer(GL_ARRAY_BUFFER, cubeVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cubeEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // Instance buffer: vec4(offset.xyz, value)
    glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
    glBufferData(GL_ARRAY_BUFFER, 1, nullptr, GL_STREAM_DRAW); // will resize each frame
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribDivisor(1, 1);

    glBindVertexArray(0);
}

void Visualizer::renderField(const float* data, int nx, int ny, int nz, bool isTemperature, int viewport_x, int viewport_y, int viewport_width, int viewport_height) {
    glViewport(viewport_x, viewport_y, viewport_width, viewport_height);
    
    unsigned int shaderProgram = isTemperature ? temperatureShaderProgram : densityShaderProgram;
    glUseProgram(shaderProgram);
    
    glm::mat4 view = glm::lookAt(camera_pos, camera_pos + camera_front, camera_up);
    glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)viewport_width / (float)viewport_height, 0.1f, 100.0f);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
    
    float threshold = isTemperature ? 280.0f : 0.00197f;

    std::vector<float> instances; // packed as x,y,z,value
    instances.reserve(nx * ny * nz / 4);
    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                float value = data[z * nx * ny + y * nx + x];
                if ((isTemperature && value > threshold) || (!isTemperature && value > threshold)) {
                    float ox = (x - nx/2) * 0.1f;
                    float oy = (y - ny/2) * 0.1f;
                    float oz = (z - nz/2) * 0.1f;
                    instances.push_back(ox);
                    instances.push_back(oy);
                    instances.push_back(oz);
                    instances.push_back(value);
                }
            }
        }
    }

    glm::mat4 model = glm::mat4(1.0f);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));

    glBindVertexArray(cubeVAO);
    glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
    glBufferData(GL_ARRAY_BUFFER, instances.size() * sizeof(float), instances.data(), GL_STREAM_DRAW);

    glDrawElementsInstanced(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0, (GLsizei)(instances.size() / 4));
    glBindVertexArray(0);
}

void Visualizer::update(const float* density_data, const float* temperature_data, int nx, int ny, int nz, bool show_temperature) {
    if (!window) return;

    // Clear the entire window
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    if (show_temperature) {
        // Calculate viewport dimensions for split view
        int half_width = window_width / 2;
        
        // Render density field on the left half
        renderField(density_data, nx, ny, nz, false, 0, 0, half_width, window_height);
        
        // Render temperature field on the right half
        renderField(temperature_data, nx, ny, nz, true, half_width, 0, half_width, window_height);
    } else {
        // Render only density field in full window
        renderField(density_data, nx, ny, nz, false, 0, 0, window_width, window_height);
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