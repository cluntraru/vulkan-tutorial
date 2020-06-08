Cmake_minimum_required(VERSION 3.7)

project(vkTutorial)

find_package(Vulkan REQUIRED)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

set(IMPLEMENTATIONS "src/main.cpp")
set(LIBS)
#set(LIBS "lib/stb_image/stb_image.cpp"
#         "lib/stb_image/stb_image_write.cpp")

# GLFW
add_subdirectory("lib/glfw")
add_subdirectory("lib/glm-0.9.9.8")

add_executable(vkTutorial ${IMPLEMENTATIONS} ${LIBS})

target_include_directories(vkTutorial PRIVATE "lib/stb_image")

# GLFW Linking
target_link_libraries(vkTutorial Vulkan::Vulkan glfw glm ${CMAKE_DL_LIBS})

# YCM compile flags
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

