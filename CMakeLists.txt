cmake_minimum_required(VERSION 3.16)
project(FluidSimulation)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find SFML
find_package(SFML 2.5 COMPONENTS graphics window system REQUIRED)

# Find OpenGL
find_package(OpenGL REQUIRED)

# Define ImGui and ImGui-SFML sources
set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/imgui)
set(IMGUI_SFML_DIR ${CMAKE_SOURCE_DIR}/imgui-sfml)
set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
)
set(IMGUI_SFML_SOURCES ${IMGUI_SFML_DIR}/imgui-SFML.cpp)

# Copy imconfig-SFML.h to imconfig.h for SFML compatibility
configure_file(${IMGUI_SFML_DIR}/imconfig-SFML.h ${IMGUI_DIR}/imconfig.h COPYONLY)

# Add executable
add_executable(fluid_sim fluid.cpp ${IMGUI_SOURCES} ${IMGUI_SFML_SOURCES})

# Include directories
target_include_directories(fluid_sim PRIVATE
    ${IMGUI_DIR}
    ${IMGUI_SFML_DIR}
    ${SFML_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(fluid_sim PRIVATE
    sfml-graphics
    sfml-window
    sfml-system
    ${OPENGL_LIBRARIES}
)

# Enable OpenMP if available (optional, since your code uses it)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(fluid_sim PRIVATE OpenMP::OpenMP_CXX)
endif()