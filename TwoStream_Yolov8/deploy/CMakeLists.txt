cmake_minimum_required(VERSION 3.16)
project(multicam_sync LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)
find_package(realsense2 REQUIRED)
find_package(yaml-cpp REQUIRED)

set(PYTHON_EXECUTABLE "python3" CACHE FILEPATH "Python executable for building multicam_sync_py")
execute_process(
  COMMAND ${PYTHON_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('INCLUDEPY')); print(sysconfig.get_config_var('EXT_SUFFIX')); print(sysconfig.get_config_var('LIBDIR')); print(sysconfig.get_config_var('LDLIBRARY'))"
  OUTPUT_VARIABLE PYTHON_CONFIG_RAW
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REPLACE "\n" ";" PYTHON_CONFIG_LIST "${PYTHON_CONFIG_RAW}")
list(GET PYTHON_CONFIG_LIST 0 PYTHON_INCLUDE_DIR)
list(GET PYTHON_CONFIG_LIST 1 PYTHON_EXT_SUFFIX)
list(GET PYTHON_CONFIG_LIST 2 PYTHON_LIB_DIR)
list(GET PYTHON_CONFIG_LIST 3 PYTHON_LIB_NAME)
set(PYTHON_LIB_PATH "${PYTHON_LIB_DIR}/${PYTHON_LIB_NAME}")

add_executable(multicam_sync multicam_sync.cpp)
target_link_libraries(multicam_sync PRIVATE ${OpenCV_LIBS} realsense2 yaml-cpp)
target_compile_options(multicam_sync PRIVATE -O3 -march=native -pthread)

add_library(multicam_sync_py MODULE multicam_sync_pybind.cpp)
target_include_directories(multicam_sync_py PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${PYTHON_INCLUDE_DIR})
target_link_libraries(multicam_sync_py PRIVATE ${OpenCV_LIBS} realsense2 yaml-cpp)
if(EXISTS ${PYTHON_LIB_PATH})
  target_link_libraries(multicam_sync_py PRIVATE ${PYTHON_LIB_PATH})
endif()
target_compile_options(multicam_sync_py PRIVATE -O3 -march=native -pthread)
set_target_properties(multicam_sync_py PROPERTIES PREFIX "" SUFFIX "${PYTHON_EXT_SUFFIX}" LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
