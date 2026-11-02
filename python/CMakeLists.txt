find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_annvil MODULE annvil_py.cpp)
target_link_libraries(_annvil PRIVATE annvil)
target_compile_options(_annvil PRIVATE -Wall -Wextra)
install(TARGETS _annvil DESTINATION annvil)
install(FILES annvil/__init__.py DESTINATION annvil)
