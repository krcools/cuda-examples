message(STATUS CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR})

cuda_add_executable(ex1-hello    ex1-hello.cu)
cuda_add_executable(ex2-info     ex2-info.cu)
cuda_add_executable(ex3-vecadd   ex3-vecadd.cu)
cuda_add_executable(ex4-julia    ex4-julia.cu)
cuda_add_executable(ex5-pi       ex5-pi.cu)
cuda_add_executable(ex6-matmat   ex6-matmat.cu)
cuda_add_executable(ex7-fdtd     ex7-fdtd.cu)



