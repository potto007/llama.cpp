# dual-context-server: single-process two-context OpenAI server (orchestrator + worker)
# sharing one llama_model. Links the reusable server core (llama-server-impl, which
# transitively brings server-context, llama-ui, cpp-httplib, llama, mtmd).
set(TARGET llama-dual-context-server)
add_executable(${TARGET} dual-context-server.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/tools/server)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/tools/mtmd)
target_link_libraries(${TARGET} PRIVATE llama-server-impl llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
