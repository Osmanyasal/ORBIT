newaction {
    trigger = "clean",
    description = "Remove ORBIT generated build files",
    execute = function()
        os.rmdir("bin")
        os.rmdir("build")
        os.remove("Makefile")
    end
}

workspace "ORBIT"
    configurations { "Debug", "Release", "Test" }
    location "build"
    startproject "orbit"

local optkit_root = "lib/OPTKIT"
local optkit_spdlog_root = optkit_root .. "/lib/spdlog"
local optkit_libpfm_root = optkit_root .. "/lib/libpfm4"
dofile(optkit_root .. "/premake5_utilities.lua")

local function base_project_setup()
    language "C++"
    cppdialect "C++17"
    targetdir "bin/%{cfg.buildcfg}"
    objdir "bin/obj/%{prj.name}/%{cfg.buildcfg}"
    includedirs {
        "src",
        optkit_root .. "/src",
        optkit_spdlog_root .. "/include",
        optkit_libpfm_root .. "/include"
    }
    libdirs {
        optkit_root .. "/bin/%{cfg.buildcfg}",
        optkit_libpfm_root .. "/lib"
    }
    links { "optkit_dynamic", "pfm", "pthread", "dl" }
    linkoptions { path.getabsolute(optkit_spdlog_root .. "/build/libspdlog.a") }
    warnings "Extra"

    if dynamic_lib_exists("nvidia-ml") then
        local nvml_include = get_nvml_include()
        if nvml_include then
            includedirs { nvml_include }
        end
        libdirs { "/usr/lib/x86_64-linux-gnu" }
        links { "nvidia-ml" }
    end

    if dynamic_lib_exists("amd_smi") then
        local rocm_include = get_rocm_include()
        if rocm_include then
            includedirs { rocm_include }
        end
        libdirs { "/opt/rocm/lib" }
        links { "amd_smi" }
    elseif dynamic_lib_exists("rocm_smi64") or dynamic_lib_exists("rocm_smi") then
        local rocm_include = get_rocm_include()
        if rocm_include then
            includedirs { rocm_include }
        end
        libdirs { "/opt/rocm/lib" }
        links { "rocm_smi64" }
    end

    if dynamic_lib_exists("cupti") then
        local cupti_include = get_cupti_include()
        if cupti_include then
            includedirs { cupti_include }
        end
        libdirs { "/usr/local/cuda/lib64" }
        links { "cupti" }
    end

    if dynamic_lib_exists("netsnmp") then
        links { "netsnmp" }
    end

    filter "system:linux"
        buildoptions { "-fopenmp" }
        linkoptions {
            "-fopenmp",
            "-rdynamic",
            "-Wl,-rpath,'$$ORIGIN/../../" .. optkit_root .. "/bin/%{cfg.buildcfg}'",
            "-Wl,-rpath,'$$ORIGIN/../../" .. optkit_libpfm_root .. "/lib'"
        }
    filter {}

    filter "configurations:Debug"
        symbols "On"
        defines { "ORBIT_MODE_DEBUG" }
    filter "configurations:Release"
        optimize "Speed"
        defines { "ORBIT_MODE_RELEASE" }
    filter "configurations:Test"
        symbols "On"
        defines { "ORBIT_MODE_TEST" }
    filter {}
end

project "orbit_static"
    kind "StaticLib"
    targetname "orbit"
    base_project_setup()
    files { "src/orbit.cc", "src/orbit.hh" }

project "orbit"
    kind "ConsoleApp"
    base_project_setup()
    files { "src/main.cc" }
    links { "orbit_static" }

project "orbit_test"
    kind "ConsoleApp"
    base_project_setup()
    files { "test/**.cc" }
    links { "orbit_static" }

project "orbit_example"
    kind "ConsoleApp"
    base_project_setup()
    files { "examples/**.cc" }
    links { "orbit_static" }