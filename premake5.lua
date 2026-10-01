workspace "ORBIT"
    configurations { "Debug", "Release", "Test" }
    location "build"
    startproject "orbit"

local function base_project_setup()
    language "C++"
    cppdialect "C++17"
    targetdir "bin/%{cfg.buildcfg}"
    objdir "bin/obj/%{prj.name}/%{cfg.buildcfg}"
    includedirs { "src" }
    warnings "Extra"

    filter "system:linux"
        buildoptions { "-fopenmp" }
        linkoptions { "-fopenmp" }
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