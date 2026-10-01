# ORBIT

OpenMP Runtime Backend for Intelligent Tuning

## Build

```bash
git submodule update --init --recursive
cd lib/OPTKIT/lib/premake5
make -f Bootstrap.mak linux
cd bin/release
alias premake5="$(pwd)/premake5"
cd ../../../..
premake5 gmake
make -j$(nproc) config=release optkit_static
cd ../..
premake5 gmake2
make -C build config=debug orbit
./bin/Debug/orbit
make -C build config=test orbit_test
./bin/Test/orbit_test
```

<img width="1408" height="768" alt="Gemini_Generated_Image_rmsyt3rmsyt3rmsy" src="https://github.com/user-attachments/assets/55260ad1-ca70-4857-9ea9-78fdc92b3240" />
