# ORBIT

<img width="1408" height="768" alt="Gemini_Generated_Image_rmsyt3rmsyt3rmsy" src="https://github.com/user-attachments/assets/55260ad1-ca70-4857-9ea9-78fdc92b3240" />

OpenMP Runtime Backend for Intelligent Tuning

## Download and Install 🚀

```bash
git clone https://github.com/Osmanyasal/ORBIT.git
git submodule update --init --recursive

pushd lib/OPTKIT/lib/premake5
make -f Bootstrap.mak linux
alias premake5="$PWD/bin/release/premake5"
popd

pushd lib/OPTKIT
premake5 gmake
make -j"$(nproc)" config=release optkit_dynamic
popd

premake5 gmake
make -C build config=release orbit
./bin/Release/orbit
make -C build config=test orbit_test
./bin/Test/orbit_test
```
