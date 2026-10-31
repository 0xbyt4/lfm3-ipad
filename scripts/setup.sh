#!/bin/sh
# Gets the patch repository, applies the patch to llama.cpp and builds llama.xcframework for iOS devices.
# The Xcode project links llamacpp-apple5-lfm3/llama.cpp/build-apple/llama.xcframework of this repository.
# PATCH_REPO sets the URL of the patch repository and PATCH_REF the commit to use. The default commit is the one
# this app was tested with. An existing llamacpp-apple5-lfm3/ directory is used as it is.
set -eu

PATCH_REPO=${PATCH_REPO:-https://github.com/0xbyt4/llamacpp-apple5-lfm3}
PATCH_REF=${PATCH_REF:-3f43a5ed454807c177e2e30d0f030b95fa4859b0}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DIR="$ROOT/llamacpp-apple5-lfm3"

if [ ! -d "$DIR" ]; then
    git clone -q "$PATCH_REPO" "$DIR"
    git -C "$DIR" checkout -q --detach "$PATCH_REF"
fi
"$DIR/scripts/setup.sh"
cd "$DIR/llama.cpp"
./build-xcframework.sh ios-device
echo "Built $DIR/llama.cpp/build-apple/llama.xcframework"
