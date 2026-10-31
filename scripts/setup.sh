#!/bin/sh
# Gets the patch repository, applies the patch to llama.cpp and builds llama.xcframework for iOS devices.
# The Xcode project links llamacpp-apple5-lfm3/llama.cpp/build-apple/llama.xcframework of this repository.
# PATCH_REPO sets the URL of the patch repository and PATCH_REF the commit to use. The default commit is the one
# this app was tested with. An existing llamacpp-apple5-lfm3/ directory must be at PATCH_REF (PATCH_REF=HEAD accepts
# any commit).
set -eu

PATCH_REPO=${PATCH_REPO:-https://github.com/0xbyt4/llamacpp-apple5-lfm3}
PATCH_REF=${PATCH_REF:-4aa1270a42969c675a437a3b8e7bc7af81ce5b49}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DIR="$ROOT/llamacpp-apple5-lfm3"

if [ ! -d "$DIR" ]; then
    # clone next to the final directory, so a failed clone or checkout leaves nothing that a re-run would use
    rm -rf "$DIR.tmp"
    git clone -q "$PATCH_REPO" "$DIR.tmp"
    git -C "$DIR.tmp" checkout -q --detach "$PATCH_REF"
    mv "$DIR.tmp" "$DIR"
elif [ ! -e "$DIR/.git" ]; then
    echo "$DIR exists and is not a git checkout. Delete it and run this script again." >&2
    exit 1
elif [ "$(git -C "$DIR" rev-parse HEAD)" != "$(git -C "$DIR" rev-parse -q --verify "$PATCH_REF^{commit}" || true)" ]; then
    echo "$DIR is not at $PATCH_REF." >&2
    echo "Delete the directory and run this script again, or set PATCH_REF=HEAD to use it as it is." >&2
    exit 1
fi
"$DIR/scripts/setup.sh"
cd "$DIR/llama.cpp"
./build-xcframework.sh ios-device
echo "Built $DIR/llama.cpp/build-apple/llama.xcframework"
