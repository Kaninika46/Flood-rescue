#!/usr/bin/env bash
# Run ONCE after unpacking the project archive, from anywhere:
#     bash scripts/bootstrap_git.sh
# It creates the git repository, adds the three third-party submodules at the pinned commits
# listed in THIRD_PARTY.md, and makes the first commit under YOUR git name. Then you push.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -d .git ]; then
    echo "This folder is already a git repository; nothing to do." >&2
    exit 1
fi
if ! git config user.name >/dev/null || ! git config user.email >/dev/null; then
    echo "Set your git identity first:" >&2
    echo '  git config --global user.name "Your Name"' >&2
    echo '  git config --global user.email "you@example.com"' >&2
    exit 1
fi

git init -q -b main

add_submodule() {
    local name=$1 url=$2 commit=$3
    rm -rf "third_party/$name"
    git submodule add -q "$url" "third_party/$name"
    git -C "third_party/$name" checkout -q "$commit"
    git add "third_party/$name"
    echo "  third_party/$name at ${commit:0:10}"
}

echo "Adding third-party submodules:"
add_submodule etci2021 https://github.com/sidgan/ETCI-2021-Competition-on-Flood-Detection 90d47ca217e240395fcf16287bd85e3cf79afe88
add_submodule spacenet8 https://github.com/SpaceNetChallenge/SpaceNet8 30ca30c2530c6a234d040f60a99a7366ff62d756
add_submodule fwdet https://github.com/csdms-contrib/fwdet b47aec828bee313d373f37899e4d98d75937eb98

git add -A
git commit -q -m "Flood Rescue AI: initial import"
git submodule status
echo
echo "Done. Push to your (empty) GitHub repository:"
echo "  git remote add origin https://github.com/<your-github-user>/<your-repo>.git"
echo "  git push -u origin main"
