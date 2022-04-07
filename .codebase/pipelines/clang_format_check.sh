#!/bin/bash

git_status=$(git status --porcelain)
if [[ $git_status ]]; then
  echo "Checkout code is not clean"
  echo "${git_status}"
  exit 1
fi

find -name '*.cpp' -o -name '*.h' -not -path "./external/*" | xargs clang-format-7 -i -style=file
git_status=$(git status --porcelain)
if [[ $git_status ]]; then
  echo "clang-format-7 is not happy, please run \"clang-format-7 -i -style /PATH/TO/foo.cpp\" to the following files"
  echo "${git_status}"
  exit 1
else
  echo "PASSED C++ format"
fi