#!/usr/bin/env bash
# Source on the host, before Singularity. Jobs must run from an immutable checkout.
SOURCE_GIT_SHA=$(git rev-parse HEAD) || return 1
SOURCE_GIT_STATUS=$(git status --porcelain) || return 1
SOURCE_GIT_STATUS_SHA=$(printf '%s' "$SOURCE_GIT_STATUS" | sha256sum | awk '{print $1}') || return 1
SOURCE_GIT_DIRTY=0
[ -n "$SOURCE_GIT_STATUS" ] && SOURCE_GIT_DIRTY=1
export SOURCE_GIT_SHA SOURCE_GIT_DIRTY SOURCE_GIT_STATUS_SHA
export SINGULARITYENV_SOURCE_GIT_SHA="$SOURCE_GIT_SHA"
export SINGULARITYENV_SOURCE_GIT_DIRTY="$SOURCE_GIT_DIRTY"
export SINGULARITYENV_SOURCE_GIT_STATUS_SHA="$SOURCE_GIT_STATUS_SHA"
echo "Source revision: $SOURCE_GIT_SHA (dirty=$SOURCE_GIT_DIRTY, status_sha=$SOURCE_GIT_STATUS_SHA)"
