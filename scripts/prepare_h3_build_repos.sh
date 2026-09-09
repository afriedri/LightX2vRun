#!/usr/bin/env bash
# Align the three sibling build repositories with their HVAT GitCode branches.
set -Eeuo pipefail

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    echo 'Usage: bash prepare_h3_build_repos.sh [CODE_DIR]'
    echo 'Default: ~/Code. Requires clean worktrees; saves differing local branch heads.'
    exit 0
fi
(( $# <= 1 )) || { echo 'Expected at most one Code directory' >&2; exit 2; }
code_dir="${1:-$HOME/Code}"
code_dir="$(cd -- "$code_dir" && pwd)"
repos=(aiter LightX2V LightX2VRun)
branches=(LightX2V/hvat-gfx1201-R9600 hvat-gfx1201-R9600 hvat-gfx1201)
stamp="$(date +%Y%m%d-%H%M%S)-$$"

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# Inspect all repositories before changing any branch. Do not stash user changes.
for repo in "${repos[@]}"; do
    path="$code_dir/$repo"
    [[ -d "$path" ]] || fail "Missing repository: $path"
    root="$(git -C "$path" rev-parse --show-toplevel)"
    [[ "$(cd -- "$root" && pwd)" == "$path" ]] || fail "Not a repository root: $path"
    [[ -z "$(git -C "$path" status --porcelain --untracked-files=all)" ]] || {
        git -C "$path" status --short
        fail "$repo has local changes; commit or save them before retrying."
    }
    for state in MERGE_HEAD CHERRY_PICK_HEAD REVERT_HEAD rebase-merge rebase-apply sequencer; do
        state_path="$(git -C "$path" rev-parse --path-format=absolute --git-path "$state")"
        [[ ! -e "$state_path" ]] || fail "$repo has an unfinished Git operation: $state"
    done
    origin="$(git -C "$path" config --get remote.origin.url)"
    case "$origin" in
        "git@gitcode.com:hvat-ai/$repo"|"git@gitcode.com:hvat-ai/$repo.git"|\
        "https://gitcode.com/hvat-ai/$repo"|"https://gitcode.com/hvat-ai/$repo.git"|\
        "ssh://git@gitcode.com/hvat-ai/$repo"|"ssh://git@gitcode.com/hvat-ai/$repo.git") ;;
        *) fail "$repo origin is not the expected HVAT fork: $origin (no remote changed)" ;;
    esac
done

# Fetch every target first, including upstream history rewrites, without pruning.
for i in "${!repos[@]}"; do
    git -C "$code_dir/${repos[i]}" fetch origin \
        "+refs/heads/${branches[i]}:refs/remotes/origin/${branches[i]}"
done

for i in "${!repos[@]}"; do
    repo="${repos[i]}"; branch="${branches[i]}"; path="$code_dir/$repo"
    target="refs/remotes/origin/$branch"
    if git -C "$path" show-ref --verify --quiet "refs/heads/$branch"; then
        local_head="$(git -C "$path" rev-parse "refs/heads/$branch")"
        remote_head="$(git -C "$path" rev-parse "$target")"
        if [[ "$local_head" != "$remote_head" ]]; then
            backup="backup/pre-h3-$repo-$stamp"
            git -C "$path" branch "$backup" "$local_head"
            printf 'Saved %s: %s -> %s\n' "$repo" "$local_head" "$backup"
        fi
    fi
    if ! git -C "$path" symbolic-ref -q HEAD >/dev/null; then
        if [[ "$(git -C "$path" rev-parse HEAD)" != "$(git -C "$path" rev-parse "$target")" ]]; then
            git -C "$path" branch "backup/pre-h3-detached-$repo-$stamp" HEAD
        fi
    fi
    git -C "$path" switch --detach "$target"
    git -C "$path" branch -f "$branch" "$target"
    git -C "$path" switch "$branch"
    git -C "$path" branch --set-upstream-to="origin/$branch" "$branch"
done

printf '\n%-14s %-39s %s\n' REPOSITORY BRANCH HEAD
for repo in "${repos[@]}"; do
    path="$code_dir/$repo"
    head="$(git -C "$path" rev-parse HEAD)"
    [[ "$head" == "$(git -C "$path" rev-parse '@{upstream}')" ]] || fail "$repo HEAD differs from upstream"
    printf '%-14s %-39s %s\n' "$repo" "$(git -C "$path" branch --show-current)" "$head"
done
printf '\nAll three branch heads match GitCode. Existing other branches are preserved.\n'
printf 'Before building, initialize Aiter submodules:\n  git -C %q submodule update --init --recursive\n' "$code_dir/aiter"
