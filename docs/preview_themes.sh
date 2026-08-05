#!/usr/bin/env bash
# Build the docs once per candidate theme so they can be compared side by side.
#
#   bash docs/preview_themes.sh          # build all, print the URLs
#   bash docs/preview_themes.sh --serve  # build all and serve on :8000
#
# Each theme lands in docs/_preview/<theme>/. Nothing here touches conf.py:
# the theme is injected with -D, so the committed configuration is untouched.

set -euo pipefail
cd "$(dirname "$0")"

THEMES=(furo renku sphinx_book_theme pydata_sphinx_theme sphinx_rtd_theme)
PYTHON="${PYTHON:-python}"

rm -rf _preview
for theme in "${THEMES[@]}"; do
    echo "building: $theme"
    # theme-specific options in conf.py will not apply to the others, so drop them
    $PYTHON -m sphinx -b html -q \
        -D html_theme="$theme" \
        -D html_theme_options.=  \
        source "_preview/$theme" 2>/dev/null \
      || $PYTHON -m sphinx -b html -q -D html_theme="$theme" source "_preview/$theme" \
      || echo "  ! $theme failed to build"
done

echo
echo "built:"
for theme in "${THEMES[@]}"; do
    [ -f "_preview/$theme/index.html" ] && echo "  $theme  ->  $(pwd)/_preview/$theme/index.html"
done

if [ "${1:-}" = "--serve" ]; then
    echo
    echo "serving on http://localhost:8000/  (Ctrl-C to stop)"
    cd _preview && $PYTHON -m http.server 8000
fi
