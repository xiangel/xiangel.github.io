#!/usr/bin/env bash
set -euo pipefail

OWNER="${GISCUS_REPO_OWNER:-xiangel}"
NAME="${GISCUS_REPO_NAME:-xiangel.github.io}"
CATEGORY="${GISCUS_CATEGORY:-Announcements}"

echo "Fetching Giscus config for ${OWNER}/${NAME}..."
echo

RESPONSE="$(gh api graphql -f query="
query {
  repository(owner: \"${OWNER}\", name: \"${NAME}\") {
    id
    hasDiscussionsEnabled
    discussionCategories(first: 20) {
      nodes { id name slug }
    }
  }
}")"

HAS_DISCUSSIONS="$(echo "$RESPONSE" | jq -r '.data.repository.hasDiscussionsEnabled')"
REPO_ID="$(echo "$RESPONSE" | jq -r '.data.repository.id')"
CATEGORY_ID="$(echo "$RESPONSE" | jq -r --arg cat "$CATEGORY" '.data.repository.discussionCategories.nodes[] | select(.name == $cat) | .id' | head -n1)"

echo "Repository ID : ${REPO_ID}"
echo "Discussions   : ${HAS_DISCUSSIONS}"
echo

if [[ "$HAS_DISCUSSIONS" != "true" ]]; then
  echo "Discussions is not enabled yet."
  echo
  echo "Please complete these steps first:"
  echo "  1. Open https://github.com/${OWNER}/${NAME}/settings"
  echo "  2. Check 'Discussions' under Features"
  echo "  3. Install Giscus app: https://github.com/apps/giscus"
  echo "  4. Re-run: npm run giscus:setup"
  exit 1
fi

echo "Available categories:"
echo "$RESPONSE" | jq -r '.data.repository.discussionCategories.nodes[] | "  - \(.name) (\(.slug)): \(.id)"'
echo

if [[ -z "$CATEGORY_ID" ]]; then
  echo "Category '${CATEGORY}' not found."
  echo "Create it in GitHub Discussions or set GISCUS_CATEGORY=General"
  exit 1
fi

echo "Selected category: ${CATEGORY}"
echo "Category ID      : ${CATEGORY_ID}"
echo
echo "Add this to astro-paper.config.ts -> features.giscus.categoryId:"
echo "  categoryId: \"${CATEGORY_ID}\","
echo
echo "Or create .env with:"
echo "  PUBLIC_GISCUS_CATEGORY_ID=${CATEGORY_ID}"
