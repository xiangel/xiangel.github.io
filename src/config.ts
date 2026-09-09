/**
 * Internal resolved configuration used throughout the codebase.
 *
 * Prefer editing `astro-paper.config.ts` instead of this file. This module exists to
 * apply defaults and expose a fully-resolved config shape (`ResolvedAstroPaperConfig`).
 */
import userConfig from "@/astro-paper.config";
import type { GiscusConfig, ResolvedAstroPaperConfig } from "./types/config";
import {
  PUBLIC_GISCUS_CATEGORY_ID,
  PUBLIC_GOOGLE_SITE_VERIFICATION,
} from "astro:env/client";

export type { GiscusConfig };

const DEFAULT_OG_IMAGE = "default-og.jpg";

function resolveGiscusConfig(
  giscus: GiscusConfig | undefined
): GiscusConfig | null {
  if (!giscus?.enabled) return null;

  const categoryId = PUBLIC_GISCUS_CATEGORY_ID || giscus.categoryId;

  return {
    ...giscus,
    categoryId,
    mapping: giscus.mapping ?? "pathname",
    lang: giscus.lang ?? userConfig.site.lang ?? "zh-CN",
    reactionsEnabled: giscus.reactionsEnabled ?? true,
  };
}

const config: ResolvedAstroPaperConfig = {
  site: {
    ...userConfig.site,
    ogImage: userConfig.site.ogImage ?? DEFAULT_OG_IMAGE,
    lang: userConfig.site.lang ?? "en",
    timezone: userConfig.site.timezone ?? "UTC",
    dir: userConfig.site.dir ?? "ltr",
    googleVerification:
      userConfig.site.googleVerification || PUBLIC_GOOGLE_SITE_VERIFICATION,
  },
  posts: {
    perPage: userConfig.posts?.perPage ?? 4,
    perIndex: userConfig.posts?.perIndex ?? 4,
    scheduledPostMargin:
      userConfig.posts?.scheduledPostMargin ?? 15 * 60 * 1000,
  },
  features: {
    lightAndDarkMode: userConfig.features?.lightAndDarkMode ?? true,
    dynamicOgImage: userConfig.features?.dynamicOgImage ?? true,
    showArchives: userConfig.features?.showArchives ?? true,
    showBackButton: userConfig.features?.showBackButton ?? true,
    editPost: userConfig.features?.editPost ?? { enabled: false },
    search: userConfig.features?.search ?? "pagefind",
    giscus: resolveGiscusConfig(userConfig.features?.giscus),
    showPageViews: userConfig.features?.showPageViews ?? true,
  },
  socials: userConfig.socials ?? [],
  shareLinks: userConfig.shareLinks ?? [],
};

export default config;
