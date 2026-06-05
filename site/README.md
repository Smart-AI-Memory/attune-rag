# attune-rag.dev

Static landing site for [attune-rag.dev](https://attune-rag.dev),
deployed on Vercel from this `site/` directory.

Modeled on attune-ai.dev: a single hand-authored `index.html` plus
reproducible asset builders.

## Rebuild assets

```bash
python build_og.py        # og.png (1200x630 social card)
python build_favicon.py   # favicon.ico + favicon.svg
python build_sitemap.py   # sitemap.xml
```

`brand.css` is the shared family brand. `vercel.json` sets
`cleanUrls`, security headers, and the og.png cache header.

## Deploy

Vercel project → Root Directory `site/`, enable Web Analytics, bind
`attune-rag.dev` (apex-primary, matching the page's canonical).
