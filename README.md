# Xiao Constantine's Blog

A personal blog about AI, systems engineering, and machine learning.

**Live at:** https://xiaoConstantine.github.io

## Setup

Built on [Jekyll](https://jekyllrb.com/) + [Poole](https://github.com/poole/poole) theme.

### Local Development

```bash
bundle install
bundle exec jekyll serve
```

Then visit `http://localhost:4000`

## Adding Posts

Create a new file in `_posts/` with format:

```
YYYY-MM-DD-title.md
```

Example:
```markdown
---
layout: post
title: My First Post
---

Your content here...
```

## Configuration

Edit `_config.yml` to customize:
- Title and tagline
- Author info
- Site URL
- Timezone

## Deployment

Push to GitHub and it auto-deploys via GitHub Pages.

```bash
git add -A
git commit -m "Your message"
git push origin main
```

## License

MIT License
