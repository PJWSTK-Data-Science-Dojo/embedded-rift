# Betting Site Scraper Build Framework - Quick Reference

### Step 1: API Investigation
Check if the site has a public/undocumented API by inspecting network requests in Chrome DevTools. If API returns data → use API approach (faster, more reliable). If API returns empty/errors → go to Step 2. Save the raw API response for reference.

### Step 2: HTML Structure Analysis
Open Chrome DevTools → Inspect Elements on target betting data (teams, odds, dates). Document CSS selectors, DOM structure, and whether content is static HTML or JavaScript-rendered. Take screenshot of page for reference.

### Step 3: Technology Choice
If HTML is static (view page source = same as DevTools) → use `requests` + `BeautifulSoup`. If content is JavaScript-rendered (view page source differs from DevTools) → use `Playwright` for browser automation.

### Step 4: Create Scraper Class
Build a class extending `BaseBettingScraper` with: (a) `get_html_content()` - fetch raw HTML, (b) `parse_matches()` - extract team/date/odds using regex/BeautifulSoup, (c) `validate_data()` - check all required fields exist.

### Step 5: Map to Data Model
Convert raw data to `BettingMatch` dataclass with `BettingMarket` and `BettingOdds` objects (same structure for all sites). Return `list[BettingMatch]` from `get_matches()` method.


## Checklist Before Production

- [ ] API/HTML approach decided (API first, then HTML fallback)
- [ ] Selectors documented and tested manually
- [ ] Error handling for network failures
- [ ] Data validation on all required fields
