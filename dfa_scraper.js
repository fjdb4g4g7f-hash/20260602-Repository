/**
 * DFA (Department of Foreign Affairs) News Scraper
 *
 * Uses Wayback Machine archives to access dfa.gov.ph articles about WPS/SCS
 * (direct access is blocked by Cloudflare)
 *
 * Usage: node dfa_scraper.js
 * Output: dfa_WPS_data.json
 */

const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const CONFIG = {
    dateFrom: '2022-07-01',
    dateTo: '2026-05-31',
    outputFile: path.join(__dirname, 'dfa_WPS_data.json'),
    minTextLength: 100, // Minimum text length to consider valid
};

// WPS/SCS keywords for URL filtering (first pass) and content filtering (second pass)
const URL_WPS_KEYWORDS = [
    'west-philippine', 'west+philippine', 'west_philippine',
    'south-china-sea', 'south+china+sea',
    '/wps', '-wps', 'wps-', 'wps.',
    'maritime', 'arbitral', 'unclos',
    'kalayaan', 'pag-asa', 'ayungin',
    'fisheries', 'coast-guard', 'coastguard',
    'navy', 'naval', 'reef', 'pacific',
    'sovereignty', 'territorial',
    'chinese-vessel', 'china-sea',
    'sea-security', 'east-sea',
];

const CONTENT_WPS_KEYWORDS = [
    'west philippine sea', 'south china sea', 'wps',
    'arbitral award', 'unclos',
];

function sleep(ms) {
    return new Promise(r => setTimeout(r, ms));
}

/**
 * Download and extract text from a Wayback Machine archived DFA article
 */
async function fetchWaybackArticle(browser, url, timestamp) {
    const context = await browser.newContext({
        userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    });
    const page = await context.newPage();

    // Use the most recent timestamp from our data
    const wbUrl = `https://web.archive.org/web/${timestamp}/${url}`;

    try {
        await page.goto(wbUrl, { waitUntil: 'domcontentloaded', timeout: 30000 });
        await sleep(1000);

        const title = await page.title();

        // Extract all text content
        const text = await page.evaluate(() => {
            const body = document.body;
            if (!body) return '';

            // Remove Wayback Machine navigation elements
            const waybackBanner = document.querySelector('#wm-ipp-base, #wm-ipp, .wm-ipp, #donato, #donation-description');
            if (waybackBanner) waybackBanner.remove();

            // Get the actual article content
            const contentSelectors = [
                '.item-page', '#content', '.content',
                'article', '.article', '.news-wrapper',
                '.main-content', '#main-content',
                '.tileItem', '.news-single',
            ];

            for (const sel of contentSelectors) {
                const el = document.querySelector(sel);
                if (el && el.textContent.trim().length > 200) {
                    return el.textContent;
                }
            }

            // Fallback: get body text
            return body.textContent || '';
        });

        // Clean up text
        const cleanText = text
            .replace(/The Wayback Machine - .*?\n/, '')
            .replace(/__wm\.\w+\([^)]+\)/g, '')
            .replace(/https:\/\/web\.archive\.org\/web\/[^/\s]+\//g, '')
            .replace(/\s+/g, ' ')
            .trim();

        // Extract date from text (look for "DD Month YYYY" pattern)
        const dateMatch = cleanText.match(/\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b/i);

        // Also try ISO date
        const isoMatch = cleanText.match(/\b(\d{4}-\d{2}-\d{2})\b/);

        let articleDate = '';
        if (dateMatch) {
            try {
                const d = new Date(dateMatch[1]);
                if (!isNaN(d.getTime())) {
                    articleDate = d.toISOString();
                }
            } catch(e) {}
        }

        return {
            title: title.replace(/- Philippine Information Agency|- DFA|- Department of Foreign Affairs/i, '').trim(),
            text: cleanText,
            articleDate: articleDate,
            wbTimestamp: timestamp,
        };

    } catch (err) {
        return null;
    } finally {
        await page.close();
        await context.close();
    }
}

/**
 * Check if URL slug suggests WPS/SCS content
 */
function isWpsUrl(url) {
    const slug = url.toLowerCase();
    return URL_WPS_KEYWORDS.some(kw => slug.includes(kw));
}

/**
 * Check if text content contains WPS/SCS keywords
 */
function hasWpsContent(text) {
    const lower = text.toLowerCase();
    const matches = CONTENT_WPS_KEYWORDS.filter(kw => lower.includes(kw));
    return matches;
}

async function main() {
    console.log('=== DFA News Scraper (via Wayback Machine) ===');

    // Step 1: Load all DFA URLs from both sections
    const articleUrls = new Map(); // url -> latest timestamp

    // Load from the article URLs file
    if (fs.existsSync(path.join(__dirname, 'dfa_article_urls.txt'))) {
        const data = fs.readFileSync(path.join(__dirname, 'dfa_article_urls.txt'), 'utf-8');
        for (const line of data.trim().split('\n')) {
            const [ts, url] = line.split('\t');
            if (ts && url && ts >= '202207' && ts <= '202605') {
                const cleanUrl = url.split('?')[0].replace(/\/$/, '');
                // Only individual articles (not listing pages)
                if (cleanUrl.includes('postsupdate') && !cleanUrl.endsWith('postsupdate')) {
                    const slug = cleanUrl.split('/').pop();
                    if (slug && slug.length > 10) { // has a meaningful slug
                        if (!articleUrls.has(cleanUrl) || ts > articleUrls.get(cleanUrl)) {
                            articleUrls.set(cleanUrl, ts);
                        }
                    }
                }
            }
        }
    }

    // Also load from statements section
    const statementsFile = path.join(__dirname, 'dfa_statements_urls.txt');
    if (fs.existsSync(statementsFile)) {
        const data = fs.readFileSync(statementsFile, 'utf-8');
        for (const line of data.trim().split('\n')) {
            const [ts, url] = line.split('\t');
            if (ts && url) {
                const cleanUrl = url.split('?')[0].replace(/\/$/, '');
                if (!articleUrls.has(cleanUrl) || ts > articleUrls.get(cleanUrl)) {
                    articleUrls.set(cleanUrl, ts);
                }
            }
        }
    }

    console.log(`\nTotal articles collected: ${articleUrls.size}`);

    // Step 2: Filter by URL keywords
    const candidates = [];
    for (const [url, ts] of articleUrls) {
        if (isWpsUrl(url)) {
            const slug = url.split('/').pop();
            candidates.push({ url, ts, slug });
        }
    }

    console.log(`URL-filtered candidates (WPS keywords in URL): ${candidates.length}`);

    // Step 3: Fetch each candidate from Wayback Machine
    const browser = await chromium.launch({
        headless: true,
        args: ['--no-sandbox', '--disable-blink-features=AutomationControlled']
    });

    const results = [];
    try {
        for (let i = 0; i < candidates.length; i++) {
            const { url, ts, slug } = candidates[i];
            process.stdout.write(`[${i+1}/${candidates.length}] ${slug.substring(0, 50)}... `);

            const data = await fetchWaybackArticle(browser, url, ts);
            if (data && data.text.length > CONFIG.minTextLength) {
                const kwMatches = hasWpsContent(data.text);
                if (kwMatches.length > 0) {
                    results.push({
                        title: data.title,
                        url: url,
                        text: data.text.substring(0, 50000),
                        date: data.articleDate,
                        wbTimestamp: data.wbTimestamp,
                        keywords: kwMatches,
                        source: 'DFA',
                        country: 'Philippines',
                    });
                    console.log(`✓ (${data.title.substring(0, 40)})`);
                } else {
                    console.log(`✗ (no WPS content)`);
                }
            } else {
                console.log(`✗ (empty/failed)`);
            }

            // Be polite to Wayback Machine
            await sleep(2000);
        }

        // Step 4: Filter by date range
        const fromDate = new Date(CONFIG.dateFrom);
        const toDate = new Date(CONFIG.dateTo);
        const inRange = results.filter(a => {
            if (!a.date) return true; // include if no date found
            const d = new Date(a.date);
            return d >= fromDate && d <= toDate;
        });

        console.log(`\n=== Results ===`);
        console.log(`Total fetched: ${candidates.length}`);
        console.log(`With WPS content: ${results.length}`);
        console.log(`In date range: ${inRange.length}`);

        // Step 5: Save
        const output = {
            config: CONFIG,
            scrapedAt: new Date().toISOString(),
            totalCandidates: candidates.length,
            totalArticles: results.length,
            inRangeCount: inRange.length,
            articles: inRange,
        };

        fs.writeFileSync(CONFIG.outputFile, JSON.stringify(output, null, 2), 'utf-8');
        console.log(`\nSaved to ${CONFIG.outputFile}`);

    } catch (err) {
        console.error('Fatal:', err.message);
    } finally {
        await browser.close();
    }
}

// Main
main().catch(console.error);
